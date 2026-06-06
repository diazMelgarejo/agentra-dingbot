"""
ml/freqai_bridge.py  —  Step 5: FreqAI ML Bridge — Orchestration
==================================================================
Ties features + labels + model into a single lifecycle and exposes one
high-level call: `generate_ml_signal(df)` → (Signal, confidence, prob_up, meta).

FreqAI-style behaviours
-----------------------
* Adaptive retraining: a model is (re)trained when none exists on disk, when
  the persisted model is older than `retrain_interval_cycles` cycles, or when
  `force_retrain=True`. A sidecar JSON tracks the cycle counter + timestamp.
* Persistence: models are joblib-saved under `model_dir`, keyed by symbol so
  BTC and ETH keep independent models.
* In-process cache: within a session the loaded model is memoised so repeated
  pipeline invocations don't retrain (keeps the LangGraph loop fast).
* Self-healing: any failure degrades to the heuristic backend rather than
  raising — the trading loop must never crash on ML.

The bridge is intentionally framework-light: it reproduces the *useful* part
of FreqTrade/FreqAI (rolling self-adaptive retraining on engineered features)
without importing the full FreqTrade bot, matching the SuperBot's modular,
testable design.
"""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import structlog

from core.state import Signal
from ml.features import FEATURE_COLUMNS, build_features, latest_feature_row
from ml.labels import align_xy, make_labels
from ml.model import MLSignalModel

logger = structlog.get_logger(__name__)

# In-process model cache: {symbol: (MLSignalModel, meta_dict)}
_MODEL_CACHE: dict[str, tuple[MLSignalModel, dict[str, Any]]] = {}


class FreqAIBridge:
    """Manages the ML signal lifecycle for one or more symbols."""

    def __init__(self, settings=None):
        if settings is None:
            from core.config import get_settings
            settings = get_settings()
        self.cfg = settings.ml
        os.makedirs(self.cfg.model_dir, exist_ok=True)

    # ── Paths ───────────────────────────────────────────────────────────────────

    def _safe(self, symbol: str) -> str:
        return symbol.replace("/", "_").replace(":", "_")

    def _model_path(self, symbol: str) -> str:
        return os.path.join(self.cfg.model_dir, f"ml_{self._safe(symbol)}.joblib")

    def _meta_path(self, symbol: str) -> str:
        return os.path.join(self.cfg.model_dir, f"ml_{self._safe(symbol)}.meta.json")

    # ── Metadata ──────────────────────────────────────────────────────────────────

    def _load_meta(self, symbol: str) -> dict[str, Any]:
        p = self._meta_path(symbol)
        if os.path.exists(p):
            try:
                return json.load(open(p))
            except Exception:
                return {}
        return {}

    def _save_meta(self, symbol: str, meta: dict[str, Any]) -> None:
        try:
            with open(self._meta_path(symbol), "w") as _f:
                json.dump(meta, _f, indent=2)
        except Exception as exc:
            logger.warning("ml_meta_save_failed", error=str(exc))

    # ── Retrain decision ───────────────────────────────────────────────────────────

    def _needs_retrain(self, symbol: str, force: bool) -> bool:
        if force:
            return True
        if not os.path.exists(self._model_path(symbol)):
            return True
        meta = self._load_meta(symbol)
        cycles = int(meta.get("cycles_since_train", 0))
        return cycles >= self.cfg.retrain_interval_cycles

    # ── Training ────────────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, symbol: str) -> MLSignalModel | None:
        """Train (or retrain) a model on the supplied history and persist it."""
        feats = build_features(df)
        labels = make_labels(df, horizon=self.cfg.label_horizon,
                             deadzone=self.cfg.label_deadzone)
        X, y = align_xy(feats, labels)

        if len(X) < self.cfg.min_train_samples:
            logger.warning("ml_insufficient_train_data",
                           have=len(X), need=self.cfg.min_train_samples, symbol=symbol)
            return None

        model = MLSignalModel(model_type=self.cfg.model_type,
                              random_state=self.cfg.random_state).fit(X, y)
        model.save(self._model_path(symbol))

        meta = {
            "symbol": symbol,
            "backend": model.backend,
            "n_train_samples": model.n_train_samples,
            "trained_at": datetime.now(UTC).isoformat(),
            "cycles_since_train": 0,
            "horizon": self.cfg.label_horizon,
        }
        self._save_meta(symbol, meta)
        _MODEL_CACHE[symbol] = (model, meta)
        logger.info("ml_trained", symbol=symbol, backend=model.backend,
                    samples=model.n_train_samples)
        return model

    # ── Model retrieval (cache → disk → train) ──────────────────────────────────────

    def _get_model(self, df: pd.DataFrame, symbol: str,
                   force_retrain:
                       bool) -> tuple[MLSignalModel, dict[str, Any]] | None:
        if self._needs_retrain(symbol, force_retrain):
            model = self.train(df, symbol)
            if model is None:
                return None
            return _MODEL_CACHE[symbol]

        # Use in-process cache if present
        if symbol in _MODEL_CACHE:
            model, meta = _MODEL_CACHE[symbol]
        else:
            try:
                model = MLSignalModel.load(self._model_path(symbol))
                meta = self._load_meta(symbol)
                _MODEL_CACHE[symbol] = (model, meta)
            except Exception as exc:
                logger.warning("ml_load_failed_retraining", error=str(exc), symbol=symbol)
                model = self.train(df, symbol)
                if model is None:
                    return None
                return _MODEL_CACHE[symbol]

        # Increment cycle counter (adaptive-retrain odometer)
        meta["cycles_since_train"] = int(meta.get("cycles_since_train", 0)) + 1
        self._save_meta(symbol, meta)
        _MODEL_CACHE[symbol] = (model, meta)
        return model, meta

    # ── Public API ────────────────────────────────────────────────────────────────

    def generate_ml_signal(
        self,
        df: pd.DataFrame,
        symbol: str = "BTC/USDT",
        force_retrain:
            bool = False,
    ) -> tuple[Signal, float, float | None, dict[str, Any]]:
        """
        Produce a directional ML signal for the latest bar.

        Returns
        -------
        (signal, confidence, prob_up, meta)
          signal     : Signal enum
          confidence : |prob_up - 0.5| * 2   ∈ [0,1]
          prob_up    : model P(up) for the latest bar, or None on failure
          meta       : dict with backend, n_train_samples, top_features, trained_at,
                       n_features, reasoning
        """
        empty_meta = {
            "backend": "none", "n_train_samples": 0, "top_features": [],
            "trained_at": None, "n_features": 0,
            "reasoning": "ML disabled or insufficient data",
        }

        if not self.cfg.enabled:
            return Signal.NEUTRAL, 0.0, None, {**empty_meta, "reasoning": "ML disabled"}

        if df is None or len(df) < (self.cfg.min_train_samples + self.cfg.label_horizon):
            return Signal.NEUTRAL, 0.0, None, {
                **empty_meta,
                "reasoning": f"need ≥{self.cfg.min_train_samples + self.cfg.label_horizon} bars",
            }

        try:
            got = self._get_model(df, symbol, force_retrain)
            if got is None:
                return Signal.NEUTRAL, 0.0, None, {**empty_meta,
                                                   "reasoning": "training skipped (insufficient labeled rows)"}
            model, meta = got

            row = latest_feature_row(df)
            if row.empty:
                return Signal.NEUTRAL, 0.0, None, {**empty_meta, "reasoning": "no feature row"}

            # Ensure column order matches training contract
            row = row.reindex(columns=model.feature_names or FEATURE_COLUMNS)
            prob_up = float(model.predict_proba_up(row)[0])
            sig = self._prob_to_signal(prob_up)
            conf = round(abs(prob_up - 0.5) * 2.0, 4)

            out_meta = {
                "backend": model.backend,
                "n_train_samples": model.n_train_samples,
                "top_features": model.feature_importance(top_k=5),
                "trained_at": meta.get("trained_at"),
                "n_features": len(model.feature_names),
                "reasoning": (
                    f"{model.backend} P(up)={prob_up:.3f} → {sig.value} "
                    f"(trained on {model.n_train_samples} samples)"
                ),
            }
            logger.info("ml_signal", symbol=symbol, prob_up=round(prob_up, 3),
                        signal=sig.value, backend=model.backend)
            return sig, conf, prob_up, out_meta

        except Exception as exc:
            logger.error("ml_signal_failed", error=str(exc), symbol=symbol)
            return Signal.NEUTRAL, 0.0, None, {**empty_meta, "reasoning": f"error: {exc}"}

    # ── Threshold mapping ───────────────────────────────────────────────────────────

    def _prob_to_signal(self, p: float) -> Signal:
        c = self.cfg
        if p >= c.prob_strong_buy:
            return Signal.STRONG_BUY
        if p >= c.prob_buy:
            return Signal.BUY
        if p <= c.prob_strong_sell:
            return Signal.STRONG_SELL
        if p <= c.prob_sell:
            return Signal.SELL
        return Signal.NEUTRAL


def clear_model_cache() -> None:
    """Test helper: drop the in-process model cache."""
    _MODEL_CACHE.clear()
