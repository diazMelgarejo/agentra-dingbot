"""
ml/model.py  —  Step 5: FreqAI ML Bridge — Model
===================================================
A directional classifier with a three-tier backend, chosen at fit time:

  1. LightGBM           (LGBMClassifier)        — primary, if installed
  2. sklearn HistGBM    (HistGradientBoosting)  — fallback, NaN-tolerant, fast
  3. heuristic          (momentum sign)         — last-resort, no training

The same `MLSignalModel` interface wraps all three so the rest of the system
(bridge, agent, tests) is backend-agnostic — exactly mirroring the
TA-Lib → pandas-ta fallback pattern used in indicators.py.

Persistence is via joblib; a sidecar dict records backend + train metadata.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


def _select_backend(preferred: str) -> str:
    """Resolve the actual backend given a preference and what's installed."""
    if preferred == "heuristic":
        return "heuristic"
    if preferred in ("lightgbm", "auto"):
        try:
            import lightgbm  # noqa: F401
            return "lightgbm"
        except ImportError:
            pass
    # sklearn HistGradientBoosting is in core sklearn (always available here)
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: F401
        return "sklearn_hgb"
    except ImportError:
        return "heuristic"


class MLSignalModel:
    """
    Directional (up/down) classifier.

    Usage
    -----
        m = MLSignalModel(model_type="lightgbm", random_state=42)
        m.fit(X, y)                 # X: feature DataFrame, y: 0/1 Series
        p = m.predict_proba_up(X)   # np.ndarray of P(up) per row
        m.save(path); MLSignalModel.load(path)
    """

    def __init__(self, model_type: str = "lightgbm", random_state: int = 42):
        self.requested_type = model_type
        self.backend = _select_backend(model_type)
        self.random_state = random_state
        self._model: Any = None
        self.feature_names: List[str] = []
        self.n_train_samples: int = 0
        self._heuristic_feature = "roll_mean_5"  # sign → up/down

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MLSignalModel":
        self.feature_names = list(X.columns)
        self.n_train_samples = int(len(X))

        # Degenerate cases → force heuristic so we never crash on tiny/one-class data
        if len(X) == 0 or y.nunique() < 2:
            logger.warning("ml_fit_degenerate",
                           n=len(X), classes=int(y.nunique()) if len(y) else 0,
                           backend="heuristic")
            self.backend = "heuristic"
            self._model = None
            return self

        if self.backend == "lightgbm":
            import lightgbm as lgb
            self._model = lgb.LGBMClassifier(
                n_estimators=200, num_leaves=31, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=self.random_state, n_jobs=1, verbose=-1,
            )
            self._model.fit(X.values, y.values)

        elif self.backend == "sklearn_hgb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            self._model = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=self.random_state,
            )
            # HistGBM tolerates NaN natively — no imputation needed
            self._model.fit(X.values, y.values)

        else:  # heuristic — nothing to train
            self._model = None

        logger.info("ml_fit_done", backend=self.backend, n=self.n_train_samples,
                    features=len(self.feature_names))
        return self

    # ── Inference ───────────────────────────────────────────────────────────────

    def predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(up) ∈ [0,1] for each row of X."""
        if self.backend == "heuristic" or self._model is None:
            return self._heuristic_proba(X)

        proba = self._model.predict_proba(X.values)
        # class order: model.classes_ — find the column for label 1.0
        classes = list(getattr(self._model, "classes_", [0.0, 1.0]))
        up_idx = classes.index(1.0) if 1.0 in classes else (len(classes) - 1)
        return np.clip(proba[:, up_idx], 0.0, 1.0)

    def _heuristic_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Momentum heuristic: map short-term mean return to a soft probability."""
        if self._heuristic_feature in X.columns:
            mom = X[self._heuristic_feature].fillna(0.0).values
        else:
            mom = np.zeros(len(X))
        # squash with a logistic; scale chosen so ~1% mean-return → ~0.62
        return np.clip(1.0 / (1.0 + np.exp(-mom * 50.0)), 0.0, 1.0)

    # ── Introspection ─────────────────────────────────────────────────────────────

    def feature_importance(self, top_k: int = 5) -> List[Dict[str, float]]:
        """Return the top-k features by importance as [{"feature","importance"}]."""
        if self._model is None or not self.feature_names:
            return []
        imp = None
        if self.backend == "lightgbm":
            imp = np.asarray(self._model.feature_importances_, dtype=float)
        elif self.backend == "sklearn_hgb":
            # HistGBM has no native feature_importances_; use permutation-free
            # proxy via the model's training loss is unavailable, so fall back
            # to equal weighting only when truly unknown.
            fi = getattr(self._model, "feature_importances_", None)
            if fi is not None:
                imp = np.asarray(fi, dtype=float)
        if imp is None or imp.sum() == 0:
            return []
        imp = imp / imp.sum()
        order = np.argsort(imp)[::-1][:top_k]
        return [{"feature": self.feature_names[i], "importance": round(float(imp[i]), 4)}
                for i in order]

    # ── Persistence ───────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({
            "backend": self.backend,
            "requested_type": self.requested_type,
            "random_state": self.random_state,
            "feature_names": self.feature_names,
            "n_train_samples": self.n_train_samples,
            "model": self._model,
        }, path)
        logger.info("ml_model_saved", path=path, backend=self.backend)

    @classmethod
    def load(cls, path: str) -> "MLSignalModel":
        import joblib
        data = joblib.load(path)
        m = cls(model_type=data.get("requested_type", "lightgbm"),
                random_state=data.get("random_state", 42))
        m.backend = data["backend"]
        m.feature_names = data["feature_names"]
        m.n_train_samples = data.get("n_train_samples", 0)
        m._model = data["model"]
        return m
