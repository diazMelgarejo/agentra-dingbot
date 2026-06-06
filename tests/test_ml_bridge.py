"""
tests/test_ml_bridge.py  —  Step 5: FreqAI ML Bridge Tests
============================================================
Coverage:
  A. features  — shape, columns, determinism, NaN handling
  B. labels    — direction sign, horizon, deadzone, alignment
  C. model     — trains & learns separable data, proba bounds, save/load,
                 degenerate→heuristic, feature importance
  D. bridge    — generate_ml_signal contract, adaptive retrain cadence,
                 persistence, threshold→signal mapping, self-healing
  E. agent     — LangGraph node returns {"ml": MLSnapshot}
  short/missing
                 data → None
                 exception-guarded
  F. pipeline  — graph compiles with ml_analyst
  full ainvoke completes
                 fetch called once
                 ml failure doesn't block the pipeline

LightGBM is optional: backend-specific assertions skip when it isn't installed,
mirroring the TA-Lib equivalence-skip pattern. The sklearn HistGBM path is the
one exercised live in CI/sandbox.
"""
from __future__ import annotations

import importlib
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from core.state import Signal

# ── Fixtures / factories ───────────────────────────────────────────────────────

def _ohlcv(n: int = 400, *, regime: str = "two_phase", seed: int = 0,
           freq: str = "1h") -> pd.DataFrame:
    """
    Deterministic OHLCV.
      regime="two_phase" : up-drift then down-drift (gives both label classes)
      regime="up"        : steady up-drift
      regime="flat"      : pure noise
    """
    rng = np.random.default_rng(seed)
    if regime == "two_phase":
        drift = np.concatenate([np.full(n // 2, 0.002), np.full(n - n // 2, -0.002)])
    elif regime == "up":
        drift = np.full(n, 0.0015)
    else:
        drift = np.zeros(n)
    close = 50_000 * np.cumprod(1 + drift + rng.normal(0, 0.004, n))
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({
        "open": close * 0.999, "high": close * 1.002,
        "low": close * 0.998, "close": close,
        "volume": rng.uniform(10, 100, n),
    }, index=idx)


def _separable(n: int = 300, seed: int = 1) -> pd.DataFrame:
    """
    Construct data where the next-bar direction is strongly predictable from
    recent momentum — so a working classifier must beat 0.5 on a holdout.
    """
    rng = np.random.default_rng(seed)
    # Build returns where sign persists in blocks (autocorrelated) → learnable
    blocks = rng.choice([-1, 1], size=n // 10)
    sign = np.repeat(blocks, 10)[:n]
    rets = sign * np.abs(rng.normal(0.003, 0.001, n))
    close = 50_000 * np.cumprod(1 + rets)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": close * 0.999, "high": close * 1.001,
        "low": close * 0.999, "close": close,
        "volume": rng.uniform(10, 50, n),
    }, index=idx)


@pytest.fixture(autouse=True)
def _isolate_models(monkeypatch):
    """Point every test's model_dir at a fresh tmp dir and clear caches."""
    d = tempfile.mkdtemp(prefix="ml_test_")
    monkeypatch.setenv("ML_MODEL_DIR", d)
    monkeypatch.setenv("ML_MIN_TRAIN_SAMPLES", "80")
    from core.config import get_settings
    get_settings.cache_clear()
    from ml.freqai_bridge import clear_model_cache
    clear_model_cache()
    yield
    get_settings.cache_clear()
    clear_model_cache()


_HAS_LGBM = importlib.util.find_spec("lightgbm") is not None


# ── A. Features ─────────────────────────────────────────────────────────────────

class TestFeatures:

    def test_columns_exact(self):
        from ml.features import FEATURE_COLUMNS, build_features
        feats = build_features(_ohlcv(200))
        assert list(feats.columns) == FEATURE_COLUMNS

    def test_row_count_matches_input(self):
        from ml.features import build_features
        df = _ohlcv(200)
        assert len(build_features(df)) == len(df)

    def test_deterministic(self):
        from ml.features import build_features
        df = _ohlcv(200, seed=7)
        a = build_features(df)
        b = build_features(df)
        pd.testing.assert_frame_equal(a, b)

    def test_empty_input_returns_empty_with_columns(self):
        from ml.features import FEATURE_COLUMNS, build_features
        out = build_features(pd.DataFrame())
        assert list(out.columns) == FEATURE_COLUMNS
        assert len(out) == 0

    def test_latest_row_has_no_nan(self):
        from ml.features import latest_feature_row
        row = latest_feature_row(_ohlcv(200))
        assert len(row) == 1
        assert not row.isna().any().any(), "latest feature row must be NaN-free"

    def test_early_rows_have_nan(self):
        from ml.features import build_features
        feats = build_features(_ohlcv(200))
        # roll_std_10 / vol_z_20 warm-up guarantees NaNs near the top
        assert feats.iloc[0].isna().any()


# ── B. Labels ───────────────────────────────────────────────────────────────────

class TestLabels:

    def test_values_are_binary_or_nan(self):
        from ml.labels import make_labels
        lab = make_labels(_ohlcv(200), horizon=3, deadzone=0.0)
        uniq = set(pd.unique(lab.dropna()))
        assert uniq <= {0.0, 1.0}

    def test_last_horizon_rows_nan(self):
        from ml.labels import make_labels
        h = 3
        lab = make_labels(_ohlcv(100), horizon=h, deadzone=0.0)
        assert lab.iloc[-h:].isna().all()

    def test_direction_sign_correct(self):
        from ml.labels import make_labels
        # Strictly increasing close → every forward return > 0 → all labels 1
        idx = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
        close = pd.Series(np.linspace(100, 200, 50), index=idx)
        df = pd.DataFrame({"open": close, "high": close, "low": close,
                           "close": close, "volume": 1.0}, index=idx)
        lab = make_labels(df, horizon=1, deadzone=0.0)
        assert (lab.dropna() == 1.0).all()

    def test_deadzone_drops_small_moves(self):
        from ml.labels import make_labels
        # Tiny oscillation inside a wide deadzone → all NaN (no trades)
        idx = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
        close = pd.Series(100 + np.sin(np.arange(50)) * 0.01, index=idx)
        df = pd.DataFrame({"open": close, "high": close, "low": close,
                           "close": close, "volume": 1.0}, index=idx)
        lab = make_labels(df, horizon=1, deadzone=0.05)  # 5% band
        assert lab.dropna().empty

    def test_align_xy_drops_nan_rows(self):
        from ml.features import build_features
        from ml.labels import align_xy, make_labels
        df = _ohlcv(200)
        X, y = align_xy(build_features(df), make_labels(df, horizon=3))
        assert len(X) == len(y)
        assert not X.isna().any().any()
        assert not y.isna().any()
        assert len(X) > 0


# ── C. Model ────────────────────────────────────────────────────────────────────

class TestModel:

    def _xy(self, df):
        from ml.features import build_features
        from ml.labels import align_xy, make_labels
        return align_xy(build_features(df), make_labels(df, horizon=1))

    def test_fit_predict_proba_in_range(self):
        from ml.model import MLSignalModel
        X, y = self._xy(_ohlcv(300))
        m = MLSignalModel(model_type="auto", random_state=0).fit(X, y)
        p = m.predict_proba_up(X)
        assert p.min() >= 0.0 and p.max() <= 1.0

    def test_learns_separable_holdout(self):
        """Train/test split: accuracy on held-out separable data must beat 0.5."""
        from ml.model import MLSignalModel
        X, y = self._xy(_separable(400))
        cut = int(len(X) * 0.7)
        Xtr, ytr = X.iloc[:cut], y.iloc[:cut]
        Xte, yte = X.iloc[cut:], y.iloc[cut:]
        if ytr.nunique() < 2 or yte.nunique() < 2:
            pytest.skip("split produced one-class fold")
        m = MLSignalModel(model_type="auto", random_state=0).fit(Xtr, ytr)
        pred = (m.predict_proba_up(Xte) >= 0.5).astype(float)
        acc = (pred == yte.values).mean()
        assert acc > 0.5, f"holdout accuracy {acc:.2f} not better than chance"

    def test_save_load_roundtrip(self):
        from ml.model import MLSignalModel
        X, y = self._xy(_ohlcv(300))
        m = MLSignalModel(model_type="auto", random_state=0).fit(X, y)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.joblib")
            m.save(path)
            m2 = MLSignalModel.load(path)
        np.testing.assert_allclose(
            m.predict_proba_up(X.iloc[-5:]),
            m2.predict_proba_up(X.iloc[-5:]),
            rtol=1e-6,
        )
        assert m2.backend == m.backend
        assert m2.feature_names == m.feature_names

    def test_degenerate_one_class_forces_heuristic(self):
        from ml.model import MLSignalModel
        X, _ = self._xy(_ohlcv(300))
        y_one = pd.Series(np.ones(len(X)), index=X.index)  # single class
        m = MLSignalModel(model_type="auto").fit(X, y_one)
        assert m.backend == "heuristic"
        p = m.predict_proba_up(X.iloc[-3:])
        assert p.min() >= 0.0 and p.max() <= 1.0

    def test_empty_fit_forces_heuristic(self):
        from ml.features import FEATURE_COLUMNS
        from ml.model import MLSignalModel
        empty = pd.DataFrame(columns=FEATURE_COLUMNS)
        m = MLSignalModel(model_type="auto").fit(empty, pd.Series(dtype=float))
        assert m.backend == "heuristic"

    def test_heuristic_backend_explicit(self):
        from ml.model import MLSignalModel
        X, y = self._xy(_ohlcv(200))
        m = MLSignalModel(model_type="heuristic").fit(X, y)
        assert m.backend == "heuristic"
        # positive momentum row → proba ≥ 0.5; negative → ≤ 0.5
        row_pos = X.iloc[[-1]].copy()
        row_pos["roll_mean_5"] = 0.05
        row_neg = X.iloc[[-1]].copy()
        row_neg["roll_mean_5"] = -0.05
        assert m.predict_proba_up(row_pos)[0] >= 0.5
        assert m.predict_proba_up(row_neg)[0] <= 0.5

    def test_feature_importance_length(self):
        from ml.model import MLSignalModel
        X, y = self._xy(_ohlcv(300))
        m = MLSignalModel(model_type="auto", random_state=0).fit(X, y)
        imp = m.feature_importance(top_k=5)
        assert isinstance(imp, list)
        assert len(imp) <= 5
        for entry in imp:
            assert "feature" in entry and "importance" in entry

    @pytest.mark.skipif(not _HAS_LGBM, reason="lightgbm not installed")
    def test_lightgbm_backend_selected_when_available(self):
        from ml.model import MLSignalModel
        X, y = self._xy(_ohlcv(300))
        m = MLSignalModel(model_type="lightgbm", random_state=0).fit(X, y)
        assert m.backend == "lightgbm"
        imp = m.feature_importance(top_k=5)
        assert len(imp) >= 1  # LightGBM exposes native importances


# ── D. Bridge ───────────────────────────────────────────────────────────────────

class TestBridge:

    def test_generate_signal_contract(self):
        from ml.freqai_bridge import FreqAIBridge
        sig, conf, prob, meta = FreqAIBridge().generate_ml_signal(_ohlcv(400), "BTC/USDT")
        assert isinstance(sig, Signal)
        assert 0.0 <= conf <= 1.0
        assert prob is None or (0.0 <= prob <= 1.0)
        assert {"backend", "n_train_samples", "top_features",
                "trained_at", "n_features", "reasoning"} <= meta.keys()

    def test_insufficient_data_returns_neutral_none(self):
        from ml.freqai_bridge import FreqAIBridge
        sig, conf, prob, meta = FreqAIBridge().generate_ml_signal(_ohlcv(20), "BTC/USDT")
        assert sig == Signal.NEUTRAL
        assert prob is None
        assert conf == 0.0

    def test_disabled_returns_neutral(self, monkeypatch):
        monkeypatch.setenv("ML_ENABLED", "false")
        from core.config import get_settings
        get_settings.cache_clear()
        from ml.freqai_bridge import FreqAIBridge
        sig, conf, prob, meta = FreqAIBridge().generate_ml_signal(_ohlcv(400), "BTC/USDT")
        assert sig == Signal.NEUTRAL and prob is None
        assert "disabled" in meta["reasoning"].lower()

    def test_persistence_creates_files(self):
        from ml.freqai_bridge import FreqAIBridge
        b = FreqAIBridge()
        b.generate_ml_signal(_ohlcv(400), "BTC/USDT", force_retrain=True)
        assert os.path.exists(b._model_path("BTC/USDT"))
        assert os.path.exists(b._meta_path("BTC/USDT"))

    def test_adaptive_cycle_counter_increments(self):
        from ml.freqai_bridge import FreqAIBridge, clear_model_cache
        b = FreqAIBridge()
        b.generate_ml_signal(_ohlcv(400), "BTC/USDT", force_retrain=True)
        assert b._load_meta("BTC/USDT")["cycles_since_train"] == 0
        # Subsequent inferences (no retrain) bump the odometer
        clear_model_cache()  # force disk load path
        b.generate_ml_signal(_ohlcv(400), "BTC/USDT")
        clear_model_cache()
        b.generate_ml_signal(_ohlcv(400), "BTC/USDT")
        assert b._load_meta("BTC/USDT")["cycles_since_train"] >= 1

    def test_retrain_triggers_after_interval(self, monkeypatch):
        monkeypatch.setenv("ML_RETRAIN_INTERVAL", "2")
        from core.config import get_settings
        get_settings.cache_clear()
        from ml.freqai_bridge import FreqAIBridge, clear_model_cache
        b = FreqAIBridge()
        b.generate_ml_signal(_ohlcv(400), "BTC/USDT", force_retrain=True)
        # Bump counter past the interval, then a normal call must reset it to 0
        for _ in range(3):
            clear_model_cache()
            b.generate_ml_signal(_ohlcv(400), "BTC/USDT")
        meta = b._load_meta("BTC/USDT")
        # After crossing interval the model retrains → counter resets low
        assert meta["cycles_since_train"] <= 2

    def test_separate_symbols_separate_models(self):
        from ml.freqai_bridge import FreqAIBridge
        b = FreqAIBridge()
        b.generate_ml_signal(_ohlcv(400, seed=1), "BTC/USDT", force_retrain=True)
        b.generate_ml_signal(_ohlcv(400, seed=2), "ETH/USDT", force_retrain=True)
        assert os.path.exists(b._model_path("BTC/USDT"))
        assert os.path.exists(b._model_path("ETH/USDT"))
        assert b._model_path("BTC/USDT") != b._model_path("ETH/USDT")

    @pytest.mark.parametrize("prob,expected", [
        (0.90, Signal.STRONG_BUY),
        (0.58, Signal.BUY),
        (0.50, Signal.NEUTRAL),
        (0.42, Signal.SELL),
        (0.10, Signal.STRONG_SELL),
    ])
    def test_prob_to_signal_thresholds(self, prob, expected):
        from ml.freqai_bridge import FreqAIBridge
        assert FreqAIBridge()._prob_to_signal(prob) == expected

    def test_self_healing_on_bad_dataframe(self):
        """A malformed frame must not raise — bridge returns NEUTRAL/none."""
        from ml.freqai_bridge import FreqAIBridge
        bad = pd.DataFrame({"close": [1, 2, 3]})  # missing OHLCV columns, too short
        sig, conf, prob, meta = FreqAIBridge().generate_ml_signal(bad, "BTC/USDT")
        assert sig == Signal.NEUTRAL


# ── E. Agent node ───────────────────────────────────────────────────────────────

class TestMLAgent:

    @pytest.mark.asyncio
    async def test_returns_ml_snapshot(self):
        from agents.ml_analyst.agent import run
        from core.state import MLSnapshot
        state = {"symbol": "BTC/USDT", "ohlcv": {"1h": _ohlcv(400)}, "errors": []}
        out = await run(state)
        assert "ml" in out
        assert isinstance(out["ml"], MLSnapshot)
        assert out["ml"].signal in Signal
        assert 0.0 <= out["ml"].confidence <= 1.0

    @pytest.mark.asyncio
    async def test_timeframe_fallback(self):
        """No 1h but 5m present → agent falls back and still produces a snapshot."""
        from agents.ml_analyst.agent import run
        state = {"symbol": "BTC/USDT", "ohlcv": {"5m": _ohlcv(400, freq="5min")}, "errors": []}
        out = await run(state)
        assert out["ml"] is not None

    @pytest.mark.asyncio
    async def test_no_data_returns_none(self):
        from agents.ml_analyst.agent import run
        out = await run({"symbol": "BTC/USDT", "ohlcv": {}, "errors": []})
        assert out["ml"] is None
        assert any("no OHLCV" in e for e in out["errors"])

    @pytest.mark.asyncio
    async def test_short_data_returns_none_snapshot(self):
        from agents.ml_analyst.agent import run
        state = {"symbol": "BTC/USDT", "ohlcv": {"1h": _ohlcv(20)}, "errors": []}
        out = await run(state)
        # Bridge returns NEUTRAL/prob=None → snapshot exists but prob_up is None
        assert out["ml"] is None or out["ml"].prob_up is None

    @pytest.mark.asyncio
    async def test_disabled_returns_none(self, monkeypatch):
        monkeypatch.setenv("ML_ENABLED", "false")
        from core.config import get_settings
        get_settings.cache_clear()
        from agents.ml_analyst.agent import run
        out = await run({"symbol": "BTC/USDT", "ohlcv": {"1h": _ohlcv(400)}, "errors": []})
        assert out["ml"] is None

    @pytest.mark.asyncio
    async def test_exception_guarded(self, monkeypatch):
        """A bridge that raises must not bubble out of the node."""
        import agents.ml_analyst.agent as mod
        def boom(*a, **k):
            raise RuntimeError("synthetic failure")
        monkeypatch.setattr(mod, "_run_bridge", boom)
        out = await mod.run({"symbol": "BTC/USDT", "ohlcv": {"1h": _ohlcv(400)}, "errors": []})
        assert out["ml"] is None
        assert any("synthetic failure" in e for e in out["errors"])


# ── F. Pipeline integration ──────────────────────────────────────────────────────

class TestPipelineWithML:

    @pytest.mark.asyncio
    async def test_graph_compiles_with_ml_node(self):
        from core.orchestrator import build_trading_graph
        g = build_trading_graph()
        assert g is not None

    @pytest.mark.asyncio
    async def test_full_pipeline_populates_ml(self, mock_debate_neutral):
        from unittest.mock import AsyncMock, patch

        from core.orchestrator import build_trading_graph
        from core.state import TradingState
        from tests.conftest import make_full_snapshot

        snap = make_full_snapshot()
        graph = build_trading_graph()
        with patch("data.snapshot.fetch_full_snapshot", AsyncMock(return_value=snap)), \
             patch("agents.onchain_analyst.agent._fetch_funding_rate", AsyncMock(return_value=0.001)):
            result = await graph.ainvoke(TradingState(symbol="BTC/USDT", dry_run=True))
        # ml_analyst ran as a 4th analyst — ml key present (snapshot or None)
        assert "ml" in result

    @pytest.mark.asyncio
    async def test_ml_failure_does_not_block_pipeline(self, mock_debate_neutral):
        from unittest.mock import AsyncMock, patch

        from core.orchestrator import build_trading_graph
        from core.state import TradingState
        from tests.conftest import make_full_snapshot

        snap = make_full_snapshot()
        graph = build_trading_graph()
        with patch("data.snapshot.fetch_full_snapshot", AsyncMock(return_value=snap)), \
             patch("agents.onchain_analyst.agent._fetch_funding_rate", AsyncMock(return_value=0.001)), \
             patch("agents.ml_analyst.agent._run_bridge",
                   side_effect=RuntimeError("ml down")):
            result = await graph.ainvoke(TradingState(dry_run=True))
        # Pipeline still completes; other analysts unaffected
        assert result is not None
        assert result.get("technical") is not None
        assert result.get("sentiment") is not None

    @pytest.mark.asyncio
    async def test_fetch_called_once_with_ml_node(self, mock_debate_neutral):
        from unittest.mock import AsyncMock, patch

        from core.orchestrator import build_trading_graph
        from core.state import TradingState
        from tests.conftest import make_full_snapshot

        snap = make_full_snapshot()
        calls = {"n": 0}
        async def counting(*a, **k):
            calls["n"] += 1
            return snap
        graph = build_trading_graph()
        with patch("data.snapshot.fetch_full_snapshot", counting), \
             patch("agents.onchain_analyst.agent._fetch_funding_rate", AsyncMock(return_value=0.001)):
            await graph.ainvoke(TradingState())
        assert calls["n"] == 1  # ML reuses ingested data, no extra fetch

    @pytest.mark.asyncio
    async def test_ml_evidence_appears_in_debate(self):
        """When an MLSnapshot is in state, _compile_evidence includes an [ML] block."""
        from agents.debate_engine.agent import _compile_evidence
        from core.state import MLSnapshot
        ml = MLSnapshot(symbol="BTC/USDT", prob_up=0.71, signal=Signal.BUY,
                        confidence=0.42, model_type="sklearn_hgb",
                        reasoning="sklearn_hgb P(up)=0.710 → BUY")
        evidence = _compile_evidence({"ml": ml})
        assert "[ML]" in evidence
        assert "P(up)=0.710" in evidence
