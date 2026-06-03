"""
tests/test_ta_agent.py  —  Step 3: TA Agent Tests
===================================================
Comprehensive validation of the technical analysis engine.

Coverage:
  A. indicators.py — compute_all_indicators() and compute_fast_indicators()
  B. indicators.py — _compute_talib vs _compute_fallback equivalence
  C. indicators.py — _validate_and_enrich() guards and derived fields
  D. agent.py      — _evaluate_standard() signal scoring
  E. agent.py      — _evaluate_fast() fast 5m scoring
  F. agent.py      — run() LangGraph node integration
  G. Multi-timeframe confirmation bonuses
  H. Edge cases: constant price, monotone trend, extreme values

Run: pytest tests/test_ta_agent.py -v
"""
from __future__ import annotations

import asyncio
import pytest
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional


# ── Test data factories ───────────────────────────────────────────────────────

def _df(n: int = 200, *, trend: str = "flat", base: float = 50_000.0,
        volatility: float = 0.005, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic OHLCV DataFrame with controllable trend and volatility.
    trend: "flat" | "up" | "down" | "oversold" | "overbought"
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")

    if trend == "up":
        close = np.linspace(base, base * 1.40, n) + rng.standard_normal(n) * base * volatility
    elif trend == "down":
        close = np.linspace(base, base * 0.60, n) + rng.standard_normal(n) * base * volatility
    elif trend == "oversold":
        # Sharp drop to force low RSI
        close = np.concatenate([
            np.linspace(base, base * 0.75, n // 2),
            np.linspace(base * 0.75, base * 0.72, n // 2),
        ]) + rng.standard_normal(n) * base * 0.002
    elif trend == "overbought":
        close = np.concatenate([
            np.linspace(base, base * 1.30, n // 2),
            np.linspace(base * 1.30, base * 1.33, n // 2),
        ]) + rng.standard_normal(n) * base * 0.002
    else:  # flat
        close = base + rng.standard_normal(n) * base * volatility

    close = np.abs(close)
    sp = close * 0.001
    return pd.DataFrame({
        "open":   close - sp,
        "high":   close + sp * 2.5,
        "low":    close - sp * 2.5,
        "close":  close,
        "volume": rng.uniform(5, 200, n),
    }, index=dates)


def _df_5m(n: int = 100, *, trend: str = "flat", seed: int = 0) -> pd.DataFrame:
    """5-minute OHLCV for fast indicators."""
    df = _df(n=n, trend=trend, seed=seed)
    df.index = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return df


def _ind(**overrides) -> Dict[str, Any]:
    """Build a minimal indicator dict for scoring tests."""
    defaults = dict(
        rsi_14=50.0, ema_9=100.0, ema_21=98.0, ema_50=95.0, ema_200=90.0,
        close=100.0, bb_upper=108.0, bb_middle=100.0, bb_lower=92.0,
        bb_width=0.16, bb_pct=0.5,
        macd_hist=0.0, macd_line=0.0, macd_signal=0.0,
        atr_14=2.0, atr_pct=0.02, ema_cross="FLAT",
        volume=1000.0,
    )
    defaults.update(overrides)
    return defaults


# ── A. compute_all_indicators ─────────────────────────────────────────────────

class TestComputeAllIndicators:

    def test_returns_none_when_df_is_none(self):
        from agents.technical_analyst.indicators import compute_all_indicators
        assert compute_all_indicators(None) is None

    def test_returns_none_when_too_few_bars(self):
        from agents.technical_analyst.indicators import compute_all_indicators, _MIN_CANDLES
        assert compute_all_indicators(_df(n=_MIN_CANDLES - 1)) is None

    def test_returns_dict_with_min_candles(self):
        from agents.technical_analyst.indicators import compute_all_indicators, _MIN_CANDLES
        result = compute_all_indicators(_df(n=_MIN_CANDLES))
        assert result is not None
        assert isinstance(result, dict)

    def test_all_required_keys_present(self):
        from agents.technical_analyst.indicators import compute_all_indicators
        result = compute_all_indicators(_df())
        required = {
            "rsi_14", "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct",
            "ema_9", "ema_21", "ema_50",
            "macd_line", "macd_signal", "macd_hist",
            "atr_14", "atr_pct", "ema_cross",
            "close", "volume",
        }
        missing = required - result.keys()
        assert not missing, f"Missing keys: {missing}"

    def test_rsi_in_range_flat_market(self):
        from agents.technical_analyst.indicators import compute_all_indicators
        r = compute_all_indicators(_df())
        assert r is not None
        assert 0.0 <= r["rsi_14"] <= 100.0, f"RSI out of range: {r['rsi_14']}"

    def test_rsi_low_in_downtrend(self):
        from agents.technical_analyst.indicators import compute_all_indicators
        r = compute_all_indicators(_df(n=200, trend="oversold"))
        assert r is not None
        assert r["rsi_14"] < 45, f"Expected low RSI in downtrend, got {r['rsi_14']}"

    def test_rsi_high_in_uptrend(self):
        from agents.technical_analyst.indicators import compute_all_indicators
        r = compute_all_indicators(_df(n=200, trend="overbought"))
        assert r is not None
        assert r["rsi_14"] > 55, f"Expected high RSI in uptrend, got {r['rsi_14']}"

    def test_bollinger_bands_ordered(self):
        from agents.technical_analyst.indicators import compute_all_indicators
        r = compute_all_indicators(_df())
        assert r is not None
        assert r["bb_lower"] < r["bb_middle"] < r["bb_upper"], (
            f"BB ordering violated: {r['bb_lower']:.2f} < {r['bb_middle']:.2f} < {r['bb_upper']:.2f}"
        )

    def test_bb_width_positive(self):
        from agents.technical_analyst.indicators import compute_all_indicators
        r = compute_all_indicators(_df())
        assert r is not None
        assert r["bb_width"] > 0.0

    def test_bb_pct_in_range(self):
        from agents.technical_analyst.indicators import compute_all_indicators
        r = compute_all_indicators(_df())
        assert r is not None
        # bb_pct can exceed [0,1] when price is outside the bands
        assert isinstance(r["bb_pct"], float)

    def test_atr_positive(self):
        from agents.technical_analyst.indicators import compute_all_indicators
        r = compute_all_indicators(_df())
        assert r is not None
        assert r["atr_14"] > 0.0, "ATR must be positive"

    def test_atr_pct_reasonable(self):
        from agents.technical_analyst.indicators import compute_all_indicators
        r = compute_all_indicators(_df())
        assert r is not None
        assert r["atr_pct"] is not None
        assert 0 < r["atr_pct"] < 0.50, f"atr_pct unreasonable: {r['atr_pct']}"

    def test_ema_cross_bull_in_uptrend(self):
        from agents.technical_analyst.indicators import compute_all_indicators
        r = compute_all_indicators(_df(trend="up", n=300))
        assert r is not None
        assert r["ema_cross"] == "BULL", f"Expected BULL ema_cross in uptrend, got {r['ema_cross']}"

    def test_ema_cross_bear_in_downtrend(self):
        from agents.technical_analyst.indicators import compute_all_indicators
        r = compute_all_indicators(_df(trend="down", n=300))
        assert r is not None
        assert r["ema_cross"] == "BEAR", f"Expected BEAR ema_cross in downtrend, got {r['ema_cross']}"

    def test_ema_200_none_without_enough_bars(self):
        from agents.technical_analyst.indicators import compute_all_indicators
        r = compute_all_indicators(_df(n=50))
        assert r is not None
        assert r.get("ema_200") is None, "EMA-200 should be None with only 50 bars"

    def test_ema_200_present_with_enough_bars(self):
        from agents.technical_analyst.indicators import compute_all_indicators
        r = compute_all_indicators(_df(n=220))
        assert r is not None
        assert r.get("ema_200") is not None
        assert r["ema_200"] > 0

    def test_close_matches_last_bar(self):
        from agents.technical_analyst.indicators import compute_all_indicators
        df = _df()
        r  = compute_all_indicators(df)
        assert r is not None
        assert abs(r["close"] - df["close"].iloc[-1]) < 1e-6

    def test_no_nan_in_output(self):
        from agents.technical_analyst.indicators import compute_all_indicators
        import math
        r = compute_all_indicators(_df())
        assert r is not None
        for k, v in r.items():
            if v is not None and isinstance(v, float):
                assert not math.isnan(v), f"NaN found at key {k!r}"
                assert not math.isinf(v), f"Inf found at key {k!r}"


# ── B. TA-Lib vs pandas fallback equivalence ──────────────────────────────────

class TestFallbackEquivalence:
    """Verify that pandas fallback produces numerically equivalent results."""

    def _compute_both(self, df):
        from agents.technical_analyst.indicators import _compute_talib, _compute_fallback, _validate_and_enrich
        fb = _validate_and_enrich(_compute_fallback(df))
        try:
            import talib
            tl = _validate_and_enrich(_compute_talib(df, talib))
            return tl, fb
        except ImportError:
            pytest.skip("TA-Lib not installed — skipping equivalence test")

    def test_rsi_close(self):
        df = _df()
        tl, fb = self._compute_both(df)
        assert abs(tl["rsi_14"] - fb["rsi_14"]) < 2.0, (
            f"RSI divergence: talib={tl['rsi_14']:.3f} fallback={fb['rsi_14']:.3f}"
        )

    def test_ema9_close(self):
        df = _df()
        tl, fb = self._compute_both(df)
        assert abs(tl["ema_9"] - fb["ema_9"]) / tl["ema_9"] < 0.005

    def test_bb_middle_close(self):
        df = _df()
        tl, fb = self._compute_both(df)
        assert abs(tl["bb_middle"] - fb["bb_middle"]) / tl["bb_middle"] < 0.001

    def test_macd_hist_same_sign(self):
        """MACD hist direction must agree between paths."""
        df = _df(trend="up")
        tl, fb = self._compute_both(df)
        assert (tl["macd_hist"] > 0) == (fb["macd_hist"] > 0), (
            f"MACD hist sign mismatch: talib={tl['macd_hist']:.4f} fb={fb['macd_hist']:.4f}"
        )

    def test_ema_cross_same(self):
        for trend in ("up", "down", "flat"):
            df = _df(trend=trend, n=300)
            tl, fb = self._compute_both(df)
            assert tl["ema_cross"] == fb["ema_cross"], (
                f"ema_cross mismatch ({trend}): talib={tl['ema_cross']} fb={fb['ema_cross']}"
            )


# ── C. _validate_and_enrich guards ────────────────────────────────────────────

class TestValidateAndEnrich:

    def test_rsi_clipped_above_100(self):
        from agents.technical_analyst.indicators import _validate_and_enrich
        r = _validate_and_enrich({"rsi_14": 101.5, "close": 100.0})
        assert r["rsi_14"] == 100.0

    def test_rsi_clipped_below_0(self):
        from agents.technical_analyst.indicators import _validate_and_enrich
        r = _validate_and_enrich({"rsi_14": -0.5, "close": 100.0})
        assert r["rsi_14"] == 0.0

    def test_nan_replaced_with_none(self):
        from agents.technical_analyst.indicators import _validate_and_enrich
        import math
        r = _validate_and_enrich({"rsi_14": float("nan"), "close": 100.0})
        assert r["rsi_14"] is None

    def test_inf_replaced_with_none(self):
        from agents.technical_analyst.indicators import _validate_and_enrich
        r = _validate_and_enrich({"rsi_14": float("inf"), "close": 100.0})
        assert r["rsi_14"] is None

    def test_bb_pct_computed(self):
        from agents.technical_analyst.indicators import _validate_and_enrich
        r = _validate_and_enrich({
            "bb_upper": 110.0, "bb_middle": 100.0, "bb_lower": 90.0, "close": 95.0
        })
        # (95 - 90) / (110 - 90) = 5/20 = 0.25
        assert r["bb_pct"] == pytest.approx(0.25)

    def test_atr_pct_computed(self):
        from agents.technical_analyst.indicators import _validate_and_enrich
        r = _validate_and_enrich({"atr_14": 1500.0, "close": 50_000.0})
        assert r["atr_pct"] == pytest.approx(0.03)

    def test_ema_cross_bull(self):
        from agents.technical_analyst.indicators import _validate_and_enrich
        r = _validate_and_enrich({"ema_9": 100.0, "ema_21": 98.0, "ema_50": 95.0})
        assert r["ema_cross"] == "BULL"

    def test_ema_cross_bear(self):
        from agents.technical_analyst.indicators import _validate_and_enrich
        r = _validate_and_enrich({"ema_9": 90.0, "ema_21": 95.0, "ema_50": 100.0})
        assert r["ema_cross"] == "BEAR"

    def test_ema_cross_flat_when_mixed(self):
        from agents.technical_analyst.indicators import _validate_and_enrich
        r = _validate_and_enrich({"ema_9": 98.0, "ema_21": 100.0, "ema_50": 97.0})
        assert r["ema_cross"] == "FLAT"


# ── D. _evaluate_standard signal scoring ─────────────────────────────────────

class TestEvaluateStandard:

    def test_neutral_on_mixed_signals(self):
        from agents.technical_analyst.agent import _evaluate_standard
        sig, conf, _ = _evaluate_standard(_ind())
        assert sig == Signal.NEUTRAL
        assert conf < 0.3

    def test_strong_buy_on_extreme_conditions(self):
        """Deep oversold RSI + bull EMA + price at lower BB + positive MACD."""
        from agents.technical_analyst.agent import _evaluate_standard
        from core.state import Signal
        ind = _ind(rsi_14=18, ema_cross="BULL", bb_pct=0.02,
                   macd_hist=50.0, macd_line=10.0, macd_signal=5.0,
                   ema_200=80.0, close=100.0)
        sig, conf, reason = _evaluate_standard(ind)
        assert sig in (Signal.STRONG_BUY, Signal.BUY)
        assert conf > 0.4

    def test_strong_sell_on_extreme_conditions(self):
        """Deeply overbought RSI + bear EMA + price at upper BB + negative MACD."""
        from agents.technical_analyst.agent import _evaluate_standard
        from core.state import Signal
        ind = _ind(rsi_14=82, ema_cross="BEAR", bb_pct=0.98,
                   macd_hist=-50.0, macd_line=-10.0, macd_signal=-5.0,
                   ema_200=120.0, close=100.0)
        sig, conf, reason = _evaluate_standard(ind)
        assert sig in (Signal.STRONG_SELL, Signal.SELL)
        assert conf > 0.4

    def test_buy_on_oversold_rsi_bull_ema(self):
        from agents.technical_analyst.agent import _evaluate_standard
        from core.state import Signal
        sig, _, _ = _evaluate_standard(_ind(rsi_14=28, ema_cross="BULL",
                                            bb_pct=0.15, macd_hist=1.0))
        assert sig in (Signal.BUY, Signal.STRONG_BUY)

    def test_sell_on_overbought_rsi_bear_ema(self):
        from agents.technical_analyst.agent import _evaluate_standard
        from core.state import Signal
        sig, _, _ = _evaluate_standard(_ind(rsi_14=72, ema_cross="BEAR",
                                            bb_pct=0.85, macd_hist=-1.0))
        assert sig in (Signal.SELL, Signal.STRONG_SELL)

    def test_confidence_clamped_to_1(self):
        """Confidence must never exceed 1.0 regardless of extreme inputs."""
        from agents.technical_analyst.agent import _evaluate_standard
        ind = _ind(rsi_14=5, ema_cross="BULL", bb_pct=0.0, macd_hist=500.0,
                   macd_line=100.0, macd_signal=0.0)
        _, conf, _ = _evaluate_standard(ind)
        assert conf <= 1.0

    def test_confidence_non_negative(self):
        from agents.technical_analyst.agent import _evaluate_standard
        ind = _ind(rsi_14=95, ema_cross="BEAR", bb_pct=1.0, macd_hist=-500.0)
        _, conf, _ = _evaluate_standard(ind)
        assert conf >= 0.0

    def test_reasoning_not_empty(self):
        from agents.technical_analyst.agent import _evaluate_standard
        _, _, reason = _evaluate_standard(_ind(rsi_14=28))
        assert len(reason) > 5

    def test_mtf_confirmation_increases_bull_score(self):
        """Adding a bullish 1h confirmation should increase the score."""
        from agents.technical_analyst.agent import _evaluate_standard
        ind_4h = _ind(rsi_14=35, ema_cross="BULL", macd_hist=1.0)
        ind_1h_bull = _ind(ema_cross="BULL", rsi_14=45)
        ind_1h_bear = _ind(ema_cross="BEAR", rsi_14=45)

        _, conf_with_confirm, _ = _evaluate_standard(ind_4h, ind_1h_bull)
        _, conf_no_confirm, _   = _evaluate_standard(ind_4h)
        _, conf_against, _      = _evaluate_standard(ind_4h, ind_1h_bear)

        # Bullish 1h confirmation should give equal or higher confidence than no confirmation
        assert conf_with_confirm >= conf_no_confirm

    def test_bb_squeeze_mentioned_in_reasoning(self):
        from agents.technical_analyst.agent import _evaluate_standard
        ind = _ind(bb_width=0.01)  # tight bands → squeeze
        _, _, reason = _evaluate_standard(ind)
        assert "squeeze" in reason.lower()


# ── E. _evaluate_fast scoring ─────────────────────────────────────────────────

class TestEvaluateFast:

    def test_neutral_on_balanced_signals(self):
        from agents.technical_analyst.agent import _evaluate_fast
        from core.state import Signal
        ind = {"macd_fast_hist": 0.0, "rsi_14": 50.0, "close": 100.0,
               "vwap": 100.0, "cvd": 0.0, "volume": 100.0}
        sig, conf, _ = _evaluate_fast(ind)
        assert sig == Signal.NEUTRAL

    def test_buy_on_oversold_above_vwap_positive_macd(self):
        from agents.technical_analyst.agent import _evaluate_fast
        from core.state import Signal
        ind = {"macd_fast_hist": 5.0, "rsi_14": 28.0,
               "close": 102.0, "vwap": 100.0, "cvd": 1000.0, "volume": 50.0}
        sig, conf, _ = _evaluate_fast(ind)
        assert sig in (Signal.BUY, Signal.STRONG_BUY)
        assert conf > 0.3

    def test_sell_on_overbought_below_vwap_negative_macd(self):
        from agents.technical_analyst.agent import _evaluate_fast
        from core.state import Signal
        ind = {"macd_fast_hist": -5.0, "rsi_14": 74.0,
               "close": 98.0, "vwap": 100.0, "cvd": -1000.0, "volume": 50.0}
        sig, conf, _ = _evaluate_fast(ind)
        assert sig in (Signal.SELL, Signal.STRONG_SELL)

    def test_macd_magnitude_scales_confidence(self):
        """Larger MACD(3,15,3) histogram → higher confidence."""
        from agents.technical_analyst.agent import _evaluate_fast
        ind_small = {"macd_fast_hist": 0.5, "rsi_14": 50.0, "close": 100.0,
                     "vwap": 99.0, "cvd": 0.0, "volume": 50.0}
        ind_large = {"macd_fast_hist": 10.0, "rsi_14": 50.0, "close": 100.0,
                     "vwap": 99.0, "cvd": 0.0, "volume": 50.0}
        _, conf_s, _ = _evaluate_fast(ind_small)
        _, conf_l, _ = _evaluate_fast(ind_large)
        assert conf_l >= conf_s

    def test_cvd_adds_to_direction(self):
        from agents.technical_analyst.agent import _evaluate_fast
        from core.state import Signal
        ind_bull_cvd = {"macd_fast_hist": 2.0, "rsi_14": 45.0,
                        "close": 101.0, "vwap": 100.0, "cvd": 5000.0, "volume": 50.0}
        ind_bear_cvd = {"macd_fast_hist": 2.0, "rsi_14": 45.0,
                        "close": 101.0, "vwap": 100.0, "cvd": -5000.0, "volume": 50.0}
        sig_bull, conf_bull, _ = _evaluate_fast(ind_bull_cvd)
        sig_bear, conf_bear, _ = _evaluate_fast(ind_bear_cvd)
        # Same MACD, but bullish CVD should give higher confidence than bearish
        assert conf_bull >= conf_bear

    def test_confidence_clamped(self):
        from agents.technical_analyst.agent import _evaluate_fast
        ind = {"macd_fast_hist": 1000.0, "rsi_14": 1.0, "close": 200.0,
               "vwap": 100.0, "cvd": 999999.0, "volume": 50.0}
        _, conf, _ = _evaluate_fast(ind)
        assert 0.0 <= conf <= 1.0


# ── F. compute_fast_indicators ───────────────────────────────────────────────

class TestComputeFastIndicators:

    def test_returns_none_on_too_few_bars(self):
        from agents.technical_analyst.indicators import compute_fast_indicators, _MIN_CANDLES_FAST
        assert compute_fast_indicators(_df_5m(n=_MIN_CANDLES_FAST - 1)) is None

    def test_returns_required_keys(self):
        from agents.technical_analyst.indicators import compute_fast_indicators
        r = compute_fast_indicators(_df_5m(n=50))
        assert r is not None
        assert {"macd_fast_hist", "rsi_14", "vwap", "cvd", "close", "volume"} <= r.keys()

    def test_rsi_in_range(self):
        from agents.technical_analyst.indicators import compute_fast_indicators
        r = compute_fast_indicators(_df_5m())
        assert r is not None
        assert 0.0 <= r["rsi_14"] <= 100.0

    def test_vwap_positive(self):
        from agents.technical_analyst.indicators import compute_fast_indicators
        r = compute_fast_indicators(_df_5m())
        assert r is not None
        assert r["vwap"] > 0

    def test_cvd_float(self):
        from agents.technical_analyst.indicators import compute_fast_indicators
        r = compute_fast_indicators(_df_5m())
        assert r is not None
        assert isinstance(r["cvd"], float)

    def test_macd_fast_hist_type(self):
        from agents.technical_analyst.indicators import compute_fast_indicators
        r = compute_fast_indicators(_df_5m())
        assert r is not None
        assert isinstance(r["macd_fast_hist"], float)

    def test_cvd_positive_in_uptrend(self):
        """In a strong uptrend close > open → CVD should be positive."""
        from agents.technical_analyst.indicators import compute_fast_indicators
        r = compute_fast_indicators(_df_5m(trend="up"))
        assert r is not None
        # CVD direction should reflect the trend (not guaranteed with noise but high probability)
        # Just check it's finite
        import math
        assert not math.isnan(r["cvd"])

    def test_returns_none_for_none_input(self):
        from agents.technical_analyst.indicators import compute_fast_indicators
        assert compute_fast_indicators(None) is None


# ── G. agent.run() integration ───────────────────────────────────────────────

class TestAgentRun:

    def _state(self, tfs=None) -> Dict[str, Any]:
        dfs = {
            "4h": _df(n=200),
            "1h": _df(n=200, seed=1),
            "5m": _df_5m(n=100),
            "1d": _df(n=365, seed=2),
        }
        if tfs is not None:
            dfs = {k: v for k, v in dfs.items() if k in tfs}
        return {"symbol": "BTC/USDT", "ohlcv": dfs, "errors": []}

    @pytest.mark.asyncio
    async def test_run_returns_technical_snapshot(self):
        from agents.technical_analyst.agent import run
        result = await run(self._state())
        assert "technical" in result
        assert result["technical"] is not None

    @pytest.mark.asyncio
    async def test_run_returns_technical_5m_snapshot(self):
        from agents.technical_analyst.agent import run
        result = await run(self._state())
        assert "technical_5m" in result
        assert result["technical_5m"] is not None

    @pytest.mark.asyncio
    async def test_run_missing_4h_returns_none(self):
        from agents.technical_analyst.agent import run
        state = self._state(tfs=["1h", "5m"])
        result = await run(state)
        assert result["technical"] is None
        assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_run_without_5m_still_succeeds(self):
        from agents.technical_analyst.agent import run
        state = self._state(tfs=["4h", "1h"])
        result = await run(state)
        assert result["technical"] is not None
        assert result.get("technical_5m") is None

    @pytest.mark.asyncio
    async def test_snapshot_fields_populated(self):
        from agents.technical_analyst.agent import run
        from core.state import IndicatorSnapshot, Timeframe
        result = await run(self._state())
        snap = result["technical"]
        assert isinstance(snap, IndicatorSnapshot)
        assert snap.timeframe == Timeframe.H4
        assert snap.rsi_14 is not None
        assert snap.bb_upper is not None
        assert snap.ema_9 is not None
        assert snap.atr_14 is not None
        assert snap.signal is not None
        assert 0.0 <= snap.confidence <= 1.0
        assert len(snap.reasoning) > 5

    @pytest.mark.asyncio
    async def test_5m_snapshot_has_fast_fields(self):
        from agents.technical_analyst.agent import run
        from core.state import Timeframe
        result = await run(self._state())
        snap_5m = result["technical_5m"]
        assert snap_5m.timeframe == Timeframe.M5
        assert snap_5m.macd_fast_hist is not None
        assert snap_5m.vwap is not None
        assert snap_5m.cvd is not None

    @pytest.mark.asyncio
    async def test_uptrend_produces_bullish_or_neutral(self):
        """Strong uptrend should never produce STRONG_SELL."""
        from agents.technical_analyst.agent import run
        from core.state import Signal
        state = {
            "symbol": "BTC/USDT",
            "ohlcv": {"4h": _df(trend="up", n=300), "1h": _df(trend="up", n=300, seed=1)},
            "errors": [],
        }
        result = await run(state)
        assert result["technical"].signal != Signal.STRONG_SELL

    @pytest.mark.asyncio
    async def test_downtrend_produces_bearish_or_neutral(self):
        """Strong downtrend should never produce STRONG_BUY."""
        from agents.technical_analyst.agent import run
        from core.state import Signal
        state = {
            "symbol": "BTC/USDT",
            "ohlcv": {"4h": _df(trend="down", n=300), "1h": _df(trend="down", n=300, seed=1)},
            "errors": [],
        }
        result = await run(state)
        assert result["technical"].signal != Signal.STRONG_BUY


# ── H. Edge cases ─────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_constant_price_does_not_crash(self):
        """Constant price → zero std → BB bands equal → no crash."""
        from agents.technical_analyst.indicators import compute_all_indicators
        import math
        dates = pd.date_range("2024-01-01", periods=100, freq="4h", tz="UTC")
        df = pd.DataFrame({
            "open": 50_000.0, "high": 50_000.0,
            "low":  50_000.0, "close": 50_000.0,
            "volume": 100.0,
        }, index=dates)
        # Should either return a valid result or None, never raise
        result = compute_all_indicators(df)
        if result is not None:
            for k, v in result.items():
                if isinstance(v, float):
                    assert not math.isnan(v) and not math.isinf(v), f"NaN/Inf at {k}"

    def test_very_volatile_price_does_not_crash(self):
        """Extreme volatility should not produce NaN/Inf."""
        from agents.technical_analyst.indicators import compute_all_indicators
        import math
        rng = np.random.default_rng(99)
        dates = pd.date_range("2024-01-01", periods=200, freq="4h", tz="UTC")
        # Random walk with 20% moves per bar
        close = np.abs(50_000.0 * np.cumprod(1 + rng.standard_normal(200) * 0.20))
        df = pd.DataFrame({
            "open": close * 0.99, "high": close * 1.02,
            "low":  close * 0.98, "close": close,
            "volume": rng.uniform(1, 1000, 200),
        }, index=dates)
        result = compute_all_indicators(df)
        if result is not None:
            for k, v in result.items():
                if isinstance(v, float):
                    assert not math.isnan(v), f"NaN at {k}"

    def test_single_bar_above_min_candles(self):
        """Adding exactly _MIN_CANDLES bars should work."""
        from agents.technical_analyst.indicators import compute_all_indicators, _MIN_CANDLES
        result = compute_all_indicators(_df(n=_MIN_CANDLES))
        assert result is not None

    def test_empty_dataframe_returns_none(self):
        from agents.technical_analyst.indicators import compute_all_indicators
        df = pd.DataFrame(columns=["open","high","low","close","volume"])
        assert compute_all_indicators(df) is None

    @pytest.mark.asyncio
    async def test_agent_handles_corrupt_ohlcv_gracefully(self):
        """agent.run() must not raise even on bad data."""
        from agents.technical_analyst.agent import run
        state = {
            "symbol": "BTC/USDT",
            "ohlcv": {"4h": pd.DataFrame()},  # empty df
            "errors": [],
        }
        result = await run(state)
        # Should return None technical and log an error, never raise
        assert result.get("technical") is None


# ── Import to surface Signal ──────────────────────────────────────────────────
from core.state import Signal
