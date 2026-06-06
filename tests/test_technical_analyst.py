"""
tests/test_technical_analyst.py
Tests for indicator computation and signal evaluation.
"""
import numpy as np
import pandas as pd

from agents.technical_analyst.agent import _evaluate
from agents.technical_analyst.indicators import _MIN_CANDLES, compute_all_indicators

# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_df(n=200, trend="flat", base=50_000.0):
    """Generate synthetic OHLCV DataFrame."""
    dates = pd.date_range(start="2024-01-01", periods=n, freq="4h", tz="UTC")
    if trend == "up":
        close = np.linspace(base, base * 1.3, n) + np.random.normal(0, base * 0.005, n)
    elif trend == "down":
        close = np.linspace(base, base * 0.7, n) + np.random.normal(0, base * 0.005, n)
    else:
        close = base + np.random.normal(0, base * 0.01, n)

    close  = np.abs(close)
    spread = close * 0.002
    return pd.DataFrame({
        "open":   close - spread,
        "high":   close + spread * 2,
        "low":    close - spread * 2,
        "close":  close,
        "volume": np.random.uniform(100, 1000, n),
    }, index=dates)


# ─── compute_all_indicators ───────────────────────────────────────────────────

def test_compute_returns_none_for_short_df():
    df = _make_df(n=10)
    assert compute_all_indicators(df) is None


def test_compute_returns_none_for_none():
    assert compute_all_indicators(None) is None


def test_compute_has_required_keys():
    df = _make_df(n=_MIN_CANDLES + 10)
    result = compute_all_indicators(df)
    assert result is not None
    required = {"rsi_14", "bb_upper", "bb_lower", "bb_middle", "ema_9", "ema_21", "ema_50",
                "macd_line", "macd_signal", "macd_hist", "atr_14", "close", "volume"}
    assert required.issubset(result.keys()), f"Missing keys: {required - result.keys()}"


def test_rsi_in_valid_range():
    df = _make_df(n=200)
    result = compute_all_indicators(df)
    assert result is not None
    assert 0 <= result["rsi_14"] <= 100, f"RSI out of range: {result['rsi_14']}"


def test_bollinger_band_ordering():
    df = _make_df(n=200)
    result = compute_all_indicators(df)
    assert result is not None
    assert result["bb_lower"] < result["bb_middle"] < result["bb_upper"]


def test_atr_positive():
    df = _make_df(n=200)
    result = compute_all_indicators(df)
    assert result is not None
    assert result["atr_14"] > 0


# ─── _evaluate signal mapping ─────────────────────────────────────────────────

from core.state import Signal  # noqa: E402


def _ind(rsi=50, e9=100, e21=98, e50=95, e200=90, close=100,
         bbu=105, bbm=100, bbl=95, bbw=0.1, mhist=0.5, mline=0.5, msig=0.3, atr=2.0):
    return dict(rsi_14=rsi, ema_9=e9, ema_21=e21, ema_50=e50, ema_200=e200,
                close=close, bb_upper=bbu, bb_middle=bbm, bb_lower=bbl, bb_width=bbw,
                macd_hist=mhist, macd_line=mline, macd_signal=msig, atr_14=atr, volume=1000)


def test_strong_buy_signal():
    # Oversold RSI + bullish EMA + price at lower BB + positive MACD
    ind = _ind(rsi=22, e9=100, e21=98, e50=95, close=95, bbl=95.5, mhist=1.0, mline=0.5, msig=0.2)
    sig, conf, _ = _evaluate(ind)
    assert sig in (Signal.STRONG_BUY, Signal.BUY)
    assert conf > 0.3


def test_neutral_signal():
    ind = _ind(rsi=50, e9=100, e21=100, e50=100, mhist=0.0, mline=0.0, msig=0.0)
    sig, _, _ = _evaluate(ind)
    assert sig == Signal.NEUTRAL


def test_bearish_signal():
    ind = _ind(rsi=75, e9=90, e21=95, e50=100, close=100, bbu=99, mhist=-1.5, mline=-0.5, msig=0.1)
    sig, conf, _ = _evaluate(ind)
    assert sig in (Signal.SELL, Signal.STRONG_SELL)
    assert conf > 0.2


def test_confidence_clamped():
    # Extreme scenario — confidence must never exceed 1.0
    ind = _ind(rsi=10, e9=100, e21=98, e50=95, close=90, bbl=95, mhist=5.0, mline=2.0, msig=0.1)
    _, conf, _ = _evaluate(ind)
    assert 0.0 <= conf <= 1.0
