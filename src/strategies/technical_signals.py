"""
Strategy A: Technical Signal Generator
  - MACD momentum (3, 15, 3)
  - RSI(14) + VWAP reversals
  - CVD (Cumulative Volume Delta) divergence
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)


@dataclass
class TechnicalSignal:
    direction: str          # "YES" | "NO" | "NEUTRAL"
    confidence: float       # 0.0–1.0
    macd_signal: str        # "BULLISH" | "BEARISH" | "NEUTRAL"
    rsi_vwap_signal: str    # "OVERSOLD_ABOVE_VWAP" | "OVERBOUGHT_BELOW_VWAP" | "NEUTRAL"
    cvd_divergence: str     # "BULLISH_DIV" | "BEARISH_DIV" | "NONE"
    rsi: float = 0.0
    macd_hist: float = 0.0
    cvd: float = 0.0
    vwap: float = 0.0
    close: float = 0.0
    reasoning: str = ""


def compute_cvd(df: pd.DataFrame) -> pd.Series:
    """
    Approximate CVD from OHLCV: use close vs open as buy/sell proxy.
    For real CVD, tick data is needed — this is a reasonable 5m approximation.
    CVD[i] = Σ (sign(close-open) * volume) up to bar i
    """
    delta = np.where(df["close"] >= df["open"], df["volume"], -df["volume"])
    return pd.Series(delta, index=df.index).cumsum()


def generate_technical_signal(df: pd.DataFrame, lookback: int = 10) -> TechnicalSignal:
    """
    Compute MACD + RSI/VWAP + CVD on the last `lookback` bars of df (5m OHLCV).
    Returns a TechnicalSignal.
    """
    if df is None or len(df) < 50:
        return TechnicalSignal("NEUTRAL", 0.0, "NEUTRAL", "NEUTRAL", "NONE",
                               reasoning="Insufficient data")

    df = df.copy()

    # ── MACD (3, 15, 3) ───────────────────────────────────────────────────────
    macd_df = ta.macd(df["close"], fast=3, slow=15, signal=3)
    if macd_df is None or macd_df.empty:
        macd_hist = 0.0
        macd_sig = "NEUTRAL"
    else:
        col = [c for c in macd_df.columns if "h" in c.lower()]
        macd_hist = float(macd_df[col[0]].iloc[-1]) if col else 0.0
        prev_hist  = float(macd_df[col[0]].iloc[-2]) if col else 0.0
        if macd_hist > 0 and macd_hist > prev_hist:
            macd_sig = "BULLISH"
        elif macd_hist < 0 and macd_hist < prev_hist:
            macd_sig = "BEARISH"
        else:
            macd_sig = "NEUTRAL"

    # ── RSI(14) ───────────────────────────────────────────────────────────────
    rsi_series = ta.rsi(df["close"], length=14)
    rsi = float(rsi_series.iloc[-1]) if rsi_series is not None else 50.0

    # ── VWAP ──────────────────────────────────────────────────────────────────
    # pandas-ta vwap needs datetime index — use rolling approximation if needed
    try:
        vwap_series = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
        vwap = float(vwap_series.iloc[-1]) if vwap_series is not None else float(df["close"].mean())
    except Exception:
        vwap = float((df["close"] * df["volume"]).sum() / df["volume"].sum())

    close = float(df["close"].iloc[-1])

    if rsi < 35 and close > vwap:
        rsi_vwap_sig = "OVERSOLD_ABOVE_VWAP"   # Bullish reversal
    elif rsi > 65 and close < vwap:
        rsi_vwap_sig = "OVERBOUGHT_BELOW_VWAP"  # Bearish reversal
    else:
        rsi_vwap_sig = "NEUTRAL"

    # ── CVD Divergence ────────────────────────────────────────────────────────
    cvd = compute_cvd(df)
    cvd_recent = cvd.iloc[-lookback:]
    price_recent = df["close"].iloc[-lookback:]

    price_trend = float(price_recent.iloc[-1] - price_recent.iloc[0])
    cvd_trend   = float(cvd_recent.iloc[-1]   - cvd_recent.iloc[0])

    if price_trend > 0 and cvd_trend < 0:
        cvd_div = "BEARISH_DIV"   # price up, volume selling → bearish
    elif price_trend < 0 and cvd_trend > 0:
        cvd_div = "BULLISH_DIV"   # price down, volume buying → bullish
    else:
        cvd_div = "NONE"

    # ── Composite direction + confidence ─────────────────────────────────────
    bull_score = 0
    bear_score = 0

    if macd_sig == "BULLISH":
        bull_score += 1
    elif macd_sig == "BEARISH":
        bear_score += 1

    if rsi_vwap_sig == "OVERSOLD_ABOVE_VWAP":
        bull_score += 1
    elif rsi_vwap_sig == "OVERBOUGHT_BELOW_VWAP":
        bear_score += 1

    if cvd_div == "BULLISH_DIV":
        bull_score += 1
    elif cvd_div == "BEARISH_DIV":
        bear_score += 1

    total = bull_score + bear_score
    if total == 0:
        direction, confidence = "NEUTRAL", 0.0
    elif bull_score > bear_score:
        direction = "YES"
        confidence = bull_score / 3.0
    else:
        direction = "NO"
        confidence = bear_score / 3.0

    reasoning = (
        f"MACD:{macd_sig}({macd_hist:.2f}) | "
        f"RSI:{rsi:.1f}/VWAP:{rsi_vwap_sig} | "
        f"CVD:{cvd_div}({float(cvd.iloc[-1]):.0f})"
    )

    return TechnicalSignal(
        direction=direction,
        confidence=confidence,
        macd_signal=macd_sig,
        rsi_vwap_signal=rsi_vwap_sig,
        cvd_divergence=cvd_div,
        rsi=rsi,
        macd_hist=macd_hist,
        cvd=float(cvd.iloc[-1]),
        vwap=vwap,
        close=close,
        reasoning=reasoning,
    )
