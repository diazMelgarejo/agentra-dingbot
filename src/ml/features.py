"""
ml/features.py  —  Step 5: FreqAI ML Bridge — Feature Engineering
==================================================================
Builds a compact, self-contained feature matrix from OHLCV.

Design goals
------------
* No TA-Lib dependency — all features computed with pandas/numpy so the
  module runs anywhere (mirrors the indicators.py fallback philosophy).
* Deterministic — same input always yields the same features.
* NaN-tolerant downstream — early rows will contain NaNs from rolling
  windows; the model layer either drops them (training) or forward-fills
  the final row (inference). We do NOT silently zero-fill, which would
  bias the model.

Feature groups (≈16 columns)
----------------------------
returns      : r_1, log_r_1
lagged       : r_lag_1, r_lag_2, r_lag_3, r_lag_5
momentum     : roll_mean_5, roll_mean_10, roll_std_5, roll_std_10
oscillator   : rsi_14
trend        : macd_hist, price_vs_ema_21, price_vs_ema_50
volatility   : atr_pct_14
volume       : vol_z_20
"""
from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd


# Canonical, ordered feature list — the model relies on this exact order.
FEATURE_COLUMNS: List[str] = [
    "r_1", "log_r_1",
    "r_lag_1", "r_lag_2", "r_lag_3", "r_lag_5",
    "roll_mean_5", "roll_mean_10", "roll_std_5", "roll_std_10",
    "rsi_14",
    "macd_hist", "price_vs_ema_21", "price_vs_ema_50",
    "atr_pct_14",
    "vol_z_20",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the feature matrix from an OHLCV DataFrame.

    Parameters
    ----------
    df : DataFrame with columns [open, high, low, close, volume], time-indexed.

    Returns
    -------
    DataFrame indexed like `df`, with exactly FEATURE_COLUMNS columns.
    Early rows contain NaNs (rolling warm-up); callers decide how to handle.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    feat = pd.DataFrame(index=df.index)

    # ── Returns ────────────────────────────────────────────────────────────────
    feat["r_1"] = close.pct_change()
    feat["log_r_1"] = np.log(close / close.shift(1))

    # ── Lagged returns ──────────────────────────────────────────────────────────
    feat["r_lag_1"] = feat["r_1"].shift(1)
    feat["r_lag_2"] = feat["r_1"].shift(2)
    feat["r_lag_3"] = feat["r_1"].shift(3)
    feat["r_lag_5"] = feat["r_1"].shift(5)

    # ── Rolling momentum / dispersion of returns ─────────────────────────────────
    feat["roll_mean_5"] = feat["r_1"].rolling(5).mean()
    feat["roll_mean_10"] = feat["r_1"].rolling(10).mean()
    feat["roll_std_5"] = feat["r_1"].rolling(5).std(ddof=0)
    feat["roll_std_10"] = feat["r_1"].rolling(10).std(ddof=0)

    # ── RSI(14), Wilder smoothing (com = 13) ─────────────────────────────────────
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    feat["rsi_14"] = 100 - 100 / (1 + rs)

    # ── MACD histogram (12,26,9) ─────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_sig = macd_line.ewm(span=9, adjust=False).mean()
    feat["macd_hist"] = macd_line - macd_sig

    # ── Price relative to EMAs (normalised distance) ─────────────────────────────
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    feat["price_vs_ema_21"] = (close - ema21) / ema21
    feat["price_vs_ema_50"] = (close - ema50) / ema50

    # ── ATR% (Wilder) ─────────────────────────────────────────────────────────────
    prev_close = close.shift(1)
    tr = np.maximum.reduce([
        (high - low).values,
        np.abs((high - prev_close).values),
        np.abs((low - prev_close).values),
    ])
    atr = pd.Series(tr, index=df.index).ewm(com=13, min_periods=14).mean()
    feat["atr_pct_14"] = atr / close

    # ── Volume z-score (20) ───────────────────────────────────────────────────────
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std(ddof=0).replace(0, np.nan)
    feat["vol_z_20"] = (volume - vol_mean) / vol_std

    # Guarantee column order/contract
    return feat[FEATURE_COLUMNS]


def latest_feature_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a single-row DataFrame of the most recent fully-formed features.
    Forward-fills any residual NaNs in the final row (e.g. vol_std==0) so the
    model always receives a complete vector at inference time.
    """
    feats = build_features(df)
    if feats.empty:
        return feats
    last = feats.ffill().iloc[[-1]]
    return last
