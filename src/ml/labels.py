"""
ml/labels.py  —  Step 5: FreqAI ML Bridge — Labeling
======================================================
Generates supervised training labels from OHLCV.

Default scheme: binary forward-direction.
  label = 1  if  close[t + horizon] / close[t] - 1 >  +deadzone   (UP)
  label = 0  if  close[t + horizon] / close[t] - 1 <  -deadzone   (DOWN)
  label = NaN otherwise (ambiguous / inside deadzone) → dropped in training

This matches Polymarket "BTC Up/Down" binary contracts and keeps the model
focused on directional edge rather than magnitude.

A deadzone of 0.0 yields pure up/down on the sign of the forward return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_labels(
    df: pd.DataFrame,
    horizon: int = 3,
    deadzone: float = 0.0,
) -> pd.Series:
    """
    Build forward-direction labels aligned to `df.index`.

    Parameters
    ----------
    df       : OHLCV DataFrame (needs a 'close' column).
    horizon  : how many bars ahead to look.
    deadzone : symmetric fractional band around 0 to treat as "no trade"
               (e.g. 0.001 = 0.1%). Rows inside the band become NaN.

    Returns
    -------
    Series of {0.0, 1.0, NaN}, same index as `df`. The final `horizon`
    rows are NaN (no future data yet).
    """
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)

    close = df["close"].astype(float)
    fwd_return = close.shift(-horizon) / close - 1.0

    labels = pd.Series(np.nan, index=df.index, dtype=float)
    labels[fwd_return > deadzone] = 1.0
    labels[fwd_return < -deadzone] = 0.0
    # Inside deadzone (|fwd_return| <= deadzone) stays NaN and is dropped.
    return labels


def align_xy(
    features: pd.DataFrame,
    labels: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Align features and labels, dropping any row where either side is NaN.
    Returns (X, y) ready for model.fit().
    """
    if features.empty or labels.empty:
        return features.iloc[0:0], labels.iloc[0:0]

    df = features.copy()
    df["__label__"] = labels
    df = df.dropna()
    y = df.pop("__label__")
    return df, y
