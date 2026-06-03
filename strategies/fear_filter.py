"""
Strategy B: Fear & Greed Filter (regime awareness)
  - CNN Fear & Greed Index (0-100, alternative.me — free)
  - VIX (via yfinance ^VIX)
  - Rolling micro-impulse detection on 5-min BTC returns

Logic (from X hybrid bots pattern):
  Extreme Fear (<25) + bullish tech signal  → STRONG confluence (bet YES)
  Extreme Greed (>75) + bearish tech signal → STRONG confluence (bet NO)
  Neutral zone (25-75)                      → weak confirmation only
  VIX > 30                                  → elevated volatility → reduce size
  VIX > 40                                  → circuit-breaker risk → skip trade
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Fear & Greed zone thresholds
EXTREME_FEAR  = 25
FEAR          = 40
GREED         = 60
EXTREME_GREED = 75

# VIX thresholds
VIX_ELEVATED = 30
VIX_EXTREME  = 40


@dataclass
class FearSignal:
    regime: str            # "EXTREME_FEAR" | "FEAR" | "NEUTRAL" | "GREED" | "EXTREME_GREED"
    fg_value: int          # 0–100
    vix: Optional[float]   # latest VIX close
    micro_impulse: str     # "BULLISH" | "BEARISH" | "NEUTRAL"
    vix_risk_level: str    # "NORMAL" | "ELEVATED" | "EXTREME"
    size_multiplier: float # scale down in high-VIX env
    reasoning: str = ""


def classify_fg(value: int) -> str:
    if value < EXTREME_FEAR:
        return "EXTREME_FEAR"
    elif value < FEAR:
        return "FEAR"
    elif value <= GREED:
        return "NEUTRAL"
    elif value <= EXTREME_GREED:
        return "GREED"
    else:
        return "EXTREME_GREED"


def detect_micro_impulse(df_5m: pd.DataFrame, window: int = 6) -> str:
    """
    Detect micro momentum impulse from last `window` 5-min bars.
    Returns BULLISH if returns strongly positive, BEARISH if strongly negative.
    Threshold: 0.3% move in either direction.
    """
    if df_5m is None or len(df_5m) < window:
        return "NEUTRAL"
    recent = df_5m["close"].iloc[-window:]
    ret = float((recent.iloc[-1] / recent.iloc[0]) - 1) * 100  # pct
    if ret >  0.30:
        return "BULLISH"
    elif ret < -0.30:
        return "BEARISH"
    return "NEUTRAL"


def compute_rolling_correlation(df_5m: pd.DataFrame, fg_value: int, window: int = 12) -> float:
    """
    Compute correlation between BTC 5m returns and Fear & Greed regime
    as a simple proxy (F&G is daily so this is a scalar comparison).
    Returns -1 to 1.
    """
    if df_5m is None or len(df_5m) < window:
        return 0.0
    rets = df_5m["close"].pct_change().iloc[-window:].dropna()
    # Positive returns during fear → counter-trend; check alignment
    mean_ret = float(rets.mean())
    if fg_value < 40 and mean_ret > 0:
        return 0.6   # fear + rising = contrarian bull signal
    elif fg_value > 60 and mean_ret < 0:
        return -0.6  # greed + falling = contrarian bear signal
    return 0.0


def generate_fear_signal(
    fg_data: dict,
    vix: Optional[float],
    df_5m: Optional[pd.DataFrame] = None,
) -> FearSignal:
    """
    Build FearSignal from Fear & Greed dict, VIX float, and optional 5m OHLCV.
    """
    fg_value = int(fg_data.get("value", 50))
    regime = classify_fg(fg_value)

    # VIX risk level + size multiplier
    if vix is None:
        vix_risk = "NORMAL"
        size_mult = 1.0
    elif vix >= VIX_EXTREME:
        vix_risk = "EXTREME"
        size_mult = 0.0   # do not trade
    elif vix >= VIX_ELEVATED:
        vix_risk = "ELEVATED"
        size_mult = 0.5
    else:
        vix_risk = "NORMAL"
        size_mult = 1.0

    # Micro impulse
    micro = detect_micro_impulse(df_5m) if df_5m is not None else "NEUTRAL"

    reasoning = (
        f"F&G:{fg_value}({regime}) | "
        f"VIX:{vix or 'N/A'}({vix_risk}) | "
        f"Micro:{micro}"
    )

    return FearSignal(
        regime=regime,
        fg_value=fg_value,
        vix=vix,
        micro_impulse=micro,
        vix_risk_level=vix_risk,
        size_multiplier=size_mult,
        reasoning=reasoning,
    )


def fear_confirms_direction(fear_sig: FearSignal, tech_direction: str) -> tuple[bool, float]:
    """
    Returns (confirmed, boost_factor):
      confirmed=True if fear regime aligns with tech direction.
      boost_factor: multiplier for edge calculation (1.0–1.5).
    """
    if fear_sig.vix_risk_level == "EXTREME":
        return False, 0.0

    r = fear_sig.regime
    if tech_direction == "YES":
        if r == "EXTREME_FEAR":
            return True, 1.5    # contrarian strong buy
        elif r == "FEAR":
            return True, 1.2
        elif r == "NEUTRAL" and fear_sig.micro_impulse == "BULLISH":
            return True, 1.0
        else:
            return False, 0.0
    elif tech_direction == "NO":
        if r == "EXTREME_GREED":
            return True, 1.5    # contrarian strong sell
        elif r == "GREED":
            return True, 1.2
        elif r == "NEUTRAL" and fear_sig.micro_impulse == "BEARISH":
            return True, 1.0
        else:
            return False, 0.0
    else:
        return False, 0.0
