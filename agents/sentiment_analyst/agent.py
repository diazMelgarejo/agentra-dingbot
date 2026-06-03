"""
agents/sentiment_analyst/agent.py  —  SuperBot v0.3.0
Reads from snapshot (already fetched by ingest_data node) rather than making
its own HTTP calls — avoids redundant network requests in the pipeline.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, Tuple, List
import structlog
from core.state import SentimentSnapshot, Signal

logger = structlog.get_logger(__name__)

_EXTREME_FEAR  = 25
_FEAR          = 40
_GREED         = 60
_EXTREME_GREED = 75


async def run(state: Dict[str, Any]) -> Dict[str, Any]:
    symbol   = state.get("symbol", "BTC/USDT")
    sent_snap = state.get("sentiment_raw", {})
    ohlcv     = state.get("ohlcv", {})

    fg_data  = sent_snap.get("fear_greed", {"value": 50, "classification": "Neutral"})
    vix      = sent_snap.get("vix")
    vix_risk = sent_snap.get("vix_risk_level", "NORMAL")
    df_5m    = ohlcv.get("5m")

    fg_value = int(fg_data.get("value", 50))
    micro    = _detect_micro_impulse(df_5m)
    sig, conf, reason = _evaluate(fg_value, vix, micro, vix_risk)

    snap = SentimentSnapshot(
        symbol=symbol, timestamp=datetime.now(timezone.utc),
        fear_greed_index=fg_value,
        fear_greed_label=fg_data.get("classification", ""),
        vix=vix, vix_risk_level=vix_risk, micro_impulse=micro,
        signal=sig, confidence=conf, reasoning=reason,
    )
    logger.info("sentiment_done", symbol=symbol, fg=fg_value, vix=vix, signal=sig.value)
    return {"sentiment": snap}


def _detect_micro_impulse(df, window: int = 6) -> str:
    if df is None or len(df) < window:
        return "NEUTRAL"
    ret = float((df["close"].iloc[-1] / df["close"].iloc[-window]) - 1) * 100
    if ret >  0.30: return "BULLISH"
    if ret < -0.30: return "BEARISH"
    return "NEUTRAL"


def _evaluate(fg: int, vix, micro: str, vix_risk: str) -> Tuple[Signal, float, str]:
    score: float = 0.0; reasons: List[str] = []
    # Fear & Greed contrarian scoring
    if   fg <= 10: score += 2.5; reasons.append(f"Extreme Fear ({fg}) — deep contrarian buy")
    elif fg <= 25: score += 2.0; reasons.append(f"Extreme Fear ({fg})")
    elif fg <= 40: score += 1.0; reasons.append(f"Fear ({fg})")
    elif fg >= 90: score -= 2.5; reasons.append(f"Extreme Greed ({fg}) — deep contrarian sell")
    elif fg >= 75: score -= 2.0; reasons.append(f"Extreme Greed ({fg})")
    elif fg >= 60: score -= 1.0; reasons.append(f"Greed ({fg})")
    else:                          reasons.append(f"Neutral fear/greed ({fg})")
    # VIX dampens everything
    if   vix_risk == "EXTREME":  score *= 0.0; reasons.append(f"VIX extreme — all signals muted")
    elif vix_risk == "ELEVATED": score *= 0.5; reasons.append(f"VIX elevated — signals halved")
    # Micro impulse confirmation
    if   micro == "BULLISH": score += 0.5; reasons.append("Micro impulse bullish")
    elif micro == "BEARISH": score -= 0.5; reasons.append("Micro impulse bearish")

    if   score >=  2.5: sig = Signal.STRONG_BUY
    elif score >=  1.0: sig = Signal.BUY
    elif score <= -2.5: sig = Signal.STRONG_SELL
    elif score <= -1.0: sig = Signal.SELL
    else:               sig = Signal.NEUTRAL
    return sig, min(abs(score) / 4.0, 1.0), " | ".join(reasons) or "No data"
