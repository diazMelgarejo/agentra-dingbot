"""
data/fear_greed.py  —  Step 2: Fear & Greed + VIX Live Data
Sources: alternative.me (free, no key) + yfinance ^VIX (free, no key)
"""
from __future__ import annotations
import asyncio
from typing import Any, Dict, Optional
import aiohttp
import yfinance as yf
import structlog

logger = structlog.get_logger(__name__)

_FG_URL       = "https://api.alternative.me/fng/?limit=1"
_HTTP_TIMEOUT = 8


async def fetch_fear_greed() -> Dict[str, Any]:
    """CNN Fear & Greed Index. Returns neutral (50) on failure — never raises."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                _FG_URL, timeout=aiohttp.ClientTimeout(total=_HTTP_TIMEOUT)
            ) as resp:
                resp.raise_for_status()
                data  = await resp.json(content_type=None)
                entry = data["data"][0]
                result = {
                    "value":          int(entry["value"]),
                    "classification": entry["value_classification"],
                    "timestamp":      entry.get("timestamp", ""),
                }
                logger.info("fear_greed_fetched", value=result["value"])
                return result
    except Exception as exc:
        logger.warning("fear_greed_failed", error=str(exc))
        return {"value": 50, "classification": "Neutral", "timestamp": ""}


def fetch_vix() -> Optional[float]:
    """Latest VIX close via yfinance (sync — run in executor). Returns None on failure."""
    try:
        hist = yf.Ticker("^VIX").history(period="5d")
        if hist.empty:
            return None
        val = float(hist["Close"].iloc[-1])
        logger.info("vix_fetched", value=round(val, 2))
        return val
    except Exception as exc:
        logger.warning("vix_failed", error=str(exc))
        return None


async def fetch_vix_async() -> Optional[float]:
    """Non-blocking wrapper for fetch_vix."""
    return await asyncio.get_event_loop().run_in_executor(None, fetch_vix)


async def fetch_sentiment_snapshot() -> Dict[str, Any]:
    """Fetch Fear & Greed + VIX concurrently. Returns combined snapshot dict."""
    fg, vix = await asyncio.gather(
        fetch_fear_greed(),
        fetch_vix_async(),
    )
    if   vix is None: vix_risk, size_mult = "NORMAL",   1.0
    elif vix >= 40:   vix_risk, size_mult = "EXTREME",  0.0
    elif vix >= 30:   vix_risk, size_mult = "ELEVATED", 0.5
    else:             vix_risk, size_mult = "NORMAL",   1.0

    return {
        "fear_greed":      fg,
        "vix":             vix,
        "vix_risk_level":  vix_risk,
        "size_multiplier": size_mult,
    }
