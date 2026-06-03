"""
data/snapshot.py  —  Unified Multi-Source Data Snapshot
Aggregates all data sources (CCXT + Polymarket + Fear&Greed + VIX)
into a single async call for one decision cycle.

This is the primary entry point for the LangGraph ingest_data node.
"""
from __future__ import annotations
import asyncio
from typing import Any, Dict, List, Optional
import structlog

from data.fetcher      import fetch_ohlcv_multi_timeframe
from data.fear_greed   import fetch_sentiment_snapshot
from data.polymarket   import fetch_polymarket_snapshot

logger = structlog.get_logger(__name__)


async def fetch_full_snapshot(
    symbol: str = "BTC/USDT",
    timeframes: Optional[List[str]] = None,
    include_polymarket: bool = True,
    polymarket_max_markets: int = 5,
) -> Dict[str, Any]:
    """
    Fetch all data needed for one complete SuperBot decision cycle.

    Returns:
        {
            "ohlcv":        {tf: DataFrame},      # CCXT multi-timeframe OHLCV
            "sentiment":    {fear_greed, vix, ...}, # Fear & Greed + VIX
            "polymarket":   {markets, enriched, farmable},  # optional
            "symbol":       str,
            "errors":       [str],
        }
    """
    result: Dict[str, Any] = {"symbol": symbol, "errors": []}
    tasks: Dict[str, Any] = {
        "ohlcv":     fetch_ohlcv_multi_timeframe(symbol, timeframes),
        "sentiment": fetch_sentiment_snapshot(),
    }
    if include_polymarket:
        tasks["polymarket"] = fetch_polymarket_snapshot(polymarket_max_markets)

    # Run all fetches concurrently
    fetched = await asyncio.gather(*tasks.values(), return_exceptions=True)

    for key, value in zip(tasks.keys(), fetched):
        if isinstance(value, Exception):
            logger.error("snapshot_fetch_failed", source=key, error=str(value))
            result["errors"].append(f"{key}: {value}")
            result[key] = {} if key != "ohlcv" else {}
        else:
            result[key] = value

    loaded_tfs = list(result.get("ohlcv", {}).keys())
    pm_count   = len(result.get("polymarket", {}).get("enriched_markets", []))
    fg_val     = result.get("sentiment", {}).get("fear_greed", {}).get("value", "N/A")
    vix_val    = result.get("sentiment", {}).get("vix", "N/A")

    logger.info(
        "snapshot_complete",
        symbol=symbol,
        timeframes=loaded_tfs,
        polymarket_markets=pm_count,
        fear_greed=fg_val,
        vix=vix_val,
        errors=len(result["errors"]),
    )
    return result
