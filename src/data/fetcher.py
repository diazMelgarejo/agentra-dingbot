"""
data/fetcher.py  —  Step 2: Live Data Ingestion
CCXT async wrapper for BTC/ETH spot OHLCV.
Unchanged from v0.2.0 (asynccontextmanager pattern, no leaks).
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import structlog

logger = structlog.get_logger(__name__)
_DEFAULT_LIMIT = 200


@asynccontextmanager
async def _exchange_ctx(sandbox: bool = True):
    import ccxt.async_support as ccxt

    from core.config import get_settings
    cfg = get_settings().exchange
    cls = getattr(ccxt, cfg.name, None)
    if cls is None:
        raise ValueError(f"Unsupported exchange: {cfg.name!r}")
    exchange = cls({
        "apiKey": cfg.api_key or None, "secret": cfg.api_secret or None,
        "sandbox": sandbox, "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    try:
        yield exchange
    finally:
        await exchange.close()


async def fetch_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=_DEFAULT_LIMIT, sandbox=True) -> list[list]:
    async with _exchange_ctx(sandbox) as exchange:
        try:
            return await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        except Exception as exc:
            logger.error("fetch_ohlcv_failed", symbol=symbol, tf=timeframe, error=str(exc))
            return []


async def fetch_ohlcv_multi_timeframe(
    symbol="BTC/USDT", timeframes=None, limit=_DEFAULT_LIMIT, sandbox=True
) -> dict[str, Any]:
    from core.config import get_settings
    if timeframes is None:
        timeframes = get_settings().trading.timeframes
    results: dict[str, Any] = {}
    async with _exchange_ctx(sandbox) as exchange:
        for tf in timeframes:
            try:
                raw = await exchange.fetch_ohlcv(symbol, tf, limit=limit)
                if not raw:
                    logger.warning("empty_ohlcv", symbol=symbol, tf=tf)
                    continue
                df = _raw_to_df(raw)
                results[tf] = df
                logger.debug("ohlcv_loaded", symbol=symbol, tf=tf, candles=len(df))
            except Exception as exc:
                logger.error("fetch_multi_failed", symbol=symbol, tf=tf, error=str(exc))
            await asyncio.sleep(0.5)
    return results


async def fetch_ticker(symbol="BTC/USDT", sandbox=True) -> dict[str, Any] | None:
    async with _exchange_ctx(sandbox) as exchange:
        try:
            return await exchange.fetch_ticker(symbol)
        except Exception as exc:
            logger.error("fetch_ticker_failed", symbol=symbol, error=str(exc))
            return None


def _raw_to_df(raw: list[list]) -> Any:
    import pandas as pd
    df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    for col in ("open","high","low","close","volume"):
        df[col] = df[col].astype(float)
    return df
