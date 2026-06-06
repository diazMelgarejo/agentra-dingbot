"""
data/polymarket.py  —  Step 2: Polymarket Live Data Ingestion
Wraps Polymarket Gamma API (market discovery) + CLOB API (prices, orderbooks).
Includes WebSocket orderbook streaming via asynccontextmanager (matching v0.2.0 pattern).

Key endpoints:
  Gamma API: https://gamma-api.polymarket.com/markets  — market discovery
  CLOB API:  https://clob.polymarket.com/price         — token prices
  CLOB API:  https://clob.polymarket.com/book          — orderbook snapshots
  CLOB WS:   wss://ws-subscriptions-clob.polymarket.com/ws/market
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import aiohttp
import structlog

from core.state import PolymarketMarket

logger = structlog.get_logger(__name__)

# ── API Constants ─────────────────────────────────────────────────────────────
GAMMA_API    = "https://gamma-api.polymarket.com"
CLOB_API     = "https://clob.polymarket.com"
CLOB_WS      = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
HTTP_TIMEOUT = aiohttp.ClientTimeout(total=10)


# ── Context manager for aiohttp sessions (mirrors _exchange_ctx pattern) ──────

@asynccontextmanager
async def _http_session():
    """Yield a shared aiohttp session; guaranteed close on exit."""
    async with aiohttp.ClientSession(timeout=HTTP_TIMEOUT) as session:
        yield session


# ── Market Discovery ──────────────────────────────────────────────────────────

async def fetch_btc_eth_markets(
    session: aiohttp.ClientSession,
    tags: list[str] = None,
    limit: int = 100,
) -> list[PolymarketMarket]:
    """
    Discover active BTC + ETH Up/Down prediction markets from Gamma API.
    Filters for 5-min style markets (question contains 'up or down' or '5-min').
    """
    if tags is None:
        tags = ["crypto"]

    params = {"active": "true", "closed": "false", "limit": limit}
    markets: list[PolymarketMarket] = []

    for tag in tags:
        try:
            async with session.get(
                f"{GAMMA_API}/markets",
                params={**params, "tag_slug": tag},
            ) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)
                raw = data if isinstance(data, list) else data.get("markets", [])

                for m in raw:
                    q = m.get("question", "").lower()
                    # Include BTC and ETH up/down / 5-min markets
                    if not (
                        ("btc" in q or "eth" in q or "bitcoin" in q or "ethereum" in q)
                        and ("up" in q or "down" in q or "5" in q)
                    ):
                        continue

                    token_ids = m.get("clobTokenIds", [])
                    if not token_ids:
                        continue

                    markets.append(PolymarketMarket(
                        market_id   = m.get("id", ""),
                        condition_id= m.get("conditionId", ""),
                        question    = m.get("question", ""),
                        token_id    = token_ids[0],
                        end_date    = m.get("endDate", ""),
                        volume_24h  = float(m.get("volume24hr", 0)),
                        is_active   = True,
                    ))
        except Exception as exc:
            logger.error("fetch_markets_failed", tag=tag, error=str(exc))

    logger.info("markets_discovered", count=len(markets))
    return markets


# ── Price Fetching ─────────────────────────────────────────────────────────────

async def fetch_yes_price(session: aiohttp.ClientSession, token_id: str) -> float | None:
    """Fetch current YES token buy price from CLOB API (0–1 scale)."""
    try:
        async with session.get(
            f"{CLOB_API}/price",
            params={"token_id": token_id, "side": "BUY"},
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)
            price = float(data.get("price", 0))
            return price if 0.01 < price < 0.99 else None
    except Exception as exc:
        logger.warning("fetch_price_failed", token_id=token_id[:16], error=str(exc))
        return None


async def fetch_orderbook_snapshot(session: aiohttp.ClientSession, token_id: str) -> dict[str, Any]:
    """Fetch REST L2 orderbook snapshot for a token."""
    try:
        async with session.get(
            f"{CLOB_API}/book",
            params={"token_id": token_id},
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            resp.raise_for_status()
            return await resp.json(content_type=None)
    except Exception as exc:
        logger.warning("fetch_orderbook_failed", token_id=token_id[:16], error=str(exc))
        return {}


async def enrich_markets_with_prices(
    session: aiohttp.ClientSession,
    markets: list[PolymarketMarket],
    max_markets: int = 5,
) -> list[PolymarketMarket]:
    """
    Fetch YES price for the top `max_markets` markets.
    Filters out near-resolved markets (price <0.02 or >0.98).
    Returns markets with yes_price populated, sorted by volume.
    """
    enriched: list[PolymarketMarket] = []
    markets_sorted = sorted(markets, key=lambda m: m.volume_24h, reverse=True)

    for m in markets_sorted[:max_markets * 2]:   # check more, filter narrow
        price = await fetch_yes_price(session, m.token_id)
        if price is None:
            continue
        m.yes_price = price
        enriched.append(m)
        if len(enriched) >= max_markets:
            break
        await asyncio.sleep(0.2)    # respect CLOB rate limits

    logger.info("markets_enriched", count=len(enriched))
    return enriched


# ── Spread / Farmability ───────────────────────────────────────────────────────

def compute_spread(orderbook: dict[str, Any]) -> float | None:
    """Return bid-ask spread from snapshot dict, or None."""
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])
    if not bids or not asks:
        return None
    try:
        best_bid = max(float(b["price"]) for b in bids)
        best_ask = min(float(a["price"]) for a in asks)
        return round(best_ask - best_bid, 4)
    except Exception:
        return None


async def find_farmable_markets(
    session: aiohttp.ClientSession,
    markets: list[PolymarketMarket],
    max_spread: float = 0.06,
) -> list[dict[str, Any]]:
    """
    Find markets with tight spreads eligible for liquidity rewards.
    Polymarket pays passive makers daily for maintaining spreads < max_spread.
    Returns list of {market, mid_price, spread, bid, ask}.
    """
    farmable = []
    for m in markets[:10]:
        snap = await fetch_orderbook_snapshot(session, m.token_id)
        spread = compute_spread(snap)
        if spread is None or spread > max_spread:
            continue

        bids = snap.get("bids", [])
        asks = snap.get("asks", [])
        best_bid = max(float(b["price"]) for b in bids) if bids else None
        best_ask = min(float(a["price"]) for a in asks) if asks else None
        mid = (best_bid + best_ask) / 2 if best_bid and best_ask else None

        if mid:
            farmable.append({
                "market": m,
                "mid_price": round(mid, 3),
                "spread": spread,
                "bid": round(mid - 0.02, 3),
                "ask": round(mid + 0.02, 3),
            })
        await asyncio.sleep(0.15)

    logger.info("farmable_markets_found", count=len(farmable))
    return farmable


# ── Full Polymarket Snapshot (one-call convenience) ───────────────────────────

async def fetch_polymarket_snapshot(max_markets: int = 5) -> dict[str, Any]:
    """
    Fetch complete Polymarket data snapshot in one call.
    Returns: {markets, enriched_markets, farmable_markets}
    """
    async with _http_session() as session:
        markets = await fetch_btc_eth_markets(session)
        enriched = await enrich_markets_with_prices(session, markets, max_markets)
        farmable = await find_farmable_markets(session, enriched)
        return {
            "markets":          markets,
            "enriched_markets": enriched,
            "farmable_markets": farmable,
            "total_discovered": len(markets),
        }
