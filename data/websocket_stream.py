"""
data/websocket_stream.py  —  Step 2: Polymarket Live Orderbook via WebSocket
Mirrors the asynccontextmanager pattern from data/fetcher.py.

Polymarket WebSocket endpoint:
  wss://ws-subscriptions-clob.polymarket.com/ws/market
  
Message types handled:
  "book"          — L2 snapshot (on subscription)
  "price_change"  — incremental orderbook delta
  "market_resolved" — market settled, stop streaming

Usage:
    async with orderbook_stream("token_id_here") as book:
        while book.is_live:
            await asyncio.sleep(1)
            print(f"mid={book.mid:.3f} spread={book.spread:.4f}")
"""
from __future__ import annotations
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Dict, Optional

import structlog

logger = structlog.get_logger(__name__)

_CLOB_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


class LocalOrderbook:
    """
    Level-2 orderbook rebuilt from REST snapshot + WebSocket incremental updates.
    Thread-safe for reads (no concurrent writes in asyncio single-thread model).
    """

    def __init__(self, token_id: str):
        self.token_id = token_id
        self.bids: Dict[str, float] = {}   # price_str → size
        self.asks: Dict[str, float] = {}
        self.is_live: bool = True
        self._ready = asyncio.Event()

    # ── Populate ──────────────────────────────────────────────────────────────

    def load_snapshot(self, snapshot: dict) -> None:
        self.bids = {b["price"]: float(b["size"]) for b in snapshot.get("bids", [])}
        self.asks = {a["price"]: float(a["size"]) for a in snapshot.get("asks", [])}
        self._ready.set()
        logger.debug("ob_snapshot", token=self.token_id[:16],
                     bids=len(self.bids), asks=len(self.asks))

    def apply_delta(self, msg: dict) -> None:
        for b in msg.get("bids", []):
            p, s = b["price"], float(b["size"])
            if s == 0: self.bids.pop(p, None)
            else:      self.bids[p] = s
        for a in msg.get("asks", []):
            p, s = a["price"], float(a["size"])
            if s == 0: self.asks.pop(p, None)
            else:      self.asks[p] = s

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def best_bid(self) -> Optional[float]:
        return max(float(p) for p in self.bids) if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return min(float(p) for p in self.asks) if self.asks else None

    @property
    def mid(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        return (bb + ba) / 2 if bb and ba else None

    @property
    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        return round(ba - bb, 4) if bb and ba else None

    @property
    def is_farmable(self) -> bool:
        """True when spread is tight enough to earn Polymarket liquidity rewards (< 6¢)."""
        return self.spread is not None and self.spread < 0.06

    async def wait_ready(self, timeout: float = 10.0) -> bool:
        try:
            await asyncio.wait_for(self._ready.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False


@asynccontextmanager
async def orderbook_stream(token_id: str):
    """
    Async context manager: yields a LocalOrderbook kept live by a background task.
    REST snapshot is fetched first; WebSocket delivers incremental deltas.

    async with orderbook_stream("token-id") as book:
        price = book.mid    # always fresh
    """
    import aiohttp
    from core.config import get_settings

    clob_api = get_settings().polymarket.clob_api
    book     = LocalOrderbook(token_id)

    # ── Step 1: REST snapshot ─────────────────────────────────────────────────
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{clob_api}/book",
                params={"token_id": token_id},
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                resp.raise_for_status()
                snap = await resp.json(content_type=None)
                book.load_snapshot(snap)
    except Exception as exc:
        logger.warning("ob_snapshot_failed", token=token_id[:16], error=str(exc))

    # ── Step 2: WebSocket background task ─────────────────────────────────────
    ws_task = asyncio.create_task(_stream_loop(book, token_id))

    try:
        yield book
    finally:
        book.is_live = False
        ws_task.cancel()
        try:
            await ws_task
        except asyncio.CancelledError:
            pass


async def _stream_loop(book: LocalOrderbook, token_id: str) -> None:
    """Background coroutine: maintain the WebSocket connection with auto-reconnect."""
    try:
        import websockets
    except ImportError:
        logger.warning("websockets_not_installed", hint="pip install websockets")
        return

    while book.is_live:
        try:
            async with websockets.connect(_CLOB_WS, ping_interval=20) as ws:
                await ws.send(json.dumps({
                    "assets_ids":             [token_id],
                    "type":                   "market",
                    "custom_feature_enabled": True,
                }))
                async for raw in ws:
                    if not book.is_live:
                        return
                    if raw == "PONG":
                        continue
                    msg = json.loads(raw)
                    etype = msg.get("event_type", "")
                    if etype in ("book", "price_change"):
                        book.apply_delta(msg)
                    elif etype == "market_resolved":
                        logger.info("market_resolved", token=token_id[:16])
                        book.is_live = False
                        return
        except asyncio.CancelledError:
            return
        except Exception as exc:
            if not book.is_live:
                return
            logger.warning("ws_reconnecting", token=token_id[:16], error=str(exc))
            await asyncio.sleep(3)
