"""
Local orderbook maintained from Polymarket CLOB WebSocket.
Supports REST snapshot + incremental updates.
Reference: https://agentbets.ai/guides/polymarket-websocket-guide/
"""
from __future__ import annotations
import asyncio
import json
import logging
from typing import Any, Dict, Optional

import websockets

from config.settings import settings

logger = logging.getLogger(__name__)


class LocalOrderbook:
    """Level-2 orderbook reconstructed from WebSocket stream."""

    def __init__(self, token_id: str):
        self.token_id = token_id
        self.bids: Dict[str, float] = {}   # price_str -> size
        self.asks: Dict[str, float] = {}
        self._ready = asyncio.Event()

    def load_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self.bids = {lvl["price"]: float(lvl["size"]) for lvl in snapshot.get("bids", [])}
        self.asks = {lvl["price"]: float(lvl["size"]) for lvl in snapshot.get("asks", [])}
        self._ready.set()
        logger.debug(f"Snapshot loaded: {len(self.bids)} bids, {len(self.asks)} asks")

    def apply_update(self, msg: Dict[str, Any]) -> None:
        for bid in msg.get("bids", []):
            p, s = bid["price"], float(bid["size"])
            if s == 0:
                self.bids.pop(p, None)
            else:
                self.bids[p] = s
        for ask in msg.get("asks", []):
            p, s = ask["price"], float(ask["size"])
            if s == 0:
                self.asks.pop(p, None)
            else:
                self.asks[p] = s

    @property
    def best_bid(self) -> Optional[float]:
        return max(float(p) for p in self.bids) if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return min(float(p) for p in self.asks) if self.asks else None

    @property
    def mid(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def is_farmable(self) -> bool:
        """True if spread is tight enough to earn liquidity rewards (< $0.06)."""
        return self.spread is not None and self.spread < 0.06

    async def wait_ready(self, timeout: float = 10.0) -> bool:
        try:
            await asyncio.wait_for(self._ready.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False


class OrderbookStreamer:
    """Streams CLOB WebSocket and maintains a LocalOrderbook."""

    def __init__(self, token_id: str):
        self.token_id = token_id
        self.book = LocalOrderbook(token_id)
        self._running = False

    async def start(self) -> None:
        import aiohttp
        # Load REST snapshot first
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{settings.clob_api}/book",
                params={"token_id": self.token_id}
            ) as resp:
                snap = await resp.json()
                self.book.load_snapshot(snap)

        self._running = True
        asyncio.create_task(self._stream())

    async def _stream(self) -> None:
        uri = settings.clob_ws_market
        while self._running:
            try:
                async with websockets.connect(uri, ping_interval=20) as ws:
                    await ws.send(json.dumps({
                        "assets_ids": [self.token_id],
                        "type": "market",
                        "custom_feature_enabled": True,
                    }))
                    async for message in ws:
                        if message == "PONG":
                            continue
                        data = json.loads(message)
                        event_type = data.get("event_type")
                        if event_type in ("book", "price_change"):
                            self.book.apply_update(data)
                        elif event_type == "market_resolved":
                            logger.info(f"Market resolved: {self.token_id}")
                            self._running = False
                            break
            except Exception as e:
                logger.warning(f"WS stream error: {e} — reconnecting in 3s")
                await asyncio.sleep(3)

    def stop(self) -> None:
        self._running = False
