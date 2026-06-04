"""
agents/executor/freqtrade_client.py  —  Optional FreqTrade Sidecar
====================================================================
A thin async REST client for an EXISTING FreqTrade install. FreqTrade is never
a hard dependency: if it isn't installed in a default path and reachable on its
API port, the SuperBot silently uses its homegrown CCXT executor instead.

Detection strategy (in `detect()`):
  1. Look for the `freqtrade` binary in common install locations:
       - `which freqtrade` (anything on PATH)
       - ~/.venvs/freqtrade/bin/freqtrade   (our documented venv)
       - /opt/homebrew/bin/freqtrade        (Apple Silicon Homebrew)
       - /usr/local/bin/freqtrade           (Intel Homebrew / pip --user)
       - ~/.local/bin/freqtrade             (pip --user)
  2. Probe the REST API `/api/v1/ping`.
  A binary OR a reachable API is enough to consider FreqTrade "available";
  execution additionally requires the API to be reachable.

Nothing here imports the `freqtrade` package (which is GPL-3.0). We only speak
to it over HTTP, preserving this project's Apache-2.0 license.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


# Common locations a `freqtrade` binary may live in on macOS / Linux.
_DEFAULT_BINARY_PATHS = [
    "~/.venvs/freqtrade/bin/freqtrade",
    "/opt/homebrew/bin/freqtrade",
    "/usr/local/bin/freqtrade",
    "~/.local/bin/freqtrade",
    "/usr/bin/freqtrade",
]


class FreqTradeClient:
    """Async REST client for an external FreqTrade instance."""

    def __init__(self, base_url: str = "http://localhost:8080",
                 username: str = "superbot",
                 password: str = "superbot_password"):
        self.base = base_url.rstrip("/")
        self.username = username
        self.password = password

    # ── Detection ───────────────────────────────────────────────────────────────

    @staticmethod
    def find_binary() -> Optional[str]:
        """Return the path to a freqtrade binary if one exists, else None."""
        # Anything already on PATH
        on_path = shutil.which("freqtrade")
        if on_path:
            return on_path
        # Known default locations
        for raw in _DEFAULT_BINARY_PATHS:
            p = Path(os.path.expanduser(raw))
            if p.exists() and os.access(p, os.X_OK):
                return str(p)
        return None

    async def ping(self, timeout: float = 3.0) -> bool:
        """True if the FreqTrade REST API answers /ping with pong."""
        try:
            import aiohttp
            auth = aiohttp.BasicAuth(self.username, self.password)
            async with aiohttp.ClientSession(auth=auth) as s:
                async with s.get(f"{self.base}/api/v1/ping",
                                 timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                    if r.status != 200:
                        return False
                    data = await r.json(content_type=None)
                    return str(data.get("status", "")).lower() == "pong"
        except Exception:
            return False

    @classmethod
    async def detect(cls, base_url: str, username: str, password: str,
                     mode: str = "auto") -> Tuple[bool, str]:
        """
        Decide whether FreqTrade should be used.

        mode:
          "off"  → never use (returns False immediately)
          "on"   → require it (returns True only if API reachable; logs error otherwise)
          "auto" → use only if a binary exists AND the API is reachable

        Returns (use_freqtrade, reason).
        """
        if mode == "off":
            return False, "FREQTRADE_MODE=off"

        client = cls(base_url, username, password)
        api_up = await client.ping()
        binary = cls.find_binary()

        if mode == "on":
            if api_up:
                return True, f"mode=on, API reachable at {base_url}"
            logger.error("freqtrade_required_but_unreachable", url=base_url)
            return False, f"mode=on but API unreachable at {base_url}"

        # auto
        if api_up and binary:
            return True, f"detected binary={binary} + API up"
        if api_up and not binary:
            return True, "API reachable (binary not found but service is up)"
        if binary and not api_up:
            return False, f"binary found ({binary}) but API not running — start FreqTrade first"
        return False, "FreqTrade not installed / not running — using homegrown executor"

    # ── REST calls ────────────────────────────────────────────────────────────────

    async def _request(self, method: str, path: str,
                       payload: Optional[dict] = None,
                       timeout: float = 10.0) -> Dict[str, Any]:
        import aiohttp
        auth = aiohttp.BasicAuth(self.username, self.password)
        async with aiohttp.ClientSession(auth=auth) as s:
            fn = getattr(s, method.lower())
            kwargs: Dict[str, Any] = {"timeout": aiohttp.ClientTimeout(total=timeout)}
            if payload is not None:
                kwargs["json"] = payload
            async with fn(f"{self.base}/api/v1{path}", **kwargs) as r:
                r.raise_for_status()
                return await r.json(content_type=None)

    async def status(self) -> List[Dict[str, Any]]:
        return await self._request("GET", "/status")

    async def profit(self) -> Dict[str, Any]:
        return await self._request("GET", "/profit")

    async def performance(self) -> List[Dict[str, Any]]:
        return await self._request("GET", "/performance")

    async def count(self) -> Dict[str, Any]:
        return await self._request("GET", "/count")

    async def force_entry(self, pair: str, side: str = "long",
                          stake_amount: Optional[float] = None,
                          price: Optional[float] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"pair": pair, "side": side}
        if stake_amount is not None:
            payload["stakeamount"] = stake_amount
        if price is not None:
            payload["price"] = price
        logger.info("freqtrade_force_entry", pair=pair, side=side, stake=stake_amount)
        return await self._request("POST", "/forceenter", payload)

    async def force_exit(self, trade_id: str, ordertype: str = "limit") -> Dict[str, Any]:
        logger.info("freqtrade_force_exit", trade_id=trade_id)
        return await self._request("POST", "/forceexit",
                                   {"tradeid": str(trade_id), "ordertype": ordertype})
