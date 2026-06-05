"""
dashboard/app.py  —  SuperBot FastAPI Dashboard + TradingView Webhook
=====================================================================
Endpoints:
  GET  /api/health               — liveness check
  GET  /api/signals              — latest cycle state snapshot
  POST /api/run                  — trigger one analysis cycle
  POST /webhook/tradingview      — TradingView alert webhook receiver
  GET  /webhook/tradingview/signals — inspect queued TV signals
  WS   /ws/signals               — WebSocket live push (Step 8)
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = structlog.get_logger(__name__)

# ── TradingView webhook source IPs (published by TradingView, rarely change)
_TV_ALLOWED_IPS = {"52.89.214.238", "34.212.75.30", "54.218.53.128", "52.32.178.7"}
TRADINGVIEW_SECRET = os.getenv("TRADINGVIEW_WEBHOOK_SECRET", "")

# ── In-process shared state ───────────────────────────────────────────────────
_latest_state: Dict[str, Any] = {}
_external_signals: List[Dict[str, Any]] = []
_MAX_STORED_SIGNALS = 50


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(title="Agentic SuperBot", version="0.3.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "version": "0.3.0",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pending_tv_signals": len(_external_signals)}

    @app.get("/api/signals")
    async def get_signals():
        return {"status": "ok" if _latest_state else "no_cycle_run",
                "data": _serialize_state(_latest_state)}

    @app.post("/api/run")
    async def trigger_run(symbol: str = "BTC/USDT"):
        asyncio.create_task(_run_cycle(symbol))
        return {"status": "cycle_started", "symbol": symbol}

    # ── TradingView webhook ───────────────────────────────────────────────────

    @app.post("/webhook/tradingview")
    async def tradingview_webhook(request: Request):
        """
        Receives TradingView alert webhooks. Configure in TradingView:
          Webhook URL : https://your-domain.com/webhook/tradingview
          Alert message: {"symbol":"{{ticker}}","action":"BUY","price":{{close}},
                          "rsi":{{plot_0}},"timeframe":"{{interval}}",
                          "strategy":"your_strategy_name"}
        """
        client_ip = request.client.host if request.client else "unknown"

        # IP check (log-only unless TV_STRICT_IP_CHECK=true)
        if client_ip not in _TV_ALLOWED_IPS:
            logger.warning("tv_webhook_unexpected_ip", ip=client_ip)
            if os.getenv("TV_STRICT_IP_CHECK", "false").lower() == "true":
                raise HTTPException(403, "IP not in TradingView allowlist")

        body = await request.body()

        # HMAC signature check (set TRADINGVIEW_WEBHOOK_SECRET in .env)
        if TRADINGVIEW_SECRET:
            expected = hmac.new(TRADINGVIEW_SECRET.encode(), body,
                                hashlib.sha256).hexdigest()
            provided = request.headers.get("X-TV-Signature", "")
            if not hmac.compare_digest(expected, provided):
                raise HTTPException(401, "Invalid webhook signature")

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"Invalid JSON: {exc}")

        # Normalise symbol (BINANCE:BTCUSDT → BTC/USDT)
        raw_sym = data.get("symbol", "")
        data["symbol_normalised"] = _normalise_symbol(raw_sym)
        data["received_at"] = datetime.now(timezone.utc).isoformat()
        data["source_ip"] = client_ip

        global _external_signals
        _external_signals.append(data)
        _external_signals = _external_signals[-_MAX_STORED_SIGNALS:]

        logger.info("tv_webhook_received",
                    symbol=data["symbol_normalised"],
                    action=data.get("action"), price=data.get("price"),
                    strategy=data.get("strategy"))

        # Auto-trigger a pipeline cycle on directional signals if configured
        if (data.get("action") in ("BUY", "SELL") and
                os.getenv("TV_AUTO_CYCLE", "false").lower() == "true"):
            asyncio.create_task(_run_cycle(data["symbol_normalised"]))

        return {"status": "received", "symbol": data["symbol_normalised"],
                "action": data.get("action"), "queued": len(_external_signals)}

    @app.get("/webhook/tradingview/signals")
    async def get_tv_signals():
        """Inspect queued TradingView signals — useful for debugging."""
        return {"count": len(_external_signals),
                "signals": _external_signals[-10:]}

    # ── WebSocket (Step 8 — currently pushes cached state every 60s) ─────────

    @app.websocket("/ws/signals")
    async def ws_signals(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                await websocket.send_json({
                    "type": "cycle_update" if _latest_state else "heartbeat",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": _serialize_state(_latest_state),
                })
                await asyncio.sleep(60)
        except WebSocketDisconnect:
            pass

    return app


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _run_cycle(symbol: str) -> None:
    global _latest_state
    try:
        from core.orchestrator import run_one_cycle
        _latest_state = await run_one_cycle(symbol, dry_run=True)
    except Exception as exc:
        logger.error("dashboard_cycle_failed", error=str(exc))


def _normalise_symbol(raw: str) -> str:
    """BINANCE:BTCUSDT → BTC/USDT  |  BTCUSDT → BTC/USDT  |  BTC/USDT → BTC/USDT"""
    s = raw.split(":")[-1].upper()
    if "/" in s:
        return s
    for quote in ("USDT", "USDC", "BTC", "ETH", "BUSD", "USD"):
        if s.endswith(quote):
            return f"{s[:-len(quote)]}/{quote}"
    return s


def _serialize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    import dataclasses
    from enum import Enum

    def _c(v):
        if isinstance(v, Enum): return v.value
        if isinstance(v, datetime): return v.isoformat()
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            return {f.name: _c(getattr(v, f.name)) for f in dataclasses.fields(v)}
        if isinstance(v, dict): return {k: _c(val) for k, val in v.items()}
        if isinstance(v, list): return [_c(i) for i in v]
        return v

    if dataclasses.is_dataclass(state) and not isinstance(state, type):
        return _c({f.name: getattr(state, f.name) for f in dataclasses.fields(state)})
    return _c(dict(state))


def get_latest_tv_signal(symbol: str) -> Optional[Dict[str, Any]]:
    """Called by agents to read the most recent injected TradingView signal."""
    norm = symbol.upper().replace("-", "/")
    for s in reversed(_external_signals):
        if s.get("symbol_normalised", "").upper() == norm:
            return s
    return None


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard.app:app", host="0.0.0.0", port=8000, reload=True)
