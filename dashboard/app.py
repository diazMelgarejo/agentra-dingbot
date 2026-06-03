"""
dashboard/app.py
─────────────────
FastAPI application serving the trading signal API.

Endpoints:
  GET /              — health check
  GET /api/health    — detailed health with agent status
  GET /api/signals   — latest signals for a symbol (Phase 2: wire to DB/Redis)
  GET /api/state     — full TradingState of last cycle (Phase 2)
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ─── Response models ──────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    agents: list[str]


class SignalResponse(BaseModel):
    symbol: str
    timestamp: str
    technical: dict | None  = None
    sentiment: dict | None  = None
    onchain:   dict | None  = None
    consensus: str | None   = None
    confidence: float | None = None
    risk_approved: bool | None = None


# ─── App factory ──────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title       = "Agentic Trader API",
        description = "Multi-agent crypto trading signal platform",
        version     = "0.2.0",
        docs_url    = "/docs",
        redoc_url   = "/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins     = ["*"],   # tighten in production
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.get("/", tags=["Meta"])
    async def root():
        return {
            "service":   "agentic-trader",
            "version":   "0.2.0",
            "timestamp": _now(),
            "docs":      "/docs",
        }

    @app.get("/api/health", response_model=HealthResponse, tags=["Meta"])
    async def health():
        return HealthResponse(
            status    = "healthy",
            version   = "0.2.0",
            timestamp = _now(),
            agents    = [
                "technical_analyst",
                "sentiment_analyst",
                "onchain_analyst",
                "debate_engine",
                "risk_manager",
                "executor",
            ],
        )

    @app.get("/api/signals", response_model=SignalResponse, tags=["Signals"])
    async def signals(
        symbol: str = Query("BTC/USDT", description="Trading pair, e.g. BTC/USDT"),
    ):
        """
        Returns latest signals.
        Phase 2: pull from Redis/DB instead of returning empty.
        """
        return SignalResponse(
            symbol    = symbol,
            timestamp = _now(),
        )

    @app.get("/api/run", tags=["Signals"])
    async def run_cycle(
        symbol:  str  = Query("BTC/USDT"),
        dry_run: bool = Query(True, description="Dry-run mode (no live orders)"),
    ):
        """
        Trigger a full analysis cycle on demand.
        Phase 2: move to background task / WebSocket push.
        """
        from core.orchestrator import build_trading_graph
        from core.state import TradingState

        graph = build_trading_graph()
        if graph is None:
            return {"error": "LangGraph not installed"}

        result = await graph.ainvoke(TradingState(symbol=symbol, dry_run=dry_run).__dict__)
        order  = result.get("order")

        return {
            "symbol":        symbol,
            "timestamp":     _now(),
            "final_signal":  str(result.get("final_signal")),
            "confidence":    result.get("final_confidence"),
            "risk_approved": result.get("risk") and result["risk"].approved,
            "order": {
                "side":        order.side        if order else None,
                "price":       order.price       if order else None,
                "stop_loss":   order.stop_loss   if order else None,
                "take_profit": order.take_profit if order else None,
                "status":      order.status.value if order else None,
            },
            "errors": result.get("errors", []),
        }

    return app


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
