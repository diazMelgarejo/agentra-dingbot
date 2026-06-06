"""
agents/ml_analyst/agent.py  —  Step 5: ML Analyst LangGraph Node
==================================================================
Wraps the FreqAI bridge as a pipeline agent. Runs in parallel with the
technical / sentiment / on-chain analysts and contributes an MLSnapshot to
the debate engine.

Contract (same shallow-dict convention as every other agent):
  input : state dict (reads `ohlcv`, `symbol`)
  output: {"ml": MLSnapshot | None, "errors": [...]}

Safety:
  * Training is CPU-bound → executed in a thread pool so the asyncio event
    loop never blocks.
  * Any failure is caught and degrades to no-signal
  an exception must never
    propagate to the LangGraph runner (which would abort the whole graph).
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import structlog

from core.state import MLSnapshot

logger = structlog.get_logger(__name__)


async def run(state: dict[str, Any]) -> dict[str, Any]:
    from core.config import get_settings

    symbol = state.get("symbol", "BTC/USDT")
    ohlcv = state.get("ohlcv", {})

    cfg = get_settings().ml
    if not cfg.enabled:
        logger.info("ml_disabled")
        return {"ml": None}

    # Choose the configured timeframe, with sensible fallbacks
    df = ohlcv.get(cfg.timeframe)
    if df is None or len(df) == 0:
        for alt in ("1h", "5m", "4h", "1d"):
            if alt in ohlcv and len(ohlcv[alt]) > 0:
                df = ohlcv[alt]
                logger.info("ml_timeframe_fallback", requested=cfg.timeframe, used=alt)
                break

    if df is None or len(df) == 0:
        msg = "ml_analyst: no OHLCV available"
        logger.warning("ml_no_data")
        return {"ml": None, "errors": [msg]}

    try:
        # Run the (potentially training) bridge off the event loop
        sig, conf, prob_up, meta = await asyncio.get_event_loop().run_in_executor(
            None, _run_bridge, df, symbol
        )
    except Exception as exc:
        # absolute backstop — never break the graph
        logger.error("ml_agent_failed", error=str(exc))
        return {"ml": None, "errors": [f"ml_analyst: {exc}"]}

    snap = MLSnapshot(
        symbol=symbol,
        timestamp=datetime.now(UTC),
        prob_up=prob_up,
        signal=sig,
        confidence=conf,
        model_type=meta.get("backend", "none"),
        n_features=meta.get("n_features", 0),
        n_train_samples=meta.get("n_train_samples", 0),
        top_features=meta.get("top_features", []),
        trained_at=meta.get("trained_at"),
        reasoning=meta.get("reasoning", ""),
    )
    logger.info("ml_done", symbol=symbol, signal=sig.value,
                prob_up=None if prob_up is None else round(prob_up, 3),
                backend=snap.model_type)
    return {"ml": snap}


def _run_bridge(df, symbol: str):
    """Synchronous helper executed in the thread pool."""
    from ml.freqai_bridge import FreqAIBridge
    bridge = FreqAIBridge()
    return bridge.generate_ml_signal(df, symbol=symbol)
