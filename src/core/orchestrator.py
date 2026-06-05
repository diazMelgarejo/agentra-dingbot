"""
core/orchestrator.py  —  Step 4: LangGraph Orchestrator
=========================================================
Builds the compiled StateGraph for the dual-pipeline SuperBot.

LangGraph 1.x API notes (verified against v1.1.3):
  - Nodes receive the STATE OBJECT directly (not a dict)
  - Nodes return a PARTIAL DICT with only the fields to update
  - StateGraph merges the returned dict back into the dataclass
  - Fan-out: multiple add_edge() calls from the same source run in parallel
  - Conditional edges: function receives state object, returns a routing key

Pipeline A — Spot BTC/ETH:
  ingest_data → [technical_analyst, sentiment_analyst, onchain_analyst] (parallel)
              → debate_engine → risk_manager →[approved]→ executor → END
                                             →[rejected]→ END

Pipeline B — Polymarket (independent, shares ingested data):
  ingest_data → polymarket_agent → END
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import structlog

from core.state import TradingState

logger = structlog.get_logger(__name__)


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_trading_graph():
    """
    Compile and return the SuperBot StateGraph.
    Returns None if langgraph is not installed (graceful degradation).
    """
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        logger.warning("langgraph_not_installed", hint="pip install langgraph")
        return None

    from agents.technical_analyst.agent import run as run_technical
    from agents.sentiment_analyst.agent import run as run_sentiment
    from agents.onchain_analyst.agent   import run as run_onchain
    from agents.debate_engine.agent     import run as run_debate
    from agents.risk_manager.agent      import run as run_risk
    from agents.executor.agent          import run as run_executor
    from agents.polymarket_agent.agent  import run as run_polymarket
    from agents.ml_analyst.agent       import run as run_ml

    graph = StateGraph(TradingState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("ingest_data",       _node_ingest_data)
    graph.add_node("technical_analyst", _wrap(run_technical))
    graph.add_node("sentiment_analyst", _wrap(run_sentiment))
    graph.add_node("onchain_analyst",   _wrap(run_onchain))
    graph.add_node("ml_analyst",        _wrap(run_ml))
    graph.add_node("polymarket_agent",  _wrap(run_polymarket))
    graph.add_node("debate_engine",     _wrap(run_debate))
    graph.add_node("risk_manager",      _wrap(run_risk))
    graph.add_node("executor",          _wrap(run_executor))

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("ingest_data")

    # ── Fan-out: ingest feeds all four parallel branches ──────────────────────
    for node in ("technical_analyst", "sentiment_analyst",
                 "onchain_analyst", "ml_analyst", "polymarket_agent"):
        graph.add_edge("ingest_data", node)

    # ── Pipeline A: spot analysis → debate → risk → conditional execute ───────
    for analyst in ("technical_analyst", "sentiment_analyst",
                    "onchain_analyst", "ml_analyst"):
        graph.add_edge(analyst, "debate_engine")

    graph.add_edge("debate_engine", "risk_manager")

    graph.add_conditional_edges(
        "risk_manager",
        _routing_should_execute,
        {"execute": "executor", "skip": END},
    )
    graph.add_edge("executor", END)

    # ── Pipeline B: Polymarket terminates independently ───────────────────────
    graph.add_edge("polymarket_agent", END)

    compiled = graph.compile()
    logger.info("graph_compiled", nodes=list(graph.nodes))
    return compiled


# ── Node wrappers ─────────────────────────────────────────────────────────────

def _wrap(agent_run_fn):
    """
    Adapter: LangGraph 1.x passes the STATE OBJECT to nodes.
    Agent functions expect a dict. This wrapper converts state → shallow dict → call → return dict.

    CRITICAL: Uses shallow field extraction (not dataclasses.asdict) so that nested
    dataclass objects (IndicatorSnapshot, SentimentSnapshot, etc.) are preserved as
    their original types. dataclasses.asdict() would deep-convert them to plain dicts,
    breaking all agents that call .signal, .confidence, etc.
    """
    import dataclasses

    async def _node(state: TradingState) -> Dict[str, Any]:
        # Shallow dict: preserves nested dataclass objects (IndicatorSnapshot etc.)
        if dataclasses.is_dataclass(state):
            state_dict = {f.name: getattr(state, f.name)
                          for f in dataclasses.fields(state)}
        else:
            state_dict = dict(state)
        result = await agent_run_fn(state_dict)
        return result or {}

    _node.__name__ = agent_run_fn.__name__
    return _node


async def _node_ingest_data(state: TradingState) -> Dict[str, Any]:
    """
    Step 1 of every cycle: fetch all data sources concurrently.
    Populates ohlcv, sentiment_raw, and polymarket_snapshot in state.
    """
    from data.snapshot import fetch_full_snapshot
    from core.config   import get_settings

    settings = get_settings()
    symbol   = state.symbol

    logger.info("ingest_start", symbol=symbol, timeframes=settings.trading.timeframes)

    snap = await fetch_full_snapshot(
        symbol=symbol,
        timeframes=settings.trading.timeframes,
        include_polymarket=True,
    )

    new_errors = snap.get("errors", [])

    tfs_loaded = list(snap.get("ohlcv", {}).keys())
    pm_count   = len(snap.get("polymarket", {}).get("enriched_markets", []))
    fg_val     = snap.get("sentiment", {}).get("fear_greed", {}).get("value", "?")
    vix_val    = snap.get("sentiment", {}).get("vix", "?")

    logger.info("ingest_done",
                symbol=symbol,
                timeframes_loaded=tfs_loaded,
                polymarket_markets=pm_count,
                fear_greed=fg_val,
                vix=vix_val,
                errors=len(snap.get("errors", [])))

    return {
        "ohlcv":               snap.get("ohlcv", {}),
        "sentiment_raw":       snap.get("sentiment", {}),
        "polymarket_snapshot": snap.get("polymarket", {}),
        "errors":              new_errors,
    }


# ── Routing function ──────────────────────────────────────────────────────────

def _routing_should_execute(state: TradingState) -> str:
    """
    Conditional edge router: 'execute' if risk approved, 'skip' otherwise.
    Also blocks execution in extreme VIX environments.
    """
    risk     = state.risk
    approved = bool(risk and getattr(risk, "approved", False))

    # Double-check VIX circuit breaker at routing level
    sentiment = state.sentiment
    if sentiment and getattr(sentiment, "vix_risk_level", "NORMAL") == "EXTREME":
        logger.warning("routing_vix_circuit_breaker")
        return "skip"

    if approved:
        logger.info("routing_execute",
                    signal=state.debate_consensus.value,
                    confidence=f"{state.debate_confidence:.1%}")
    else:
        logger.info("routing_skip",
                    reason=getattr(risk, "reasoning", "no risk assessment") if risk else "no risk")

    return "execute" if approved else "skip"


# ── Single-cycle runner (CLI / test use) ──────────────────────────────────────

async def run_one_cycle(
    symbol: str = "BTC/USDT",
    dry_run: bool = True,
) -> Dict[str, Any]:
    """
    Run one complete analysis + execution cycle.
    Returns the final state as a dict.
    Used by CLI and integration tests.
    """
    graph = build_trading_graph()
    if graph is None:
        return {"error": "langgraph_not_installed"}

    initial = TradingState(symbol=symbol, dry_run=dry_run)
    result  = await graph.ainvoke(initial)

    # result is a dict (LangGraph returns dict from ainvoke)
    signal   = result.get("final_signal")
    conf     = result.get("final_confidence", 0.0)
    order    = result.get("order")
    pm_dec   = result.get("polymarket_decision")
    errors   = result.get("errors", [])

    logger.info("cycle_complete",
                symbol=symbol,
                signal=str(signal),
                confidence=f"{conf:.1%}" if conf else "0%",
                order_placed=order is not None,
                polymarket_trade=pm_dec is not None and getattr(pm_dec, "should_trade", False),
                errors=len(errors))

    return result
