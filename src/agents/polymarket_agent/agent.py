"""
agents/polymarket_agent/agent.py  —  SuperBot v0.3.0
NEW agent: bridges LangGraph pipeline → Polymarket hybrid decision engine.

Pipeline:
  1. Reads enriched Polymarket markets from state
  2. Reads 5m technical signal + fear/sentiment signal
  3. Runs hybrid_decision engine (Bayesian edge + fractional Kelly)
  4. Writes PolymarketDecision to state for executor
"""
from __future__ import annotations
from typing import Any, Dict
import structlog

from core.state import MarketDirection, PolymarketDecision, Signal

logger = structlog.get_logger(__name__)


async def run(state: Dict[str, Any]) -> Dict[str, Any]:
    from strategies.technical_signals import generate_technical_signal, TechnicalSignal
    from strategies.fear_filter       import generate_fear_signal, FearSignal, fear_confirms_direction
    from strategies.hybrid_decision   import make_decision

    ohlcv     = state.get("ohlcv", {})
    sentiment = state.get("sentiment")
    pm_snap   = state.get("polymarket_snapshot", {})
    markets   = pm_snap.get("enriched_markets", [])

    if not markets:
        logger.info("polymarket_no_markets")
        return {"polymarket_decision": None}

    # ── Build technical signal from 5m data ───────────────────────────────────
    df_5m = ohlcv.get("5m")
    if df_5m is None or df_5m.empty:
        logger.warning("polymarket_no_5m_data")
        return {"polymarket_decision": None}

    tech_sig: TechnicalSignal = generate_technical_signal(df_5m)
    logger.info("pm_tech_signal", direction=tech_sig.direction, conf=f"{tech_sig.confidence:.2f}")

    # ── Build fear signal from sentiment snapshot ──────────────────────────────
    fg_value = 50
    vix      = None
    if sentiment:
        fg_value = getattr(sentiment, "fear_greed_index", None) or 50
        vix      = getattr(sentiment, "vix", None)

    fear_sig: FearSignal = generate_fear_signal(
        {"value": fg_value}, vix, df_5m
    )

    # Skip if VIX extreme
    if fear_sig.vix_risk_level == "EXTREME":
        logger.warning("polymarket_vix_circuit_breaker", vix=vix)
        return {"polymarket_decision": PolymarketDecision(
            should_trade=False, reasoning=f"VIX extreme ({vix})"
        )}

    # ── Find best market and make decision ────────────────────────────────────
    best_decision: PolymarketDecision = PolymarketDecision(should_trade=False, reasoning="No viable market")

    from core.config import get_settings
    bankroll = get_settings().polymarket.bankroll_usdc

    for market in markets[:3]:    # check top 3 by volume
        yes_price = market.yes_price
        if yes_price <= 0:
            continue

        d = make_decision(tech_sig, fear_sig, yes_price, bankroll=bankroll)

        if d.should_trade:
            best_decision = PolymarketDecision(
                should_trade    = True,
                direction       = MarketDirection(d.direction),
                market_id       = market.market_id,
                question        = market.question,
                yes_price       = yes_price,
                posterior_prob  = d.posterior_prob,
                edge_pct        = d.edge_pct,
                kelly_fraction  = d.kelly_fraction,
                position_usdc   = d.position_usdc,
                fear_regime     = fear_sig.regime,
                tech_direction  = tech_sig.direction,
                boost_factor    = 1.0,
                reasoning       = d.reasoning,
            )
            logger.info("polymarket_trade_found",
                question=market.question[:60],
                direction=d.direction,
                edge=f"{d.edge_pct:.1f}%",
                size=f"${d.position_usdc:.2f}",
            )
            break

    return {"polymarket_decision": best_decision}
