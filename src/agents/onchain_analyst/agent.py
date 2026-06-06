"""
agents/onchain_analyst/agent.py
────────────────────────────────
On-chain / derivatives data signals:
  - Funding rate (perpetual futures) — extreme values predict reversals
  - Open interest (stub — Phase 2)
  - Exchange netflow (stub — Phase 2)

Binance perpetuals use symbol format: BTC/USDT:USDT
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import structlog

from core.state import OnChainSnapshot, Signal

logger = structlog.get_logger(__name__)


async def run(state: dict[str, Any]) -> dict[str, Any]:
    symbol = state.get("symbol", "BTC/USDT")
    perp_symbol = _to_perp_symbol(symbol)

    try:
        funding = await _fetch_funding_rate(perp_symbol)
    except Exception as exc:
        logger.error("onchain_fetch_failed", error=str(exc))
        funding = None

    sig, conf, reason = _evaluate(funding)

    snap = OnChainSnapshot(
        symbol       = symbol,
        timestamp    = datetime.now(UTC),
        funding_rate = funding,
        signal       = sig,
        confidence   = conf,
        reasoning    = reason,
    )

    logger.info("onchain_done", symbol=symbol, funding=funding, signal=sig.value)
    return {"onchain": snap}


# ─── Data fetchers ─────────────────────────────────────────────────────────────

async def _fetch_funding_rate(perp_symbol: str) -> float | None:
    """
    Fetch perpetual funding rate from Binance futures.
    Returns None on failure (agent will signal NEUTRAL).
    """
    try:
        import ccxt.async_support as ccxt

        exchange = ccxt.binance({"options": {"defaultType": "future"}})
        try:
            data = await exchange.fetch_funding_rate(perp_symbol)
            return data.get("fundingRate")
        finally:
            await exchange.close()
    except Exception as exc:
        logger.warning("funding_rate_unavailable", symbol=perp_symbol, error=str(exc))
        return None


# ─── Signal scoring ────────────────────────────────────────────────────────────

def _evaluate(funding: float | None) -> tuple[Signal, float, str]:
    """
    Funding rate interpretation:
      High positive  → longs are overcrowded → contrarian sell
      High negative  → shorts are overcrowded → contrarian buy
      Near zero      → balanced market → neutral
    """
    if funding is None:
        return Signal.NEUTRAL, 0.0, "Funding rate data unavailable"

    reasons = []
    score   = 0.0

    if funding > 0.020:
        score -= 2.0
        reasons.append(f"Extreme long funding ({funding:.4%}) — contrarian sell")
    elif funding > 0.010:
        score -= 1.5
        reasons.append(f"High long funding ({funding:.4%})")
    elif funding > 0.005:
        score -= 0.5
        reasons.append(f"Elevated funding ({funding:.4%})")
    elif funding < -0.010:
        score += 2.0
        reasons.append(f"Extreme short funding ({funding:.4%}) — contrarian buy")
    elif funding < -0.005:
        score += 1.5
        reasons.append(f"Negative funding ({funding:.4%})")
    elif funding < 0.000:
        score += 0.5
        reasons.append(f"Slightly negative funding ({funding:.4%})")
    else:
        reasons.append(f"Neutral funding ({funding:.4%})")

    if score >= 1.5 or score >= 0.5:
        sig = Signal.BUY
    elif score <= -1.5 or score <= -0.5:
        sig = Signal.SELL
    else:
        sig = Signal.NEUTRAL

    # Refine to STRONG_* for extreme values
    if score >= 2.0:
        sig = Signal.STRONG_BUY
    elif score <= -2.0:
        sig = Signal.STRONG_SELL

    confidence = min(abs(score) / 2.0, 1.0)
    return sig, confidence, " | ".join(reasons)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _to_perp_symbol(spot_symbol: str) -> str:
    """Convert BTC/USDT → BTC/USDT:USDT (Binance perp format)."""
    if ":" not in spot_symbol:
        base, quote = spot_symbol.split("/")
        return f"{base}/{quote}:{quote}"
    return spot_symbol
