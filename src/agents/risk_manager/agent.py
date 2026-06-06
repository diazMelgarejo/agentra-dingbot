"""
agents/risk_manager/agent.py  —  SuperBot v0.3.0
MERGED: ATR-based spot risk (from v0.2.0) + Kelly circuit breaker (from Polymarket bot).
Handles both spot BTC/ETH trade risk AND Polymarket position risk.
"""
from __future__ import annotations

from typing import Any

import structlog

from backtesting.polymarket_backtest import equity_circuit_breaker
from core.state import RiskAssessment, Signal

logger = structlog.get_logger(__name__)

_RISK_RULES: dict[Signal, dict[str, float]] = {
    Signal.STRONG_BUY:  {"position_pct": 25.0, "sl_atr_mult": 1.5, "tp_rr": 3.0},
    Signal.BUY:         {"position_pct": 15.0, "sl_atr_mult": 2.0, "tp_rr": 2.5},
    Signal.SELL:        {"position_pct": 10.0, "sl_atr_mult": 2.0, "tp_rr": 2.0},
    Signal.STRONG_SELL: {"position_pct": 20.0, "sl_atr_mult": 1.5, "tp_rr": 2.5},
}
_MAX_LOSS_PCT    = 10.0
_FALLBACK_SL_PCT = 2.5


async def run(state: dict[str, Any]) -> dict[str, Any]:
    from core.config import get_settings
    consensus  = state.get("debate_consensus", Signal.NEUTRAL)
    confidence = state.get("debate_confidence", 0.0)
    tech       = state.get("technical")
    settings   = get_settings()
    min_conf   = settings.trading.min_confidence_threshold

    # ── Check Polymarket VIX circuit breaker ─────────────────────────────────
    sentiment = state.get("sentiment")
    if sentiment and getattr(sentiment, "vix_risk_level", "NORMAL") == "EXTREME":
        return _reject("VIX extreme — all trading halted")

    # ── Equity-based daily circuit breaker ───────────────────────────────────────
    # Uses EQUITY (NAV incl. open P&L), NOT balance — see LESSONS.md L-19
    current_equity = float(state.get("equity", 0.0) or 0.0)
    start_equity   = float(state.get("start_of_day_equity", 0.0) or 0.0)
    if current_equity > 0 and start_equity > 0 and equity_circuit_breaker(current_equity, start_equity,
                                  settings.polymarket.daily_drawdown_limit_pct / 100.0):
            return _reject(
                f"Daily equity circuit breaker: equity dropped "
                f"{(start_equity - current_equity)/start_equity:.1%} "
                f"(limit {settings.polymarket.daily_drawdown_limit_pct:.0f}%)")

    # ── Standard spot risk gates ──────────────────────────────────────────────
    if consensus == Signal.NEUTRAL:
        return _reject("NEUTRAL signal — no spot trade")
    if confidence < min_conf:
        return _reject(f"Confidence {confidence:.1%} below threshold {min_conf:.1%}")

    rules = _RISK_RULES.get(consensus)
    if rules is None:
        return _reject(f"No risk rules for {consensus.value!r}")

    atr   = getattr(tech, "atr_14", None) if tech else None
    close = getattr(tech, "close",  None) if tech else None

    if atr and close and close > 0:
        sl_dist = atr * rules["sl_atr_mult"]
        sl_pct  = (sl_dist / close) * 100
    else:
        sl_pct = _FALLBACK_SL_PCT
        logger.warning("atr_unavailable", fallback_sl_pct=_FALLBACK_SL_PCT)

    tp_pct   = sl_pct * rules["tp_rr"]
    pos_pct  = rules["position_pct"] * confidence
    max_loss = pos_pct * (sl_pct / 100)

    if max_loss > _MAX_LOSS_PCT:
        pos_pct  = _MAX_LOSS_PCT / (sl_pct / 100)
        max_loss = _MAX_LOSS_PCT
        logger.info("position_capped", new_pos_pct=round(pos_pct, 2))

    max_allowed = settings.trading.max_position_size_pct
    if pos_pct > max_allowed:
        pos_pct = max_allowed

    # ── Additional: reduce size in elevated VIX environment ───────────────────
    if sentiment and getattr(sentiment, "vix_risk_level", "NORMAL") == "ELEVATED":
        pos_pct *= 0.5
        logger.info("vix_elevated_size_halved")

    assessment = RiskAssessment(
        approved          = True,
        position_size_pct = round(pos_pct, 2),
        stop_loss_pct     = round(sl_pct, 2),
        take_profit_pct   = round(tp_pct, 2),
        risk_reward_ratio = round(rules["tp_rr"], 2),
        max_loss_pct      = round(max_loss, 4),
        reasoning=(
            f"{consensus.value} @ {confidence:.0%} → "
            f"{pos_pct:.1f}% pos | SL:{sl_pct:.1f}% | TP:{tp_pct:.1f}% | RR:{rules['tp_rr']:.1f}"
        ),
    )
    logger.info("risk_approved", pos=assessment.position_size_pct, sl=assessment.stop_loss_pct)
    return {"risk": assessment}


def _reject(reason: str) -> dict[str, Any]:
    logger.info("risk_rejected", reason=reason)
    return {"risk": RiskAssessment(approved=False, reasoning=reason)}
