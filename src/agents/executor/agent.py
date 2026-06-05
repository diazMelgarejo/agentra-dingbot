"""
agents/executor/agent.py  —  SuperBot v0.3.0
Handles BOTH spot CCXT execution (BTC/ETH) AND Polymarket CLOB execution.
dry_run=True (default) — never places real orders without explicit --live flag.
"""
from __future__ import annotations
from typing import Any, Dict
import structlog
from core.state import OrderStatus, Signal, TradeOrder


import asyncio as _asyncio
import ccxt

_MAX_RETRIES = 3
_RETRY_ERRORS = (ccxt.RateLimitExceeded, ccxt.DDoSProtection,
                 ccxt.RequestTimeout, ccxt.ExchangeNotAvailable)
_FATAL_ERRORS = (ccxt.AuthenticationError, ccxt.PermissionDenied)

async def _place_spot_order_safe(order):
    """Wrap _place_spot_order with retry logic and kill-switch check."""
    from agents.executor.safety import KillSwitch, is_live_trading_enabled
    ks = KillSwitch()
    if ks.is_armed():
        logger.critical("executor_halted_kill_switch")
        return None

    last_exc = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return await _place_spot_order(order)
        except _FATAL_ERRORS as exc:
            logger.critical("executor_fatal_error_no_retry",
                            error=str(exc), type=type(exc).__name__)
            return None
        except ccxt.InsufficientFunds as exc:
            logger.warning("executor_insufficient_funds", error=str(exc))
            return None
        except _RETRY_ERRORS as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning("executor_retry", attempt=attempt,
                           max=_MAX_RETRIES, error=str(exc), wait=wait)
            if attempt < _MAX_RETRIES:
                await _asyncio.sleep(wait)
        except Exception as exc:
            logger.error("executor_unexpected_error", error=str(exc))
            return None
    logger.error("executor_max_retries_exhausted", error=str(last_exc))
    return None
logger = structlog.get_logger(__name__)


async def run(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute spot trade (from LangGraph pipeline). Polymarket execution is handled separately."""
    risk    = state.get("risk")
    consensus = state.get("debate_consensus", Signal.NEUTRAL)
    tech    = state.get("technical")
    symbol  = state.get("symbol", "BTC/USDT")
    dry_run = state.get("dry_run", True)

    if not risk or not risk.approved:
        return {"order": None}

    close = getattr(tech, "close", None) or 0.0
    if close <= 0:
        return {"order": None, "errors": ["executor: no price"]}

    side = "buy" if consensus in (Signal.STRONG_BUY, Signal.BUY) else "sell"
    sl   = _calc_sl(close, risk.stop_loss_pct, side)
    tp   = _calc_tp(close, risk.take_profit_pct, side)

    order = TradeOrder(
        symbol=symbol, side=side, order_type="limit", amount=0.0,
        price=round(close, 2), stop_loss=round(sl, 2), take_profit=round(tp, 2),
        status=OrderStatus.DRY_RUN,
    )

    # Detect optional FreqTrade sidecar (used only if installed & reachable)
    ft_used, ft_reason = await _maybe_freqtrade(order, side, dry_run)
    order.routed_via = "freqtrade" if ft_used else "homegrown"

    logger.info("spot_order_built", symbol=symbol, side=side, price=close,
                sl=round(sl, 2), tp=round(tp, 2), dry_run=dry_run,
                routed_via=order.routed_via, ft=ft_reason)

    # Homegrown CCXT execution only when FreqTrade did NOT handle it and live
    if not dry_run and not ft_used:
        order = await _place_spot_order(order)

    return {
        "order": order,
        "final_signal": consensus,
        "final_confidence": state.get("debate_confidence", 0.0),
    }


async def _maybe_freqtrade(order: "TradeOrder", side: str, dry_run: bool):
    """
    If a FreqTrade sidecar is installed and reachable, route the entry through it.
    Returns (used: bool, reason: str). Never raises — FreqTrade is fully optional.

    In dry_run we still *detect* FreqTrade (so logs/dashboards show the routing
    decision) but do not place a real force_entry.
    """
    try:
        from core.config import get_settings
        from agents.executor.freqtrade_client import FreqTradeClient
        ftc = get_settings().freqtrade

        use, reason = await FreqTradeClient.detect(
            ftc.base_url, ftc.username, ftc.password, mode=ftc.mode
        )
        if not use:
            return False, reason

        if dry_run:
            return True, f"{reason} (dry_run — no force_entry sent)"

        client = FreqTradeClient(ftc.base_url, ftc.username, ftc.password)
        ft_side = "long" if side == "buy" else "short"
        resp = await client.force_entry(order.symbol, side=ft_side, price=order.price)
        order.exchange_order_id = str(resp.get("trade_id") or resp.get("id") or "ft")
        order.status = OrderStatus.PLACED
        return True, f"force_entry sent ({reason})"
    except Exception as exc:
        logger.warning("freqtrade_route_failed_fallback_homegrown", error=str(exc))
        return False, f"freqtrade error: {exc}"


async def execute_polymarket_order(decision, dry_run: bool = True) -> Dict[str, Any]:
    """
    Execute a PolymarketDecision. Called directly from deploy/live.py.
    Separated from the LangGraph flow to keep the agent graph clean.
    """
    if not decision or not decision.should_trade:
        return {"placed": False, "reason": "no decision"}

    market_id = decision.market_id
    direction = decision.direction.value if hasattr(decision.direction, "value") else decision.direction
    yes_price = decision.yes_price
    size_usdc = decision.position_usdc
    shares    = round(size_usdc / yes_price, 2) if yes_price > 0 else 0

    if dry_run:
        logger.info("polymarket_dry_run", direction=direction, shares=shares,
                    price=yes_price, size_usdc=size_usdc, market=market_id[:20])
        return {"placed": True, "dry_run": True, "shares": shares, "price": yes_price}

    try:
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY, SELL
        from core.config import get_settings
        cfg = get_settings().polymarket
        from py_clob_client.client import ClobClient
        client = ClobClient(
            cfg.clob_api, key=cfg.private_key,
            chain_id=cfg.chain_id, signature_type=1,
            funder=cfg.proxy_address,
        )
        client.set_api_creds(client.create_or_derive_api_creds())
        token_id = decision.market_id   # simplification — real impl uses YES token_id
        order_args = OrderArgs(price=yes_price, size=shares, side=BUY, token_id=token_id)
        signed = client.create_order(order_args)
        resp   = client.post_order(signed, OrderType.GTC)
        logger.info("polymarket_order_placed", resp=str(resp))
        return {"placed": True, "dry_run": False, "resp": str(resp)}
    except Exception as exc:
        logger.error("polymarket_order_failed", error=str(exc))
        return {"placed": False, "error": str(exc)}


def _calc_sl(price: float, sl_pct: float, side: str) -> float:
    return price * (1 - sl_pct / 100) if side == "buy" else price * (1 + sl_pct / 100)

def _calc_tp(price: float, tp_pct: float, side: str) -> float:
    return price * (1 + tp_pct / 100) if side == "buy" else price * (1 - tp_pct / 100)


async def _place_spot_order(order: TradeOrder) -> TradeOrder:
    from data.fetcher import _exchange_ctx
    from core.config import get_settings
    settings = get_settings()
    async with _exchange_ctx(sandbox=settings.exchange.sandbox) as exchange:
        try:
            resp = await exchange.create_order(
                order.symbol, order.order_type, order.side, order.amount, order.price
            )
            order.exchange_order_id = resp.get("id")
            order.status = OrderStatus.PLACED
            logger.info("spot_order_placed", id=order.exchange_order_id)
        except Exception as exc:
            order.status = OrderStatus.FAILED
            logger.error("spot_order_failed", error=str(exc))
    return order
