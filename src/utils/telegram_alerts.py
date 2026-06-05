"""Telegram notification helper (non-blocking)."""
from __future__ import annotations
import asyncio
import logging
from typing import Optional

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)


async def send_alert(message: str) -> None:
    """Send a Telegram message. Silently fails if token/chat not configured."""
    token = settings.telegram_bot_token
    chat_id = settings.telegram_chat_id
    if not token or not chat_id:
        logger.debug("Telegram not configured — skipping alert")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"})
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")


def send_alert_sync(message: str) -> None:
    """Fire-and-forget sync wrapper."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(send_alert(message))
        else:
            loop.run_until_complete(send_alert(message))
    except Exception as e:
        logger.warning(f"send_alert_sync: {e}")


def trade_opened_msg(market: str, side: str, price: float, size: float, edge: float) -> str:
    emoji = "🟢" if side == "YES" else "🔴"
    return (
        f"{emoji} *TRADE OPENED*\n"
        f"Market: {market[:60]}\n"
        f"Side: {side} @ {price:.2f}c\n"
        f"Size: ${size:.2f} USDC\n"
        f"Edge: {edge:.1f}%"
    )


def trade_closed_msg(market: str, pnl: float) -> str:
    emoji = "✅" if pnl >= 0 else "❌"
    return f"{emoji} *TRADE CLOSED* | {market[:40]} | P&L: ${pnl:+.3f} USDC"


def circuit_breaker_msg(daily_pnl: float) -> str:
    return (
        f"🛑 *CIRCUIT BREAKER TRIGGERED*\n"
        f"Daily P&L: ${daily_pnl:.3f} USDC\n"
        f"All trading halted for today."
    )
