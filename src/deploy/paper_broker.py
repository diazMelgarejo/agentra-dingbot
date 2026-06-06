"""
Paper broker: simulates order fills with slippage for testing.
Used by backtesting and manual paper trading validation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

SLIPPAGE_PCT = 0.002   # 0.2% slippage on fill


@dataclass
class PaperOrder:
    order_id: str
    market_id: str
    side: str                 # "YES" | "NO"
    requested_price: float
    filled_price: float
    size_usdc: float
    shares: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: str = "OPEN"      # OPEN | WIN | LOSS | CANCELLED
    pnl_usdc: float = 0.0


class PaperBroker:
    """Simulate order placement and settlement."""

    def __init__(self):
        self.orders: list[PaperOrder] = []
        self._order_counter = 0

    def place_order(
        self, market_id: str, side: str, price: float, size_usdc: float
    ) -> PaperOrder:
        fill_price = price + SLIPPAGE_PCT * (1 if side == "YES" else -1)
        fill_price = max(0.01, min(0.99, fill_price))
        shares = size_usdc / fill_price
        self._order_counter += 1
        order = PaperOrder(
            order_id=f"PAPER-{self._order_counter:06d}",
            market_id=market_id,
            side=side,
            requested_price=price,
            filled_price=fill_price,
            size_usdc=size_usdc,
            shares=shares,
        )
        self.orders.append(order)
        logger.info(f"[PAPER] {side} {shares:.2f} shares @ {fill_price:.4f}c | ${size_usdc:.2f}")
        return order

    def settle_order(self, order_id: str, resolved_price: float) -> float:
        """
        Settle an open order.
        resolved_price = 1.0 if outcome matches side, else 0.0.
        Returns realised P&L.
        """
        for order in self.orders:
            if order.order_id == order_id and order.status == "OPEN":
                payout = order.shares * resolved_price
                pnl = payout - order.size_usdc
                order.pnl_usdc = pnl
                order.status = "WIN" if pnl >= 0 else "LOSS"
                logger.info(f"Settled {order_id}: P&L ${pnl:+.3f}")
                return pnl
        return 0.0

    def summary(self) -> dict:
        closed = [o for o in self.orders if o.status in ("WIN", "LOSS")]
        wins = [o for o in closed if o.status == "WIN"]
        total_pnl = sum(o.pnl_usdc for o in closed)
        win_rate = len(wins) / len(closed) if closed else 0.0
        return {
            "total_trades": len(closed),
            "wins": len(wins),
            "losses": len(closed) - len(wins),
            "win_rate": round(win_rate, 4),
            "total_pnl_usdc": round(total_pnl, 4),
        }
