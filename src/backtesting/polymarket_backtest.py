"""
src/backtesting/polymarket_backtest.py  —  Step 6: Polymarket Validation
=========================================================================
Brier score calibration, dynamic fee model, and equity-based circuit breaker.

Key facts from research (2026):
  - Taker fees peak at 1.56% at p=0.5 on 5-min crypto markets
  - Makers pay ZERO fees and earn 20% rebate share
  - 250ms taker delay in force
  - Never hardcode fee rates — fetch feeRateBps dynamically
  - Edge floor (8%) must be measured NET of fees + delay

The fee curve is symmetric: fee = fee_rate × p × (1-p) × 4
(This is an approximation matching the "peak at 0.5" behaviour.)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# ── Brier Score Calibration ──────────────────────────────────────────────────

class BrierScore:
    """
    Mean squared error between predicted probabilities and binary outcomes.
    A perfect calibrator scores 0.0; a random guess at 0.5 scores ~0.25.

    Gate criteria (from research):
      - Score < 0.20 (usable calibration)
      - Score must BEAT the base-rate baseline by at least a small margin
    """

    @staticmethod
    def compute(probs: List[float], outcomes: List[int]) -> float:
        """
        Brier score = mean((prob_i - outcome_i)^2).
        Lower is better. Perfect = 0.0, random = 0.25.
        """
        if not probs or not outcomes or len(probs) != len(outcomes):
            raise ValueError("probs and outcomes must be non-empty and same length")
        arr = np.array(probs, dtype=float)
        out = np.array(outcomes, dtype=float)
        return float(np.mean((arr - out) ** 2))

    @staticmethod
    def passes_gate(
        score: float,
        baseline: float,
        threshold: float = 0.20,
        margin: float = 0.005,
    ) -> bool:
        """
        Two conditions must both hold:
          1. score < threshold (0.20 = usable calibration)
          2. score < baseline - margin (beats the base-rate baseline)

        The baseline is typically the Brier score of a naive predictor
        that always predicts the market price (or the historical base rate).
        """
        return score < threshold and score < (baseline - margin)

    @staticmethod
    def baseline_score(outcomes: List[int]) -> float:
        """Brier score of a naive predictor that always predicts the base rate."""
        if not outcomes:
            return 0.25
        base_rate = float(np.mean(outcomes))
        return float(np.mean([(base_rate - o) ** 2 for o in outcomes]))


# ── Dynamic Fee Model ────────────────────────────────────────────────────────

class PolymarketFeeModel:
    """
    Polymarket taker fee model.

    The fee curve is: fee_fraction = fee_rate_bps/10000 × 4 × p × (1−p)
    This produces a symmetric bell curve peaking at p=0.5.

    At p=0.5 with fee_rate_bps=100 (1%): fee = 1% × 4 × 0.5 × 0.5 = 1%
    At p=0.5 with fee_rate_bps=156 (1.56%): fee ≈ 1.56% (Polymarket 2026)

    Makers pay ZERO fees — always prefer maker orders when edge allows.
    """

    def __init__(self, fee_rate_bps: float = 156.0):
        """
        fee_rate_bps: taker fee rate in basis points (default 156 = 1.56%).
        Fetch live value from CLOB API feeRateBps — never hardcode in production.
        """
        self.fee_rate = fee_rate_bps / 10_000.0  # convert to fraction

    def cost(self, yes_price: float, stake_usdc: float) -> float:
        """
        Taker fee in USDC for a given YES price and stake.
        Fee peaks at yes_price=0.5 and approaches 0 at extremes.
        """
        # Symmetric bell curve: f(p) = fee_rate × 4 × p × (1-p)
        fee_fraction = self.fee_rate * 4.0 * yes_price * (1.0 - yes_price)
        return max(0.0, fee_fraction * stake_usdc)

    def net_edge(
        self,
        our_prob: float,
        market_price: float,
        stake_usdc: float,
    ) -> float:
        """
        Net edge in USDC after taker fees on both entry and exit.
        Returns negative if fees exceed gross edge.

        Formula:
          gross_edge_pct = (our_prob - market_price) / market_price
          fee_cost = 2 × cost(market_price, stake) / stake
          net_edge = (gross_edge_pct - fee_cost) × stake
        """
        if market_price <= 0 or market_price >= 1:
            return 0.0

        gross_edge_pct = (our_prob - market_price)  # raw edge fraction
        fee_cost       = 2.0 * self.cost(market_price, stake_usdc) / stake_usdc
        net_edge_pct   = gross_edge_pct - fee_cost
        return net_edge_pct * stake_usdc

    def is_worthwhile(
        self,
        our_prob: float,
        market_price: float,
        stake_usdc: float,
        min_edge_usdc: float = 0.0,
    ) -> bool:
        """True when net edge (after fees) exceeds the minimum threshold."""
        return self.net_edge(our_prob, market_price, stake_usdc) > min_edge_usdc


# ── Equity-Based Circuit Breaker ─────────────────────────────────────────────

def equity_circuit_breaker(
    current_equity: float,
    start_equity: float,
    limit_pct: float = 0.05,
) -> bool:
    """
    Returns True (halt) when equity has dropped by ≥ limit_pct from start-of-day.

    CRITICAL: This uses EQUITY (including unrealised open-trade P&L), NOT balance.
    Using balance silently ignores open-trade losses until they close — the exact
    failure mode documented in production incidents. An account with $1000 balance
    but $60 in losing open trades has $940 equity and should fire at the 5% limit.

    Parameters
    ----------
    current_equity : account NAV = balance + unrealised P&L from open trades
    start_equity   : NAV at start of trading day (midnight UTC)
    limit_pct      : daily loss limit as a fraction (0.05 = 5%)
    """
    if start_equity <= 0:
        return False
    drawdown = (start_equity - current_equity) / start_equity
    return drawdown >= limit_pct
