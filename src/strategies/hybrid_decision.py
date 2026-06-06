"""
Hybrid Decision Engine: Bayesian edge check + confluence gate.

Pipeline:
  1. Technical Signal → direction (YES/NO/NEUTRAL), confidence 0–1
  2. Fear Filter → confirms/rejects direction + regime boost
  3. Bayesian Update: posterior_prob = bayes_update(market_price, tech_conf, boost)
  4. Edge = |posterior - market_price| * 100 (in percentage points)
  5. Trade only if edge >= MIN_EDGE_PCT (default 8%)

References:
  - Kelly criterion for prediction markets [navnoorbawa.substack.com/p/...]
  - Fractional Kelly (0.25x) recommended for real-world risk management
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from core.config import get_settings
from strategies.fear_filter import FearSignal, fear_confirms_direction
from strategies.technical_signals import TechnicalSignal

settings = get_settings()

logger = logging.getLogger(__name__)

# Agent weights for combining signals (can be tuned)
W_TECHNICAL = 0.60
W_FEAR      = 0.40


@dataclass
class HybridDecision:
    should_trade: bool
    direction: str              # "YES" | "NO"
    market_price: float         # current YES price from CLOB
    posterior_prob: float       # our estimated probability
    edge_pct: float             # edge in percentage points
    kelly_fraction: float       # fractional Kelly stake fraction
    position_usdc: float        # dollar amount to bet
    reasoning: str


def bayesian_update(prior: float, likelihood_ratio: float) -> float:
    """
    Update prior probability with a likelihood ratio (Bayes rule, binary).
    posterior = (prior * LR) / (prior * LR + (1-prior))
    """
    num = prior * likelihood_ratio
    return num / (num + (1 - prior)) if (num + (1 - prior)) > 0 else prior


def fractional_kelly(prob: float, price: float, fraction: float) -> float:
    """
    Fractional Kelly for a binary prediction market.
    b = (1/price - 1) when buying YES at `price` (0-1 USDC per share)
    f* = (p*b - (1-p)) / b * fraction
    Returns fraction of bankroll (0–1), clipped to [0, 0.20].
    """
    if price <= 0 or price >= 1:
        return 0.0
    b = (1 / price) - 1
    if b <= 0:
        return 0.0
    full_kelly = (prob * b - (1 - prob)) / b
    frac = max(0.0, min(0.20, full_kelly * fraction))
    return frac


def make_decision(
    tech_sig: TechnicalSignal,
    fear_sig: FearSignal,
    market_price: float,         # YES token price from CLOB (0–1)
    bankroll: float = None,
) -> HybridDecision:
    """
    Core hybrid decision logic. Returns HybridDecision with trade recommendation.
    """
    if bankroll is None:
        bankroll = settings.bankroll_usdc
    min_edge = settings.min_edge_pct
    kelly_frac = settings.kelly_fraction

    # ── Gate 1: Technical direction must not be NEUTRAL ───────────────────────
    if tech_sig.direction == "NEUTRAL" or tech_sig.confidence < 0.2:
        return HybridDecision(
            should_trade=False, direction="NEUTRAL",
            market_price=market_price, posterior_prob=market_price,
            edge_pct=0.0, kelly_fraction=0.0, position_usdc=0.0,
            reasoning=f"Tech NEUTRAL or low conf ({tech_sig.confidence:.2f})"
        )

    direction = tech_sig.direction

    # ── Gate 2: Fear filter confluence ────────────────────────────────────────
    confirmed, boost = fear_confirms_direction(fear_sig, direction)
    if not confirmed:
        return HybridDecision(
            should_trade=False, direction=direction,
            market_price=market_price, posterior_prob=market_price,
            edge_pct=0.0, kelly_fraction=0.0, position_usdc=0.0,
            reasoning=f"Fear filter rejected: {fear_sig.reasoning}"
        )

    # ── Gate 3: Bayesian edge calculation ─────────────────────────────────────
    # Combined confidence: weighted average of tech confidence + fear boost
    combined_conf = W_TECHNICAL * tech_sig.confidence + W_FEAR * min(boost / 1.5, 1.0)

    # Likelihood ratio: how much more likely is our direction given signals?
    # LR > 1 = our signals favor YES; LR < 1 = favor NO
    if direction == "YES":
        prior = market_price
        lr = 1.0 + combined_conf * 2.0   # scale: 1.0–3.0
        posterior = bayesian_update(prior, lr)
        edge_pct = (posterior - market_price) * 100
    else:
        # NO
        prior = 1 - market_price          # NO token implicit price
        lr = 1.0 + combined_conf * 2.0
        posterior_no = bayesian_update(prior, lr)
        posterior = 1 - posterior_no       # for display: posterior of YES side
        edge_pct = (prior - market_price + posterior_no - prior) * 100
        edge_pct = (posterior_no - prior) * 100

    edge_pct = abs(edge_pct)

    if edge_pct < min_edge:
        return HybridDecision(
            should_trade=False, direction=direction,
            market_price=market_price, posterior_prob=posterior,
            edge_pct=edge_pct, kelly_fraction=0.0, position_usdc=0.0,
            reasoning=f"Edge {edge_pct:.1f}% < min {min_edge}%"
        )

    # ── VIX size multiplier ───────────────────────────────────────────────────
    size_mult = fear_sig.size_multiplier

    # ── Fractional Kelly sizing ────────────────────────────────────────────────
    if direction == "YES":
        kf = fractional_kelly(posterior, market_price, kelly_frac)
    else:
        no_price = 1 - market_price
        kf = fractional_kelly(posterior_no if 'posterior_no' in dir() else 1 - posterior,
                              no_price, kelly_frac)

    kf *= size_mult
    position_usdc = round(bankroll * kf, 2)

    reasoning = (
        f"Tech:{direction}({tech_sig.confidence:.2f}) + "
        f"Fear:{fear_sig.regime}(boost×{boost}) → "
        f"Edge:{edge_pct:.1f}% | Kelly:{kf:.3f} | ${position_usdc:.2f}"
    )

    return HybridDecision(
        should_trade=True,
        direction=direction,
        market_price=market_price,
        posterior_prob=posterior,
        edge_pct=edge_pct,
        kelly_fraction=kf,
        position_usdc=position_usdc,
        reasoning=reasoning,
    )
