"""
core/state.py  —  Agentic SuperBot v0.3.0
Unified TradingState: BTC/ETH spot signals + Polymarket prediction market decisions.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any


def _utcnow() -> datetime:
    return datetime.now(UTC)


# ─── Enums ────────────────────────────────────────────────────────────────────

class Signal(StrEnum):
    STRONG_BUY  = "STRONG_BUY"
    BUY         = "BUY"
    NEUTRAL     = "NEUTRAL"
    SELL        = "SELL"
    STRONG_SELL = "STRONG_SELL"


class Timeframe(StrEnum):
    M5 = "5m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


class OrderStatus(StrEnum):
    PENDING = "pending"
    DRY_RUN = "dry_run"
    PLACED = "placed"
    FILLED  = "filled"
    FAILED  = "failed"
    CANCELLED = "cancelled"


class MarketDirection(StrEnum):
    YES = "YES"
    NO = "NO"
    NEUTRAL = "NEUTRAL"


# ─── Agent Snapshots (spot market) ────────────────────────────────────────────

@dataclass
class IndicatorSnapshot:
    symbol: str
    timeframe: Timeframe
    timestamp: datetime = field(default_factory=_utcnow)
    close: float | None = None
    volume: float | None = None
    rsi_14: float | None = None
    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
    bb_width: float | None = None
    ema_9: float | None = None
    ema_21: float | None = None
    ema_50: float | None = None
    ema_200: float | None = None
    macd_line: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None
    # New: MACD(3,15,3) + VWAP + CVD for Polymarket signals
    macd_fast_hist: float | None = None   # MACD(3,15,3) histogram
    vwap: float | None = None
    cvd: float | None = None              # Cumulative Volume Delta
    atr_14: float | None = None
    signal: Signal = Signal.NEUTRAL
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class SentimentSnapshot:
    symbol: str
    timestamp: datetime = field(default_factory=_utcnow)
    fear_greed_index: int | None = None
    fear_greed_label: str | None = None
    news_sentiment_score: float = 0.0
    vix: float | None = None             # New: VIX from yfinance
    vix_risk_level: str = "NORMAL"          # "NORMAL"|"ELEVATED"|"EXTREME"
    micro_impulse: str = "NEUTRAL"          # New: 5-min BTC micro-impulse
    signal: Signal = Signal.NEUTRAL
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class OnChainSnapshot:
    symbol: str
    timestamp: datetime = field(default_factory=_utcnow)
    exchange_netflow: float | None = None
    funding_rate: float | None = None
    open_interest: float | None = None
    signal: Signal = Signal.NEUTRAL
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class RiskAssessment:
    """Spot market risk (ATR-based stops)."""
    approved: bool = False
    position_size_pct: float = 0.0
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0
    risk_reward_ratio: float = 0.0
    max_loss_pct: float = 0.0
    reasoning: str = ""




@dataclass
class MLSnapshot:
    """Output of the FreqAI-style ML signal bridge."""
    symbol: str
    timestamp: datetime = field(default_factory=_utcnow)
    prob_up: float | None = None          # P(price up over horizon)  [0,1]
    signal: Signal = Signal.NEUTRAL
    confidence: float = 0.0                  # |prob_up - 0.5| * 2
    model_type: str = "none"                 # lightgbm | sklearn_hgb | heuristic | none
    n_features: int = 0
    n_train_samples: int = 0
    top_features: list[dict[str, Any]] = field(default_factory=list)  # [{"feature","importance"}]
    trained_at: str | None = None
    reasoning: str = ""


@dataclass
class TradeOrder:
    symbol: str
    side: str
    order_type: str
    amount: float
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    timestamp: datetime = field(default_factory=_utcnow)
    status: OrderStatus = OrderStatus.PENDING
    exchange_order_id: str | None = None
    routed_via: str = "homegrown"   # "homegrown" | "freqtrade"


# ─── Polymarket-specific snapshots ────────────────────────────────────────────

@dataclass
class PolymarketMarket:
    """A discovered Polymarket prediction market."""
    market_id: str = ""
    condition_id: str = ""
    question: str = ""
    token_id: str = ""             # YES token
    yes_price: float = 0.50        # current YES token price (0–1)
    end_date: str = ""
    volume_24h: float = 0.0
    is_active: bool = True


@dataclass
class PolymarketDecision:
    """Output of the Polymarket hybrid decision engine."""
    should_trade: bool = False
    direction: MarketDirection = MarketDirection.NEUTRAL
    market_id: str = ""
    question: str = ""
    yes_price: float = 0.50
    posterior_prob: float = 0.50
    edge_pct: float = 0.0
    kelly_fraction: float = 0.0
    position_usdc: float = 0.0
    fear_regime: str = "NEUTRAL"
    tech_direction: str = "NEUTRAL"
    boost_factor: float = 1.0
    reasoning: str = ""


# ─── Unified Top-Level Graph State ────────────────────────────────────────────

@dataclass
class TradingState:
    """
    Single state object through LangGraph pipeline.
    Covers both spot BTC/ETH trading AND Polymarket prediction markets.
    """
    # Identity
    symbol: str = "BTC/USDT"
    dry_run: bool = True
    timestamp: datetime = field(default_factory=_utcnow)

    # OHLCV data (keyed by timeframe)
    ohlcv: dict[str, Any] = field(default_factory=dict)

    # Raw ingested data (written by ingest_data node, consumed by agents)
    sentiment_raw:       dict[str, Any] = field(default_factory=dict)   # F&G + VIX snapshot
    polymarket_snapshot: dict[str, Any] = field(default_factory=dict)   # Gamma+CLOB snapshot

    # Spot agent outputs
    technical: IndicatorSnapshot | None = None
    technical_5m: IndicatorSnapshot | None = None   # 5m fast signals
    sentiment: SentimentSnapshot | None = None
    onchain: OnChainSnapshot | None = None
    ml:      MLSnapshot | None    = None   # FreqAI ML bridge signal
    bull_case: str = ""
    bear_case: str = ""
    debate_consensus: Signal = Signal.NEUTRAL
    debate_confidence: float = 0.0
    risk: RiskAssessment | None = None
    final_signal: Signal = Signal.NEUTRAL
    final_confidence: float = 0.0
    order: TradeOrder | None = None

    # Polymarket outputs
    polymarket_markets: list[PolymarketMarket] = field(default_factory=list)
    polymarket_decision: PolymarketDecision | None = None
    liquidity_farming_active: bool = False

    # Audit
    errors: Annotated[list[str], operator.add] = field(default_factory=list)
    agent_log: list[dict[str, Any]] = field(default_factory=list)
