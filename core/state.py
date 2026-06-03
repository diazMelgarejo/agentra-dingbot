"""
core/state.py  —  Agentic SuperBot v0.3.0
Unified TradingState: BTC/ETH spot signals + Polymarket prediction market decisions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional
import operator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ─── Enums ────────────────────────────────────────────────────────────────────

class Signal(str, Enum):
    STRONG_BUY  = "STRONG_BUY"
    BUY         = "BUY"
    NEUTRAL     = "NEUTRAL"
    SELL        = "SELL"
    STRONG_SELL = "STRONG_SELL"


class Timeframe(str, Enum):
    M5 = "5m"; H1 = "1h"; H4 = "4h"; D1 = "1d"


class OrderStatus(str, Enum):
    PENDING = "pending"; DRY_RUN = "dry_run"; PLACED = "placed"
    FILLED  = "filled";  FAILED  = "failed";  CANCELLED = "cancelled"


class MarketDirection(str, Enum):
    YES = "YES"; NO = "NO"; NEUTRAL = "NEUTRAL"


# ─── Agent Snapshots (spot market) ────────────────────────────────────────────

@dataclass
class IndicatorSnapshot:
    symbol: str; timeframe: Timeframe
    timestamp: datetime = field(default_factory=_utcnow)
    close: Optional[float] = None; volume: Optional[float] = None
    rsi_14: Optional[float] = None
    bb_upper: Optional[float] = None; bb_middle: Optional[float] = None; bb_lower: Optional[float] = None
    bb_width: Optional[float] = None
    ema_9: Optional[float] = None; ema_21: Optional[float] = None
    ema_50: Optional[float] = None; ema_200: Optional[float] = None
    macd_line: Optional[float] = None; macd_signal: Optional[float] = None; macd_hist: Optional[float] = None
    # New: MACD(3,15,3) + VWAP + CVD for Polymarket signals
    macd_fast_hist: Optional[float] = None   # MACD(3,15,3) histogram
    vwap: Optional[float] = None
    cvd: Optional[float] = None              # Cumulative Volume Delta
    atr_14: Optional[float] = None
    signal: Signal = Signal.NEUTRAL
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class SentimentSnapshot:
    symbol: str; timestamp: datetime = field(default_factory=_utcnow)
    fear_greed_index: Optional[int] = None; fear_greed_label: Optional[str] = None
    news_sentiment_score: float = 0.0
    vix: Optional[float] = None             # New: VIX from yfinance
    vix_risk_level: str = "NORMAL"          # "NORMAL"|"ELEVATED"|"EXTREME"
    micro_impulse: str = "NEUTRAL"          # New: 5-min BTC micro-impulse
    signal: Signal = Signal.NEUTRAL; confidence: float = 0.0; reasoning: str = ""


@dataclass
class OnChainSnapshot:
    symbol: str; timestamp: datetime = field(default_factory=_utcnow)
    exchange_netflow: Optional[float] = None; funding_rate: Optional[float] = None
    open_interest: Optional[float] = None
    signal: Signal = Signal.NEUTRAL; confidence: float = 0.0; reasoning: str = ""


@dataclass
class RiskAssessment:
    """Spot market risk (ATR-based stops)."""
    approved: bool = False; position_size_pct: float = 0.0
    stop_loss_pct: float = 0.0; take_profit_pct: float = 0.0
    risk_reward_ratio: float = 0.0; max_loss_pct: float = 0.0; reasoning: str = ""




@dataclass
class MLSnapshot:
    """Output of the FreqAI-style ML signal bridge."""
    symbol: str
    timestamp: datetime = field(default_factory=_utcnow)
    prob_up: Optional[float] = None          # P(price up over horizon)  [0,1]
    signal: Signal = Signal.NEUTRAL
    confidence: float = 0.0                  # |prob_up - 0.5| * 2
    model_type: str = "none"                 # lightgbm | sklearn_hgb | heuristic | none
    n_features: int = 0
    n_train_samples: int = 0
    top_features: List[Dict[str, Any]] = field(default_factory=list)  # [{"feature","importance"}]
    trained_at: Optional[str] = None
    reasoning: str = ""


@dataclass
class TradeOrder:
    symbol: str; side: str; order_type: str; amount: float
    price: Optional[float] = None; stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = field(default_factory=_utcnow)
    status: OrderStatus = OrderStatus.PENDING
    exchange_order_id: Optional[str] = None


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
    ohlcv: Dict[str, Any] = field(default_factory=dict)

    # Raw ingested data (written by ingest_data node, consumed by agents)
    sentiment_raw:       Dict[str, Any] = field(default_factory=dict)   # F&G + VIX snapshot
    polymarket_snapshot: Dict[str, Any] = field(default_factory=dict)   # Gamma+CLOB snapshot

    # Spot agent outputs
    technical: Optional[IndicatorSnapshot] = None
    technical_5m: Optional[IndicatorSnapshot] = None   # 5m fast signals
    sentiment: Optional[SentimentSnapshot] = None
    onchain: Optional[OnChainSnapshot] = None
    ml:      Optional[MLSnapshot]    = None   # FreqAI ML bridge signal
    bull_case: str = ""; bear_case: str = ""
    debate_consensus: Signal = Signal.NEUTRAL; debate_confidence: float = 0.0
    risk: Optional[RiskAssessment] = None
    final_signal: Signal = Signal.NEUTRAL; final_confidence: float = 0.0
    order: Optional[TradeOrder] = None

    # Polymarket outputs
    polymarket_markets: List[PolymarketMarket] = field(default_factory=list)
    polymarket_decision: Optional[PolymarketDecision] = None
    liquidity_farming_active: bool = False

    # Audit
    errors: Annotated[List[str], operator.add] = field(default_factory=list)
    agent_log: List[Dict[str, Any]] = field(default_factory=list)
