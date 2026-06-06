"""SQLite trade logger using SQLAlchemy Core."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

from config.settings import settings
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    text,
)

logger = logging.getLogger(__name__)

os.makedirs(os.path.dirname(settings.db_path) if os.path.dirname(settings.db_path) else ".", exist_ok=True)

_engine = create_engine(f"sqlite:///{settings.db_path}", echo=False)
_meta = MetaData()

trades_table = Table(
    "trades", _meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime, default=datetime.utcnow),
    Column("market_id", String),
    Column("token_id", String),
    Column("question", String),
    Column("side", String),         # YES or NO
    Column("price", Float),         # entry price
    Column("size_usdc", Float),     # position size
    Column("kelly_f", Float),       # kelly fraction used
    Column("edge_pct", Float),      # estimated edge %
    Column("signal_tech", String),  # BUY/SELL/NEUTRAL
    Column("signal_fear", String),  # BULLISH/BEARISH/NEUTRAL
    Column("confluence", Integer),  # 1 if both agreed
    Column("outcome", String),      # WIN/LOSS/OPEN/CANCELLED
    Column("pnl_usdc", Float),
    Column("paper_mode", Integer),  # 1=paper, 0=live
)

_meta.create_all(_engine)


def log_trade(trade: dict[str, Any]) -> None:
    try:
        with _engine.begin() as conn:
            conn.execute(trades_table.insert().values(**{
                k: v for k, v in trade.items()
                if k in [c.name for c in trades_table.columns]
            }))
        logger.debug(f"Trade logged: {trade.get('market_id', '?')} {trade.get('side', '?')} @ {trade.get('price', '?')}c")
    except Exception as e:
        logger.error(f"log_trade failed: {e}")


def get_daily_pnl() -> float:
    """Return today's realised P&L in USDC."""
    today = datetime.utcnow().date()
    try:
        with _engine.connect() as conn:
            result = conn.execute(
                text("SELECT SUM(pnl_usdc) FROM trades WHERE DATE(timestamp)=:d AND outcome!='OPEN'"),
                {"d": str(today)},
            ).scalar()
            return float(result or 0.0)
    except Exception:
        return 0.0


def get_open_trades() -> int:
    """Return count of currently open positions."""
    try:
        with _engine.connect() as conn:
            return conn.execute(
                text("SELECT COUNT(*) FROM trades WHERE outcome='OPEN'")
            ).scalar() or 0
    except Exception:
        return 0
