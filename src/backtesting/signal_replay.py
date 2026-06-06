"""
src/backtesting/signal_replay.py  —  Step 6: Signal Replay Engine
=================================================================
Saves SuperBot signals to JSON and replays them against OHLCV history
to produce TradeResult records that feed Monte Carlo and walk-forward.

Design
------
- Framework-agnostic: works whether FreqTrade is installed or not
- Slippage model: configurable pct applied to entry AND exit prices
- Fee model: configurable taker pct applied to both legs
- Produces TradeResult objects compatible with monte_carlo.run_monte_carlo()

Signal replay logic
-------------------
For each BUY signal, look forward for the next SELL signal.
Entry price = bar close at signal timestamp + slippage.
Exit  price = bar close at exit signal timestamp + slippage.
Net P&L = (exit - entry) / entry - 2 × fee_pct
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import structlog

from backtesting.monte_carlo import TradeResult

logger = structlog.get_logger(__name__)


@dataclass
class SignalRecord:
    """One signal event emitted by the SuperBot pipeline."""
    signal: str          # BUY | SELL | STRONG_BUY | STRONG_SELL | NEUTRAL
    price: float         # close price at signal time
    timestamp: str       # ISO-format UTC timestamp
    confidence: float    # 0–1


@dataclass
class BacktestMetrics:
    """Summary metrics computed from a list of TradeResult."""
    n_trades: int
    win_rate: float
    total_pnl_pct: float
    profit_factor: float
    max_dd: float
    sharpe: float


# ── Persistence ──────────────────────────────────────────────────────────────

def save_signals(records: list[SignalRecord], path: str) -> None:
    """Persist a list of SignalRecord to a JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(r) for r in records], f, indent=2)
    logger.info("signals_saved", path=path, n=len(records))


def load_signals(path: str) -> list[SignalRecord]:
    """Load SignalRecord list from a JSON file."""
    with open(path) as f:
        raw = json.load(f)
    records = [SignalRecord(**r) for r in raw]
    logger.info("signals_loaded", path=path, n=len(records))
    return records


# ── Replay ───────────────────────────────────────────────────────────────────

def replay_as_trades(
    records: list[SignalRecord],
    df: pd.DataFrame,
    slippage_pct: float = 0.001,
    fee_pct: float = 0.0004,
) -> list[TradeResult]:
    """
    Simulate trades by pairing BUY/STRONG_BUY signals with subsequent
    SELL/STRONG_SELL signals on the provided OHLCV DataFrame.

    Uses close prices at the nearest bar to each signal timestamp.
    Slippage applied as: entry_price * (1 + slippage_pct) for buys.
    Fee applied twice: entry + exit, as a fraction of notional.

    Returns list of TradeResult
    skips unmatched entries.
    """
    if df is None or df.empty or not records:
        return []

    # Pair BUY entries with next SELL exits
    buy_signals  = [r for r in records if r.signal in ("BUY", "STRONG_BUY")]
    sell_signals = [r for r in records if r.signal in ("SELL", "STRONG_SELL")]
    trades: list[TradeResult] = []

    for entry in buy_signals:
        # Find the first SELL signal that comes AFTER this BUY
        exits = [s for s in sell_signals if s.timestamp > entry.timestamp]
        if not exits:
            continue
        exit_sig = exits[0]

        # Get closest bars in the DataFrame
        entry_price = _price_at(df, entry.timestamp)
        exit_price  = _price_at(df, exit_sig.timestamp)
        if entry_price is None or exit_price is None:
            continue

        # Apply slippage: buy fills slightly higher, sell slightly lower
        fill_entry = entry_price * (1.0 + slippage_pct)
        fill_exit  = exit_price  * (1.0 - slippage_pct)

        # Net P&L = price move - 2× round-trip fees
        raw_pnl = (fill_exit - fill_entry) / fill_entry
        net_pnl = raw_pnl - 2.0 * fee_pct

        # Rough duration in bars
        dur = _bars_between(df, entry.timestamp, exit_sig.timestamp)
        trades.append(TradeResult(pnl_pct=net_pnl, duration_bars=dur,
                                   signal=entry.signal))

    logger.info("replay_done", n_signals=len(records), n_trades=len(trades),
                slippage_pct=slippage_pct, fee_pct=fee_pct)
    return trades


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(trades: list[TradeResult]) -> BacktestMetrics:
    """Compute summary statistics from a list of completed trades."""
    if not trades:
        return BacktestMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    import math
    pnls = np.array([t.pnl_pct for t in trades])

    # Win rate
    win_rate = float(np.mean(pnls > 0))

    # Total compound P&L
    total_pnl = float(np.prod(1 + pnls) - 1.0)

    # Profit factor = gross_wins / gross_losses
    gross_win  = float(np.sum(pnls[pnls > 0])) if np.any(pnls > 0) else 0.0
    gross_loss = float(abs(np.sum(pnls[pnls < 0]))) if np.any(pnls < 0) else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    # Max drawdown
    equity = np.concatenate([[1.0], np.cumprod(1 + pnls)])
    peak   = np.maximum.accumulate(equity)
    dd_arr = (peak - equity) / np.where(peak == 0, 1.0, peak)
    max_dd = float(np.max(dd_arr))

    # Sharpe (annualised, daily trade assumption)
    mu  = float(np.mean(pnls))
    sig = float(np.std(pnls, ddof=1))
    sharpe = (mu / sig * math.sqrt(252)) if sig > 0 else 0.0

    return BacktestMetrics(
        n_trades=len(trades),
        win_rate=win_rate,
        total_pnl_pct=total_pnl,
        profit_factor=profit_factor,
        max_dd=max_dd,
        sharpe=sharpe,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _price_at(df: pd.DataFrame, timestamp_str: str) -> float | None:
    """Return the close price of the bar closest to timestamp_str."""
    try:
        ts = pd.Timestamp(timestamp_str)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        idx = df.index.get_indexer([ts], method="nearest")
        if idx[0] < 0 or idx[0] >= len(df):
            return None
        return float(df["close"].iloc[idx[0]])
    except Exception:
        return None


def _bars_between(df: pd.DataFrame, ts_start: str, ts_end: str) -> int:
    """Count bars between two timestamps."""
    try:
        t0 = pd.Timestamp(ts_start).tz_localize("UTC") if pd.Timestamp(ts_start).tzinfo is None else pd.Timestamp(ts_start)
        t1 = pd.Timestamp(ts_end).tz_localize("UTC") if pd.Timestamp(ts_end).tzinfo is None else pd.Timestamp(ts_end)
        i0 = df.index.get_indexer([t0], method="nearest")[0]
        i1 = df.index.get_indexer([t1], method="nearest")[0]
        return max(1, abs(int(i1) - int(i0)))
    except Exception:
        return 1
