"""
Walk-forward backtest on real historical 5-min BTC OHLCV data.

Usage:
    python backtest.py --days 30
    python backtest.py --days 60 --kelly 0.25

Data source: Binance via ccxt (free, no API key needed for public OHLCV).
Simulates 5-min Polymarket BTC Up/Down markets using actual price direction.

Output:
  - Console: Sharpe, Win Rate, Profit Factor, Max Drawdown, Trade Count
  - equity_curve.png (matplotlib)
  - backtest_trades.csv
"""
import argparse
import asyncio
import logging
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from utils.data_fetchers import fetch_btc_ohlcv
from strategies.technical_signals import generate_technical_signal
from strategies.hybrid_decision import make_decision, fractional_kelly
from deploy.paper_broker import PaperBroker

logging.basicConfig(level="INFO", format="%(levelname)-8s %(message)s")
logger = logging.getLogger("backtest")

SLIPPAGE = 0.002        # 0.2% slippage on entry
MARKET_FEE = 0.0        # Polymarket 0% fee (currently)
BANKROLL = 100.0        # starting USDC


# ── Helper: simulate 5-min market outcome from OHLCV ─────────────────────────
def simulate_market_outcome(df: pd.DataFrame, bar_idx: int, direction: str) -> float:
    """
    Returns 1.0 if direction was correct (UP/DOWN vs actual next bar close),
    else 0.0. Used to simulate Polymarket settlement.
    """
    if bar_idx + 1 >= len(df):
        return 0.0
    current = df["close"].iloc[bar_idx]
    nxt     = df["close"].iloc[bar_idx + 1]
    up = nxt >= current
    if (direction == "YES" and up) or (direction == "NO" and not up):
        return 1.0
    return 0.0


# ── Walk-forward engine ────────────────────────────────────────────────────────
async def run_backtest(days: int = 30, kelly_frac: float = 0.25) -> None:
    limit = days * 24 * 12 + 100   # 5-min bars
    logger.info(f"Fetching {days} days of 5-min BTC data (~{limit} bars)...")
    df = await fetch_btc_ohlcv("5m", min(limit, 1500))

    if df.empty or len(df) < 100:
        logger.error("Not enough data. Check network / ccxt Binance access.")
        return

    logger.info(f"Got {len(df)} bars: {df.index[0]} → {df.index[-1]}")

    broker = PaperBroker()
    equity = [BANKROLL]
    bankroll = BANKROLL
    trades = []

    WARMUP = 50   # bars needed for indicators

    # Walk-forward: train on rolling 50-bar window, test on next bar
    for i in range(WARMUP, len(df) - 1):
        window = df.iloc[max(0, i - 100):i]   # rolling 100-bar window

        # ── Run strategy signals ────────────────────────────────────────────
        tech_sig = generate_technical_signal(window)
        if tech_sig.direction == "NEUTRAL":
            equity.append(equity[-1])
            continue

        # Simplified fear signal for backtest (no live API calls)
        class MockFear:
            regime = "NEUTRAL"
            fg_value = 50
            vix = 20.0
            micro_impulse = "NEUTRAL"
            vix_risk_level = "NORMAL"
            size_multiplier = 1.0

        yes_price = 0.50 + (np.random.randn() * 0.03)  # simulate market price ≈ 50%
        yes_price = max(0.40, min(0.60, yes_price))

        decision = make_decision(tech_sig, MockFear(), yes_price, bankroll=bankroll)
        if not decision.should_trade:
            equity.append(equity[-1])
            continue

        # ── Simulate trade ─────────────────────────────────────────────────
        size = max(1.0, min(decision.position_usdc, bankroll * 0.20))
        fill_price = yes_price * (1 + SLIPPAGE if decision.direction == "YES" else 1 - SLIPPAGE)
        fill_price = max(0.01, min(0.99, fill_price))
        shares = size / fill_price

        outcome = simulate_market_outcome(df, i, decision.direction)
        payout = shares * outcome
        pnl = payout - size
        bankroll += pnl

        trade = {
            "bar": i,
            "timestamp": str(df.index[i]),
            "direction": decision.direction,
            "entry_price": round(fill_price, 4),
            "size_usdc": round(size, 4),
            "outcome": "WIN" if outcome == 1.0 else "LOSS",
            "pnl_usdc": round(pnl, 4),
            "bankroll": round(bankroll, 4),
            "edge_pct": round(decision.edge_pct, 2),
            "tech": tech_sig.direction,
            "confidence": round(tech_sig.confidence, 3),
        }
        trades.append(trade)
        equity.append(bankroll)

    # ── Metrics ───────────────────────────────────────────────────────────────
    if not trades:
        logger.warning("No trades generated. Try lowering MIN_EDGE_PCT in .env")
        return

    df_trades = pd.DataFrame(trades)
    wins = df_trades[df_trades["outcome"] == "WIN"]
    losses = df_trades[df_trades["outcome"] == "LOSS"]

    win_rate = len(wins) / len(df_trades) * 100
    total_pnl = df_trades["pnl_usdc"].sum()
    gross_profit = wins["pnl_usdc"].sum()
    gross_loss   = abs(losses["pnl_usdc"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    equity_arr = np.array(equity)
    returns = np.diff(equity_arr) / equity_arr[:-1]
    sharpe = (returns.mean() / returns.std() * np.sqrt(252 * 24 * 12)
              if returns.std() > 0 else 0.0)

    peak = equity_arr[0]
    max_dd = 0.0
    for e in equity_arr:
        peak = max(peak, e)
        dd = (peak - e) / peak * 100
        max_dd = max(max_dd, dd)

    print("
" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Period       : {days} days")
    print(f"Bars         : {len(df)}")
    print(f"Total Trades : {len(df_trades)}")
    print(f"Win Rate     : {win_rate:.1f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Sharpe Ratio : {sharpe:.2f}")
    print(f"Max Drawdown : {max_dd:.2f}%")
    print(f"Net P&L      : ${total_pnl:+.2f} ({total_pnl/BANKROLL*100:+.1f}%)")
    print(f"Final Bankroll: ${bankroll:.2f}")
    print("="*60)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    df_trades.to_csv("data/backtest_trades.csv", index=False)
    logger.info("Saved: data/backtest_trades.csv")

    # ── Equity curve plot ──────────────────────────────────────────────────────
    plt.figure(figsize=(12, 5))
    plt.plot(equity_arr, label="Equity", color="#00c8ff", linewidth=1.5)
    plt.axhline(y=BANKROLL, color="gray", linestyle="--", alpha=0.5, label="Start")
    plt.title(f"Hybrid SuperBot — {days}d Equity Curve | Sharpe {sharpe:.2f} | DD {max_dd:.1f}%")
    plt.xlabel("Bar")
    plt.ylabel("USDC")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/equity_curve.png", dpi=150)
    logger.info("Saved: data/equity_curve.png")
    plt.close()


if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",  type=int,   default=30,   help="Lookback days")
    parser.add_argument("--kelly", type=float, default=0.25, help="Kelly fraction")
    args = parser.parse_args()
    asyncio.run(run_backtest(days=args.days, kelly_frac=args.kelly))
