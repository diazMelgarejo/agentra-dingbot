"""
src/backtesting/backtest.py  —  Step 6: Integrated Backtest Runner
====================================================================
Runs a complete backtesting pipeline:
  1. Fetch real Binance OHLCV (free, no API key)
  2. Generate SuperBot signals via LangGraph pipeline (or simple TA fallback)
  3. Replay signals on OHLCV via signal_replay.py
  4. Validate with Monte Carlo (P95 gate)
  5. Walk-forward cross-validation (fold consistency gate)

Usage
-----
    python src/backtesting/backtest.py --days 30
    python src/backtesting/backtest.py --days 90 --sims 5000
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from backtesting.monte_carlo import TradeResult, run_monte_carlo  # noqa: E402
from backtesting.signal_replay import SignalRecord, compute_metrics, replay_as_trades  # noqa: E402
from backtesting.walk_forward import WalkForwardValidator  # noqa: E402

logger = structlog.get_logger("backtest")


# ── Data fetching ─────────────────────────────────────────────────────────────

async def _fetch_ohlcv(symbol: str = "BTC/USDT", days: int = 30) -> pd.DataFrame:
    """Fetch hourly OHLCV from Binance (public, no key)."""
    from data.fetcher import fetch_ohlcv_multi_timeframe
    dfs = await fetch_ohlcv_multi_timeframe(symbol, timeframes=["1h"], limit=days * 24)
    df = dfs.get("1h")
    if df is None or df.empty:
        raise RuntimeError(f"No OHLCV data returned for {symbol}")
    logger.info("ohlcv_loaded", symbol=symbol, bars=len(df))
    return df


# ── Simple signal generator (TA only, no LLM required) ───────────────────────

def _generate_ta_signals(df: pd.DataFrame) -> list[SignalRecord]:
    """
    Rule-based signal generator for backtesting (no LLM, no API key).
    EMA 9/21 crossover + RSI filter:
      BUY  when EMA9 crosses above EMA21 and RSI < 60
      SELL when EMA9 crosses below EMA21 and RSI > 40
    """
    df = df.copy()
    df["ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

    delta = df["close"].diff()
    gain  = delta.clip(lower=0).ewm(com=13, min_periods=14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, min_periods=14, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Crossover detection
    df["ema_cross"] = (df["ema9"] > df["ema21"]).astype(int)
    df["cross_up"]  = (df["ema_cross"] == 1) & (df["ema_cross"].shift(1) == 0)
    df["cross_dn"]  = (df["ema_cross"] == 0) & (df["ema_cross"].shift(1) == 1)

    records: list[SignalRecord] = []
    for ts, row in df.iterrows():
        if row["cross_up"] and row["rsi"] < 60:
            records.append(SignalRecord(
                signal="BUY", price=float(row["close"]),
                timestamp=ts.isoformat(), confidence=0.65))
        elif row["cross_dn"] and row["rsi"] > 40:
            records.append(SignalRecord(
                signal="SELL", price=float(row["close"]),
                timestamp=ts.isoformat(), confidence=0.65))

    logger.info("signals_generated", n=len(records),
                buys=sum(1 for r in records if r.signal == "BUY"),
                sells=sum(1 for r in records if r.signal == "SELL"))
    return records


# ── Walk-forward signal fn ────────────────────────────────────────────────────

def _wf_signal_fn(df_train: pd.DataFrame, df_test: pd.DataFrame) -> list[dict]:
    """Walk-forward fold: generate signals on test data, replay, return trades."""
    signals = _generate_ta_signals(df_test)
    trades  = replay_as_trades(signals, df_test, slippage_pct=0.001, fee_pct=0.0004)
    return [{"pnl_pct": t.pnl_pct, "signal": t.signal} for t in trades]


# ── Main backtest pipeline ────────────────────────────────────────────────────

async def run_backtest(
    symbol: str = "BTC/USDT",
    days:   int = 30,
    n_sims: int = 1000,
    capital: float = 1000.0,
    max_p95_dd: float = 0.30,
    save_signals: bool = False,
) -> dict:
    """
    Full Step 6 backtest pipeline.
    Returns a result dict with all metrics and gate decisions.
    """
    # 1. Data
    df = await _fetch_ohlcv(symbol, days)

    # 2. Signals
    signals = _generate_ta_signals(df)
    if save_signals:
        from backtesting.signal_replay import save_signals as save_fn
        path = str(_ROOT / "data" / "backtest_signals.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_fn(signals, path)
        logger.info("signals_saved", path=path)

    # 3. Signal replay → trades
    trades = replay_as_trades(signals, df, slippage_pct=0.001, fee_pct=0.0004)
    if not trades:
        logger.warning("no_trades_generated")
        return {"status": "no_trades", "n_trades": 0}

    # 4. Summary metrics
    metrics = compute_metrics(trades)

    # 5. Monte Carlo validation (P95 gate)
    mc_trades = [TradeResult(pnl_pct=t.pnl_pct, signal=t.signal) for t in trades]
    mc_report = run_monte_carlo(mc_trades, n_sims=n_sims, capital=capital, seed=42)
    mc_pass, mc_reason = mc_report.passes_gate(max_p95_dd=max_p95_dd)

    # 6. Walk-forward validation
    wfv = WalkForwardValidator(train_bars=200, test_bars=50, embargo_bars=10)
    wf_folds = wfv.run(df, _wf_signal_fn)
    wf_report = wfv.report(wf_folds)
    wf_pass, wf_reason = wf_report.passes_gate()

    result = {
        "status":          "pass" if (mc_pass and wf_pass) else "fail",
        "symbol":          symbol,
        "days":            days,
        "n_signals":       len(signals),
        "n_trades":        metrics.n_trades,
        "win_rate":        round(metrics.win_rate, 4),
        "sharpe":          round(metrics.sharpe, 3),
        "profit_factor":   round(metrics.profit_factor, 3),
        "max_dd":          round(metrics.max_dd, 4),
        "total_pnl_pct":   round(metrics.total_pnl_pct, 4),
        "mc_p50_dd":       round(mc_report.p50_max_dd, 4),
        "mc_p95_dd":       round(mc_report.p95_max_dd, 4),   # ← size capital here
        "mc_p95_ratio":    round(mc_report.p95_max_dd / max(mc_report.p50_max_dd, 0.001), 2),
        "mc_prob_profit":  round(mc_report.prob_profit, 4),
        "mc_pass":         mc_pass,
        "mc_reason":       mc_reason,
        "wf_n_folds":      wf_report.n_folds,
        "wf_consistent":   round(wf_report.consistent_folds_pct, 2),
        "wf_median_sharpe":round(wf_report.median_sharpe, 3),
        "wf_pass":         wf_pass,
        "wf_reason":       wf_reason,
    }
    return result


def _print_report(r: dict) -> None:
    print(f"\n{'='*60}")
    print(f"BACKTEST REPORT — {r['symbol']}  ({r['days']} days)")
    print(f"{'='*60}")
    print(f"  Trades:        {r['n_trades']}  (from {r['n_signals']} signals)")
    print(f"  Win rate:      {r['win_rate']:.1%}")
    print(f"  Sharpe:        {r['sharpe']:.2f}")
    print(f"  Profit factor: {r['profit_factor']:.2f}")
    print(f"  Backtest DD:   {r['max_dd']:.1%}")
    print()
    print(f"  Monte Carlo ({r.get('mc_n_sims', 1000):,} sims)")
    print(f"    P50 DD:  {r['mc_p50_dd']:.1%}")
    print(f"    P95 DD:  {r['mc_p95_dd']:.1%}  ← fund capital against this")
    print(f"    P95/P50: {r['mc_p95_ratio']:.2f}×  (expected: 1.5–3×)")
    print(f"    Prob profit: {r['mc_prob_profit']:.1%}")
    print(f"    Gate:    {'✅ PASS' if r['mc_pass'] else '❌ FAIL'} — {r['mc_reason']}")
    print()
    print(f"  Walk-Forward ({r['wf_n_folds']} folds)")
    print(f"    Consistent:    {r['wf_consistent']:.0%}")
    print(f"    Median Sharpe: {r['wf_median_sharpe']:.2f}")
    print(f"    Gate:          {'✅ PASS' if r['wf_pass'] else '❌ FAIL'} — {r['wf_reason']}")
    print()
    print(f"  OVERALL: {'✅ PASS — proceed to dry-run' if r['status']=='pass' else '❌ FAIL — do not go live'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",  default="BTC/USDT")
    parser.add_argument("--days",    type=int, default=30)
    parser.add_argument("--sims",    type=int, default=1000)
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("--save-signals", action="store_true")
    args = parser.parse_args()

    result = asyncio.run(run_backtest(
        symbol=args.symbol, days=args.days,
        n_sims=args.sims, capital=args.capital,
        save_signals=args.save_signals,
    ))
    _print_report(result)
