"""
backtesting/monte_carlo.py  —  Step 6: Monte Carlo Validation Engine
======================================================================
Corrected after research: reports P50 AND P95 drawdown distributions.
Size capital against P95 (research shows it runs 1.5-3x the backtest figure).
5,000 simulations minimum
block bootstrap preserves autocorrelation.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TradeResult:
    pnl_pct: float
    duration_bars: int = 1
    signal: str = ""


@dataclass
class MCReport:
    n_sims: int
    n_trades: int
    capital: float
    median_final_equity: float
    p5_final_equity: float
    p95_final_equity: float
    prob_profit: float
    median_max_dd: float
    p50_max_dd: float
    p95_max_dd: float    # ← SIZE CAPITAL AGAINST THIS
    p99_max_dd: float
    median_sharpe: float
    p5_sharpe: float
    p95_sharpe: float
    ruin_probability: float

    def summary(self) -> str:
        return "\n".join([
            f"Monte Carlo ({self.n_sims:,} sims, {self.n_trades} trades, ${self.capital:,.0f})",
            f"  Prob profit  : {self.prob_profit:.1%}",
            f"  Risk of ruin : {self.ruin_probability:.1%}",
            f"  Final equity : P5=${self.p5_final_equity:,.0f}  P50=${self.median_final_equity:,.0f}  P95=${self.p95_final_equity:,.0f}",
            f"  Max drawdown : P50={self.p50_max_dd:.1%}  P95={self.p95_max_dd:.1%}  P99={self.p99_max_dd:.1%}",
            f"  Sharpe       : P5={self.p5_sharpe:.2f}  median={self.median_sharpe:.2f}  P95={self.p95_sharpe:.2f}",
            f"  ⚠  Fund for P95 drawdown ({self.p95_max_dd:.1%}), not P50 ({self.p50_max_dd:.1%})",
        ])

    def passes_gate(self,
                    max_p95_dd: float = 0.30,
                    min_prob_profit: float = 0.55) -> tuple[bool, str]:
        if self.p95_max_dd > max_p95_dd:
            return False, f"P95 DD {self.p95_max_dd:.1%} > tolerance {max_p95_dd:.1%}"
        if self.prob_profit < min_prob_profit:
            return False, f"Prob profit {self.prob_profit:.1%} < {min_prob_profit:.1%}"
        return True, f"Gate passed (P95 DD={self.p95_max_dd:.1%}, prob_profit={self.prob_profit:.1%})"


def run_monte_carlo(
    trades: Sequence[TradeResult],
    n_sims: int = 5000,
    capital: float = 1000.0,
    block_size: int = 10,
    seed: int | None = None,
    ruin_threshold: float = 0.20,
) -> MCReport:
    """Block-bootstrap Monte Carlo on completed trade records."""
    if len(trades) < 5:
        logger.warning("mc_insufficient_trades", n=len(trades))
        return _empty(n_sims, len(trades), capital)

    rng = np.random.default_rng(seed)
    pnls = np.array([t.pnl_pct for t in trades])
    n = len(pnls)

    blocks = [pnls[i:i+block_size] for i in range(0, n - block_size + 1)] or [pnls]

    finals = np.empty(n_sims)
    dds    = np.empty(n_sims)
    sharps = np.empty(n_sims)

    for i in range(n_sims):
        sim: list = []
        while len(sim) < n:
            sim.extend(blocks[rng.integers(len(blocks))].tolist())
        arr = np.array(sim[:n])
        eq = np.concatenate([[capital], capital * np.cumprod(1 + arr)])
        finals[i] = eq[-1]
        peak = np.maximum.accumulate(eq)
        dds[i] = float(np.max((peak - eq) / np.where(peak == 0, 1, peak)))
        mu, sig = np.mean(arr), np.std(arr, ddof=1)
        sharps[i] = (mu / sig * math.sqrt(252)) if sig > 0 else 0.0

    return MCReport(
        n_sims=n_sims, n_trades=n, capital=capital,
        median_final_equity=float(np.median(finals)),
        p5_final_equity=float(np.percentile(finals, 5)),
        p95_final_equity=float(np.percentile(finals, 95)),
        prob_profit=float(np.mean(finals > capital)),
        median_max_dd=float(np.median(dds)),
        p50_max_dd=float(np.percentile(dds, 50)),
        p95_max_dd=float(np.percentile(dds, 95)),
        p99_max_dd=float(np.percentile(dds, 99)),
        median_sharpe=float(np.median(sharps)),
        p5_sharpe=float(np.percentile(sharps, 5)),
        p95_sharpe=float(np.percentile(sharps, 95)),
        ruin_probability=float(np.mean(finals < capital * ruin_threshold)),
    )


def _empty(n_sims, n_trades, capital) -> MCReport:
    return MCReport(
        n_sims=n_sims,
        n_trades=n_trades,
        capital=capital,
        median_final_equity=capital,
        p5_final_equity=capital,
        p95_final_equity=capital,
        prob_profit=0.0,
        median_max_dd=0.0,
        p50_max_dd=0.0,
        p95_max_dd=0.0,
        p99_max_dd=0.0,
        median_sharpe=0.0,
        p5_sharpe=0.0,
        p95_sharpe=0.0,
        ruin_probability=0.0,
    )


def main():
    import argparse
    import json
    import sys
    p = argparse.ArgumentParser()
    p.add_argument("--trades", required=True)
    p.add_argument("--sims", type=int, default=5000)
    p.add_argument("--capital", type=float, default=1000.0)
    p.add_argument("--p95-dd-gate", type=float, default=0.30)
    args = p.parse_args()
    with open(args.trades) as _f:
        raw = json.load(_f)
    trades = [TradeResult(**t) for t in raw]
    r = run_monte_carlo(trades, args.sims, args.capital)
    print(r.summary())
    ok, reason = r.passes_gate(max_p95_dd=args.p95_dd_gate)
    print(f"\n{'✅ PASS' if ok else '❌ FAIL'} — {reason}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
