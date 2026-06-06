"""
backtesting/walk_forward.py  —  Step 6: Walk-Forward Validation
===============================================================
Walk-forward analysis with purged/embargoed splits — the community-standard
defense against overfitting for financial time-series.

Why walk-forward beats a single train/test split
-------------------------------------------------
A single split gives one estimate. Walk-forward gives N estimates across
multiple market regimes, revealing whether the strategy degrades on unseen
data or adapts. López de Prado calls single-split financial ML "scientifically
unsound"; walk-forward is the reconcilable-with-paper-trading alternative.

Purging and embargo (from AFML ch. 12)
---------------------------------------
- Purging: remove training samples whose *labels* (forward returns) overlap
  with the test window. Without this, the model sees future data indirectly.
- Embargo: gap a few bars after each test window before the next training
  window begins. Prevents leakage via serial correlation.

Usage
-----
    from backtesting.walk_forward import WalkForwardValidator
    wfv = WalkForwardValidator(train_bars=500, test_bars=100, embargo_bars=10)
    results = wfv.run(df, signal_fn)   # signal_fn(df_train, df_test) -> trades
    report = wfv.report(results)
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class WalkForwardFold:
    fold_idx: int
    train_start: Any          # timestamp
    train_end: Any
    test_start: Any
    test_end: Any
    n_trades: int = 0
    sharpe: float = 0.0
    max_dd: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_pnl_pct: float = 0.0
    trades: list[dict] = field(default_factory=list)


@dataclass
class WalkForwardReport:
    folds: list[WalkForwardFold]
    n_folds: int
    data_months: float

    @property
    def median_sharpe(self) -> float:
        vals = [f.sharpe for f in self.folds if f.n_trades > 0]
        return float(np.median(vals)) if vals else 0.0

    @property
    def worst_fold_dd(self) -> float:
        return max((f.max_dd for f in self.folds), default=0.0)

    @property
    def consistent_folds_pct(self) -> float:
        """Fraction of folds with positive total P&L."""
        if not self.folds:
            return 0.0
        return sum(1 for f in self.folds if f.total_pnl_pct > 0) / len(self.folds)

    def summary(self) -> str:
        lines = [
            f"Walk-Forward Report ({self.n_folds} folds, {self.data_months:.1f} months)",
            f"  Median Sharpe       : {self.median_sharpe:.3f}",
            f"  Worst fold max-DD   : {self.worst_fold_dd:.1%}",
            f"  Consistent folds    : {self.consistent_folds_pct:.0%} (positive P&L)",
            "",
            f"  {'Fold':<5} {'Trades':<8} {'Sharpe':<8} {'MaxDD':<8} {'WinRate':<9} {'PnL%'}",
            "  " + "-" * 55,
        ]
        for f in self.folds:
            lines.append(
                f"  {f.fold_idx:<5} {f.n_trades:<8} {f.sharpe:<8.2f} "
                f"{f.max_dd:<8.1%} {f.win_rate:<9.1%} {f.total_pnl_pct:.2%}"
            )
        return "\n".join(lines)

    def passes_gate(self,
                    min_consistent_folds: float = 0.60,
                    min_median_sharpe: float = 0.30) -> tuple[bool, str]:
        if self.consistent_folds_pct < min_consistent_folds:
            return False, (
                f"Only {self.consistent_folds_pct:.0%} of folds profitable "
                f"(need ≥{min_consistent_folds:.0%})"
            )
        if self.median_sharpe < min_median_sharpe:
            return False, (
                f"Median Sharpe {self.median_sharpe:.2f} < {min_median_sharpe:.2f}"
            )
        return True, (
            f"Gate passed: {self.consistent_folds_pct:.0%} consistent, "
            f"Sharpe={self.median_sharpe:.2f}"
        )


class WalkForwardValidator:
    """
    Walk-forward validator with purged/embargoed splits.

    Parameters
    ----------
    train_bars   : number of bars in each training window
    test_bars    : number of bars in each test window (out-of-sample)
    embargo_bars : bars to skip between train end and test start (leakage guard)
    label_horizon: forward-look of labels in bars (for purging)
    """

    def __init__(
        self,
        train_bars: int = 500,
        test_bars: int = 100,
        embargo_bars: int = 10,
        label_horizon: int = 3,
    ):
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.embargo_bars = embargo_bars
        self.label_horizon = label_horizon

    def splits(self, n: int) -> list[tuple[range, range]]:
        """
        Generate purged/embargoed (train_idx, test_idx) pairs.
        Each test window advances by `test_bars`
        training expands to fill
        all available history before the embargo gap.
        """
        result = []
        start = self.train_bars
        while start + self.embargo_bars + self.test_bars <= n:
            # Purge: training labels whose forward horizon overlaps test start
            purge_end = start - self.label_horizon
            train_idx = range(0, max(0, purge_end))
            # Embargo gap
            test_start = start + self.embargo_bars
            test_end   = min(test_start + self.test_bars, n)
            test_idx   = range(test_start, test_end)
            if len(train_idx) > 0 and len(test_idx) > 0:
                result.append((train_idx, test_idx))
            start += self.test_bars
        return result

    def run(
        self,
        df: pd.DataFrame,
        signal_fn: Callable[[pd.DataFrame, pd.DataFrame], list[dict]],
    ) -> list[WalkForwardFold]:
        """
        Run the signal function on each (train, test) split.

        signal_fn signature:
            (df_train, df_test) -> List[{"pnl_pct": float, "signal": str, ...}]

        The function trains on df_train, generates signals on df_test, and
        returns a list of trade dicts with at minimum a "pnl_pct" key.
        """
        folds = []
        splits = self.splits(len(df))

        if not splits:
            logger.warning("wf_no_splits",
                           n=len(df), train=self.train_bars, test=self.test_bars)
            return folds

        months = len(df) / (24 * 30)  # rough, assumes hourly bars
        logger.info("wf_start", n_splits=len(splits), data_months=f"{months:.1f}")

        for i, (tr_idx, te_idx) in enumerate(splits):
            df_train = df.iloc[list(tr_idx)]
            df_test  = df.iloc[list(te_idx)]
            try:
                trades = signal_fn(df_train, df_test)
            except Exception as exc:
                logger.warning("wf_fold_failed", fold=i, error=str(exc))
                trades = []

            fold = _score_fold(i, df_train, df_test, trades)
            folds.append(fold)
            logger.debug("wf_fold_done", fold=i, trades=fold.n_trades,
                         sharpe=f"{fold.sharpe:.2f}", dd=f"{fold.max_dd:.1%}")

        return folds

    def report(self, folds: list[WalkForwardFold]) -> WalkForwardReport:
        if not folds:
            return WalkForwardReport(folds=[], n_folds=0, data_months=0.0)
        # Use test_bars count × folds (timestamps are not integers)
        total_bars = len(folds) * self.test_bars
        return WalkForwardReport(
            folds=folds,
            n_folds=len(folds),
            data_months=total_bars / (24 * 30),
        )


# ── Fold scoring helper ──────────────────────────────────────────────────────

def _score_fold(idx: int, df_train: pd.DataFrame, df_test: pd.DataFrame,
                trades: list[dict]) -> WalkForwardFold:
    import math
    pnls = [t.get("pnl_pct", 0.0) for t in trades]
    n = len(pnls)
    fold = WalkForwardFold(
        fold_idx=idx,
        train_start=df_train.index[0] if len(df_train) else None,
        train_end=df_train.index[-1] if len(df_train) else None,
        test_start=df_test.index[0] if len(df_test) else None,
        test_end=df_test.index[-1] if len(df_test) else None,
        n_trades=n,
        trades=trades,
    )
    if n == 0:
        return fold

    arr = np.array(pnls)
    # Equity curve
    equity = np.concatenate([[1.0], np.cumprod(1 + arr)])
    peak   = np.maximum.accumulate(equity)
    fold.max_dd = float(np.max((peak - equity) / np.where(peak == 0, 1, peak)))
    fold.total_pnl_pct = float(equity[-1] - 1.0)
    # Sharpe
    mu, sig = np.mean(arr), np.std(arr, ddof=1)
    fold.sharpe = float((mu / sig * math.sqrt(252)) if sig > 0 else 0.0)
    # Win rate
    fold.win_rate = float(np.mean(arr > 0)) if n > 0 else 0.0
    # Profit factor
    gross_win  = float(np.sum(arr[arr > 0])) if np.any(arr > 0) else 0.0
    gross_loss = float(abs(np.sum(arr[arr < 0]))) if np.any(arr < 0) else 0.0
    fold.profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    return fold
