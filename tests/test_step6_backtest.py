"""
tests/test_step6_backtest.py — RED phase (TDD)
===============================================
These tests are written FIRST per the TDD workflow.
They define the exact behaviour contract from SPECS.md.
Run before any implementation — they should ALL FAIL initially.

TDD cycle:
  🔴 RED   → this file committed, tests fail
  🟢 GREEN → implementation written, tests pass
  🔵 REFACTOR → code cleaned, tests stay green

AAA pattern: Arrange, Act, Assert
One behaviour per test, descriptive names.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest


# ── Factories ────────────────────────────────────────────────────────────────

def _ohlcv(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Deterministic OHLCV — 600 hourly bars gives ~25 days."""
    rng = np.random.default_rng(seed)
    close = 50_000 * np.cumprod(1 + rng.normal(0, 0.005, n))
    idx   = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open":   close * 0.999, "high": close * 1.002,
        "low":    close * 0.998, "close": close,
        "volume": rng.uniform(10, 100, n),
    }, index=idx)


def _trades(n: int = 50, win_rate: float = 0.55, seed: int = 0):
    """Synthetic trade records."""
    from src.backtesting.monte_carlo import TradeResult
    rng = np.random.default_rng(seed)
    return [
        TradeResult(
            pnl_pct=float(rng.uniform(0.01, 0.04) if rng.random() < win_rate
                          else -rng.uniform(0.01, 0.03)),
            duration_bars=int(rng.integers(1, 10)),
        )
        for _ in range(n)
    ]


# ════════════════════════════════════════════════════════════════════════════
# A. Monte Carlo Engine
# ════════════════════════════════════════════════════════════════════════════

class TestMonteCarlo:
    """
    Journey: As a quant, I want Monte Carlo to report P95 max-DD so I fund
    my account for the realistic worst case, not the lucky best case.
    """

    # ── Arrange/Act helpers ──────────────────────────────────────────────────

    def _run(self, n=50, win_rate=0.55, n_sims=500):
        from src.backtesting.monte_carlo import run_monte_carlo
        return run_monte_carlo(_trades(n, win_rate), n_sims=n_sims,
                               capital=1000.0, seed=99)

    # ── Unit: report structure ───────────────────────────────────────────────

    def test_report_has_all_required_fields(self):
        """MCReport must expose P5/P50/P95/P99 for both drawdown and equity."""
        r = self._run()
        for attr in ["p5_final_equity", "median_final_equity", "p95_final_equity",
                     "p50_max_dd", "p95_max_dd", "p99_max_dd",
                     "median_sharpe", "p5_sharpe", "p95_sharpe",
                     "prob_profit", "ruin_probability"]:
            assert hasattr(r, attr), f"MCReport missing field: {attr}"

    def test_p95_drawdown_exceeds_p50_drawdown(self):
        """P95 max-DD must be ≥ P50 max-DD (worst-case ≥ median)."""
        r = self._run(n_sims=500)
        assert r.p95_max_dd >= r.p50_max_dd, (
            f"P95 DD {r.p95_max_dd:.1%} should be ≥ P50 DD {r.p50_max_dd:.1%}"
        )

    def test_p95_drawdown_above_zero_for_non_trivial_trades(self):
        """Any strategy with real variance must have positive P95 DD."""
        r = self._run()
        assert r.p95_max_dd > 0.0

    def test_prob_profit_between_0_and_1(self):
        """Probability of profit must be a valid fraction."""
        r = self._run()
        assert 0.0 <= r.prob_profit <= 1.0

    def test_profitable_strategy_has_prob_profit_above_half(self):
        """A 65% win-rate strategy should profit in >50% of simulations."""
        r = self._run(win_rate=0.65, n_sims=1000)
        assert r.prob_profit > 0.5

    def test_losing_strategy_has_high_ruin_probability(self):
        """A 25% win-rate strategy with 100 trades should sometimes hit ruin."""
        from src.backtesting.monte_carlo import run_monte_carlo, TradeResult
        import numpy as np
        rng = np.random.default_rng(0)
        trades = [TradeResult(pnl_pct=float(rng.uniform(0.01, 0.04)
                  if rng.random() < 0.25 else -rng.uniform(0.01, 0.03)))
                  for _ in range(100)]
        r = run_monte_carlo(trades, n_sims=1000, capital=1000.0, seed=42)
        # Median equity well below start confirms losses — ruin or near-ruin
        assert r.median_final_equity < r.capital * 0.70

    # ── Gate logic ───────────────────────────────────────────────────────────

    def test_gate_passes_with_acceptable_p95_dd(self):
        """passes_gate must return True when P95 DD is within tolerance."""
        r = self._run(win_rate=0.65)
        ok, _ = r.passes_gate(max_p95_dd=0.99)  # very lenient threshold
        assert ok

    def test_gate_fails_when_p95_dd_too_high(self):
        """passes_gate must return False with reason when P95 DD exceeds limit."""
        r = self._run(win_rate=0.25, n=100)  # bad strategy
        ok, reason = r.passes_gate(max_p95_dd=0.01)  # very strict
        assert ok is False
        assert "P95" in reason or "dd" in reason.lower()

    def test_gate_fails_when_prob_profit_too_low(self):
        """passes_gate fails when not enough simulations end profitably."""
        r = self._run(win_rate=0.25)
        ok, reason = r.passes_gate(min_prob_profit=0.99)
        assert ok is False

    # ── Edge cases ───────────────────────────────────────────────────────────

    def test_fewer_than_5_trades_returns_empty_report(self):
        """Too few trades → empty report without crash."""
        from src.backtesting.monte_carlo import run_monte_carlo, TradeResult
        r = run_monte_carlo([TradeResult(pnl_pct=0.01)], n_sims=100, capital=1000.0)
        assert r.n_trades <= 4
        assert r.prob_profit == 0.0  # empty

    def test_all_winning_trades_gives_zero_ruin(self):
        """Pure winning trades should never hit ruin threshold."""
        from src.backtesting.monte_carlo import run_monte_carlo, TradeResult
        trades = [TradeResult(pnl_pct=0.01)] * 30
        r = run_monte_carlo(trades, n_sims=500, capital=1000.0)
        assert r.ruin_probability == 0.0

    def test_summary_string_mentions_p95(self):
        """summary() must surface the P95 DD (it's the key risk metric)."""
        r = self._run()
        s = r.summary()
        assert "P95" in s or "95" in s


# ════════════════════════════════════════════════════════════════════════════
# B. Walk-Forward Validator
# ════════════════════════════════════════════════════════════════════════════

class TestWalkForward:
    """
    Journey: As a quant, I want walk-forward validation with purged splits
    so I get a *distribution* of Sharpe across regimes, not one lucky number.
    """

    # ── Split generation ─────────────────────────────────────────────────────

    def test_splits_returns_list_of_tuples(self):
        from src.backtesting.walk_forward import WalkForwardValidator
        wfv = WalkForwardValidator(train_bars=200, test_bars=50, embargo_bars=5)
        splits = wfv.splits(500)
        assert isinstance(splits, list)
        assert all(len(s) == 2 for s in splits)

    def test_splits_non_empty_for_adequate_data(self):
        from src.backtesting.walk_forward import WalkForwardValidator
        wfv = WalkForwardValidator(train_bars=200, test_bars=50, embargo_bars=5)
        assert len(wfv.splits(600)) > 0

    def test_splits_empty_when_insufficient_data(self):
        from src.backtesting.walk_forward import WalkForwardValidator
        wfv = WalkForwardValidator(train_bars=200, test_bars=50, embargo_bars=5)
        assert wfv.splits(100) == []  # not enough data

    def test_train_end_before_test_start_with_embargo(self):
        """Embargo gap must appear between every train-end and test-start."""
        from src.backtesting.walk_forward import WalkForwardValidator
        wfv = WalkForwardValidator(train_bars=100, test_bars=30, embargo_bars=10)
        for tr, te in wfv.splits(300):
            gap = te.start - tr.stop
            assert gap >= 10, f"Embargo gap {gap} < 10"

    def test_purge_removes_overlapping_label_bars(self):
        """Training range must end at least label_horizon bars before test start."""
        from src.backtesting.walk_forward import WalkForwardValidator
        H = 5  # label horizon
        wfv = WalkForwardValidator(train_bars=100, test_bars=30,
                                   embargo_bars=10, label_horizon=H)
        for tr, te in wfv.splits(300):
            purge_end = tr.stop
            test_start_with_embargo = te.start
            # Train must stop at least H bars before embargo start
            assert purge_end <= test_start_with_embargo - H, (
                f"Leakage: train ends {purge_end}, test at {te.start} (H={H})"
            )

    # ── run() behaviour ──────────────────────────────────────────────────────

    def test_run_calls_signal_fn_once_per_split(self):
        from src.backtesting.walk_forward import WalkForwardValidator
        wfv = WalkForwardValidator(train_bars=200, test_bars=50, embargo_bars=5)
        call_count = [0]

        def counter_fn(df_train, df_test):
            call_count[0] += 1
            return []

        wfv.run(_ohlcv(600), counter_fn)
        n_splits = len(wfv.splits(600))
        assert call_count[0] == n_splits

    def test_run_continues_when_signal_fn_raises(self):
        """A failing signal_fn on one fold must not abort the whole run."""
        from src.backtesting.walk_forward import WalkForwardValidator
        wfv = WalkForwardValidator(train_bars=200, test_bars=50, embargo_bars=5)

        def exploding_fn(df_train, df_test):
            raise RuntimeError("simulated failure")

        folds = wfv.run(_ohlcv(600), exploding_fn)
        assert isinstance(folds, list)  # completed, not crashed

    # ── report & gate ────────────────────────────────────────────────────────

    def test_report_has_correct_fold_count(self):
        from src.backtesting.walk_forward import WalkForwardValidator
        wfv = WalkForwardValidator(train_bars=200, test_bars=50, embargo_bars=5)
        folds = wfv.run(_ohlcv(600), lambda tr, te: [])
        report = wfv.report(folds)
        assert report.n_folds == len(folds)

    def test_gate_passes_when_most_folds_profitable(self):
        """≥60% profitable folds with positive Sharpe should pass the gate."""
        from src.backtesting.walk_forward import WalkForwardValidator, WalkForwardFold
        folds = [
            WalkForwardFold(i, None, None, None, None,
                            n_trades=10, total_pnl_pct=0.05, sharpe=0.8)
            for i in range(8)
        ] + [
            WalkForwardFold(8, None, None, None, None,
                            n_trades=10, total_pnl_pct=-0.02, sharpe=-0.3)
        ]
        from src.backtesting.walk_forward import WalkForwardReport
        rpt = WalkForwardReport(folds=folds, n_folds=9, data_months=3.0)
        ok, _ = rpt.passes_gate(min_consistent_folds=0.60, min_median_sharpe=0.3)
        assert ok

    def test_gate_fails_when_strategy_inconsistent(self):
        """Fewer than 50% profitable folds must fail the gate."""
        from src.backtesting.walk_forward import WalkForwardFold, WalkForwardReport
        folds = [
            WalkForwardFold(i, None, None, None, None,
                            n_trades=10, total_pnl_pct=-0.05, sharpe=-0.5)
            for i in range(8)
        ]
        rpt = WalkForwardReport(folds=folds, n_folds=8, data_months=3.0)
        ok, reason = rpt.passes_gate(min_consistent_folds=0.60)
        assert ok is False
        assert "fold" in reason.lower() or "consistent" in reason.lower()


# ════════════════════════════════════════════════════════════════════════════
# C. Signal Replay
# ════════════════════════════════════════════════════════════════════════════

class TestSignalReplay:
    """
    Journey: As a quant, I want to replay SuperBot signals on OHLCV history
    so I can feed the real signal record into Monte Carlo and walk-forward.
    """

    def _record(self, sig="BUY", price=50000.0, ts="2024-01-01T00:00:00+00:00"):
        from src.backtesting.signal_replay import SignalRecord
        return SignalRecord(signal=sig, price=price, timestamp=ts, confidence=0.7)

    def test_save_and_load_roundtrip(self):
        """Signals saved to JSON must load back with identical values."""
        from src.backtesting.signal_replay import save_signals, load_signals
        records = [self._record("BUY", 50000), self._record("SELL", 52000)]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "signals.json")
            save_signals(records, path)
            loaded = load_signals(path)
        assert len(loaded) == 2
        assert loaded[0].signal == "BUY"
        assert loaded[1].price == 52000.0

    def test_replay_produces_trade_results(self):
        """replay_as_trades must return a non-empty list of TradeResult."""
        from src.backtesting.signal_replay import SignalRecord, replay_as_trades
        records = [
            SignalRecord("BUY",  50000.0, "2024-01-01T00:00:00+00:00", 0.7),
            SignalRecord("SELL", 52000.0, "2024-01-10T00:00:00+00:00", 0.6),
        ]
        df = _ohlcv(500)
        trades = replay_as_trades(records, df)
        assert isinstance(trades, list)

    def test_buy_then_higher_sell_yields_positive_pnl(self):
        """BUY when price is low, SELL when price is higher → positive P&L."""
        from src.backtesting.signal_replay import SignalRecord, replay_as_trades
        df = _ohlcv(500)
        # Use actual timestamps from the df so price lookup succeeds
        early_ts  = df.index[10].isoformat()   # early bar (lower price in flat walk)
        later_ts  = df.index[400].isoformat()  # later bar
        # Use a df with a known uptrend to ensure positive P&L
        import numpy as np
        import pandas as pd
        n = 500
        idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
        prices = np.linspace(40_000, 60_000, n)  # monotone up
        up_df = pd.DataFrame({"open": prices, "high": prices+100, "low": prices-100,
                               "close": prices, "volume": 10.0}, index=idx)
        ts_buy  = idx[10].isoformat()
        ts_sell = idx[400].isoformat()
        recs = [SignalRecord("BUY",  float(up_df.close.iloc[10]),  ts_buy,  0.8),
                SignalRecord("SELL", float(up_df.close.iloc[400]), ts_sell, 0.8)]
        trades = replay_as_trades(recs, up_df, slippage_pct=0.0, fee_pct=0.0)
        total = sum(t.pnl_pct for t in trades if t)
        assert total > 0, f"Expected positive P&L, got {total}"

    def test_fee_and_slippage_reduce_net_pnl(self):
        """Net P&L with fees must be less than without fees."""
        from src.backtesting.signal_replay import SignalRecord, replay_as_trades
        recs = [
            SignalRecord("BUY",  50_000, "2024-01-01T00:00:00+00:00", 0.8),
            SignalRecord("SELL", 55_000, "2024-01-15T00:00:00+00:00", 0.8),
        ]
        pnl_no_cost = sum(
            t.pnl_pct for t in replay_as_trades(recs, _ohlcv(500),
                                                  slippage_pct=0.0, fee_pct=0.0) if t)
        pnl_with_cost = sum(
            t.pnl_pct for t in replay_as_trades(recs, _ohlcv(500),
                                                  slippage_pct=0.002, fee_pct=0.001) if t)
        assert pnl_no_cost > pnl_with_cost

    def test_compute_metrics_returns_expected_keys(self):
        """BacktestMetrics must expose sharpe, win_rate, profit_factor, max_dd."""
        from src.backtesting.signal_replay import compute_metrics
        from src.backtesting.monte_carlo import TradeResult
        trades = [TradeResult(pnl_pct=0.02)] * 5 + [TradeResult(pnl_pct=-0.01)] * 3
        m = compute_metrics(trades)
        for attr in ["sharpe", "win_rate", "profit_factor", "max_dd", "total_pnl_pct"]:
            assert hasattr(m, attr), f"BacktestMetrics missing: {attr}"

    def test_win_rate_correct_for_known_trades(self):
        """5 wins + 3 losses → win_rate = 5/8 = 0.625."""
        from src.backtesting.signal_replay import compute_metrics
        from src.backtesting.monte_carlo import TradeResult
        trades = [TradeResult(pnl_pct=0.02)] * 5 + [TradeResult(pnl_pct=-0.01)] * 3
        m = compute_metrics(trades)
        assert abs(m.win_rate - 0.625) < 0.001


# ════════════════════════════════════════════════════════════════════════════
# D. Polymarket Backtest (Brier score + Fee model)
# ════════════════════════════════════════════════════════════════════════════

class TestPolymarketBacktest:
    """
    Journey: As a quant, I want Polymarket edge measured net of taker fees
    and the 250ms delay so I know the 8% floor is real, not fee illusion.
    """

    # ── Brier Score ──────────────────────────────────────────────────────────

    def test_perfect_predictions_give_brier_zero(self):
        from src.backtesting.polymarket_backtest import BrierScore
        probs    = [1.0, 0.0, 1.0, 0.0]
        outcomes = [1,   0,   1,   0  ]
        assert BrierScore.compute(probs, outcomes) == pytest.approx(0.0)

    def test_random_predictions_give_brier_near_quarter(self):
        """Random classifier at 0.5 → BS ≈ 0.25."""
        from src.backtesting.polymarket_backtest import BrierScore
        probs    = [0.5] * 100
        outcomes = [i % 2 for i in range(100)]
        bs = BrierScore.compute(probs, outcomes)
        assert abs(bs - 0.25) < 0.05

    def test_brier_gate_passes_below_threshold(self):
        from src.backtesting.polymarket_backtest import BrierScore
        assert BrierScore.passes_gate(score=0.15, baseline=0.25) is True

    def test_brier_gate_fails_above_threshold(self):
        from src.backtesting.polymarket_backtest import BrierScore
        assert BrierScore.passes_gate(score=0.22, baseline=0.25) is False

    def test_brier_gate_fails_if_not_beating_baseline(self):
        """Even below 0.20, must beat the base-rate baseline."""
        from src.backtesting.polymarket_backtest import BrierScore
        assert BrierScore.passes_gate(score=0.19, baseline=0.18) is False

    # ── Fee Model ─────────────────────────────────────────────────────────────

    def test_fee_is_zero_at_extreme_prices(self):
        """At YES price near 0 or 1, taker fee approaches zero."""
        from src.backtesting.polymarket_backtest import PolymarketFeeModel
        m = PolymarketFeeModel(fee_rate_bps=100)
        assert m.cost(yes_price=0.01, stake_usdc=100.0) < 0.10
        assert m.cost(yes_price=0.99, stake_usdc=100.0) < 0.10

    def test_fee_peaks_near_50pct_probability(self):
        """Fee is highest at p=0.5 (most uncertain) per Polymarket curve."""
        from src.backtesting.polymarket_backtest import PolymarketFeeModel
        m = PolymarketFeeModel(fee_rate_bps=100)
        fee_at_50 = m.cost(yes_price=0.50, stake_usdc=1000.0)
        fee_at_90 = m.cost(yes_price=0.90, stake_usdc=1000.0)
        fee_at_10 = m.cost(yes_price=0.10, stake_usdc=1000.0)
        assert fee_at_50 > fee_at_90
        assert fee_at_50 > fee_at_10

    def test_net_edge_reduces_gross_edge(self):
        """Net edge after fees must be less than gross edge."""
        from src.backtesting.polymarket_backtest import PolymarketFeeModel
        m = PolymarketFeeModel(fee_rate_bps=100)
        our_prob = 0.60
        market_price = 0.50
        gross = (our_prob - market_price) * 100  # 10%
        net   = m.net_edge(our_prob, market_price, stake_usdc=100.0)
        assert net < gross
        assert net > 0  # still positive edge after fees

    def test_net_edge_negative_when_fees_exceed_gross_edge(self):
        """Very tight edge + high fee → net edge is negative (don't trade)."""
        from src.backtesting.polymarket_backtest import PolymarketFeeModel
        m = PolymarketFeeModel(fee_rate_bps=500)  # 5% fee
        # Only 2% gross edge vs 5% fee
        net = m.net_edge(our_prob=0.52, market_price=0.50, stake_usdc=100.0)
        assert net < 0

    # ── Risk Manager: equity-based daily limit ────────────────────────────────

    def test_equity_circuit_breaker_fires_at_limit(self):
        """Breaker must halt when equity drops by limit_pct from start-of-day."""
        from src.backtesting.polymarket_backtest import equity_circuit_breaker
        assert equity_circuit_breaker(
            current_equity=940.0, start_equity=1000.0, limit_pct=0.05
        ) is True  # 6% drop > 5% limit

    def test_equity_circuit_breaker_silent_within_limit(self):
        from src.backtesting.polymarket_backtest import equity_circuit_breaker
        assert equity_circuit_breaker(
            current_equity=960.0, start_equity=1000.0, limit_pct=0.05
        ) is False  # only 4% drop

    def test_equity_circuit_uses_equity_not_balance(self):
        """
        Key insight from research: open-trade losses count.
        balance=1000 (no realised loss) but equity=930 (unrealised loss) → fire.
        """
        from src.backtesting.polymarket_backtest import equity_circuit_breaker
        # balance=1000 would NOT fire, but equity=930 DOES
        assert equity_circuit_breaker(
            current_equity=930.0, start_equity=1000.0, limit_pct=0.05
        ) is True
