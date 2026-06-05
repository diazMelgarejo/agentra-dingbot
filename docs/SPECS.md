# SPECS.md — Agentra DingBot v0.3.x
*Created per TDD workflow: specs before tests, tests before code.*

---

## User Journeys

### Journey 1 — Backtest Validation (Step 6)
```
As a quant developer,
I want to run walk-forward backtests with P95 drawdown gates,
So that I never deploy a strategy that looks good in-sample but fails live.
```

```
As a quant developer,
I want to validate Polymarket edge net of fees and 250ms delay,
So that I know my 8% floor is real profit, not fee illusion.
```

```
As a quant developer,
I want Monte Carlo simulation to report P95 max-drawdown (not just median),
So that I fund my account for the realistic worst case.
```

### Journey 2 — Executor Safety (Step 7)
```
As a trader,
I want the bot to refuse to start if my API keys include withdraw permission,
So that a compromised key can never drain my exchange wallet.
```

```
As a trader,
I want every order to be sanity-checked before submission,
So that rounding bugs, zero-quantity orders, and bad prices never reach the exchange.
```

```
As a trader,
I want a tested CCXT failure matrix (partial fills, rejections, timeouts),
So that every failure mode has a verified recovery path.
```

```
As a trader,
I want a confirmation gate that requires explicit opt-in before live trading,
So that paper mode is the safe default and going live is a deliberate act.
```

---

## Step 6 — Backtest Validation

### 6.1 Walk-Forward Validator (`src/backtesting/walk_forward.py`)
**Behaviour contract:**
- `WalkForwardValidator(train_bars, test_bars, embargo_bars, label_horizon)` creates a validator
- `.splits(n)` returns `[(train_range, test_range), ...]` with purged/embargoed boundaries
- Purging: training labels whose forward window overlaps the test period are removed
- Embargo: `embargo_bars` gap after each train-end before test-start
- `.run(df, signal_fn)` calls `signal_fn(df_train, df_test) -> List[trade_dicts]`
- `.report(folds)` returns `WalkForwardReport` with median Sharpe, worst DD, consistent-folds %
- `.passes_gate()` requires ≥60% folds profitable AND median Sharpe ≥ 0.30

**Edge cases:**
- `n < train_bars + embargo + test_bars` → returns empty splits (no crash)
- `signal_fn` raises → fold gets 0 trades, logged, continues
- Empty trade list → fold scores zeros, counts as failed fold

### 6.2 Monte Carlo Engine (`src/backtesting/monte_carlo.py`)
**Behaviour contract:**
- `run_monte_carlo(trades, n_sims, capital)` returns `MCReport`
- Uses block bootstrap (size=10) to preserve autocorrelation
- Reports P5/P50/P95/P99 for: final equity, max drawdown, Sharpe
- `MCReport.passes_gate(max_p95_dd, min_prob_profit)` is the go/no-go signal
- **P95 max-drawdown is the gate metric** (not median, not backtest single-path)
- Fewer than 5 trades → returns empty report (no crash)

**Edge cases:**
- All trades same sign → valid distribution
- `n_sims=0` → error
- `capital=0` → handled gracefully

### 6.3 Signal Replay (`src/backtesting/signal_replay.py`)
**Behaviour contract:**
- `save_signals(signals: List[SignalRecord], path)` → JSON file
- `load_signals(path)` → `List[SignalRecord]`
- `replay_as_trades(signals, df) -> List[TradeResult]` — simulates entries/exits on OHLCV
- `compute_metrics(trades) -> BacktestMetrics` — Sharpe, win rate, profit factor, max DD
- Slippage model: configurable `slippage_pct` (default 0.1%)
- Fee model: configurable `fee_pct` (default 0.04%)

### 6.4 Polymarket Backtest (`src/backtesting/polymarket_backtest.py`)
**Behaviour contract:**
- `BrierScore.compute(probs, outcomes) -> float` — mean squared error [0,1]
- `BrierScore.passes_gate(score, baseline) -> bool` — score < 0.20 AND beats baseline
- `PolymarketFeeModel(fee_rate_bps)` — dynamic fee from `feeRateBps`
- `fee_model.cost(yes_price, stake_usdc) -> float` — taker fee in USDC
- `fee_model.net_edge(our_prob, market_price, stake) -> float` — edge after fees + delay
- `PolymarketBacktester.run(resolved_markets) -> PMBacktestReport`

### 6.5 Risk Manager — Equity-Based Daily Limit
**Behaviour contract:**
- Daily loss limit computed against **equity** (start-of-day NAV including open P&L)
- Not balance (which ignores open trade losses silently)
- `equity_circuit_breaker(current_equity, start_equity, limit_pct) -> bool`

---

## Step 7 — Executor Safety

### 7.1 Executor Safety Module (`src/agents/executor/safety.py`)
**Behaviour contract:**

#### `check_api_permissions(exchange) -> PermissionResult`
- Calls exchange.fetchPermissions() or equivalent
- Returns `PermissionResult(trade_ok, withdraw_ok, transfer_ok)`
- **HARD FAIL if withdraw_ok is True** — log critical error, return False
- If exchange doesn't support permission check → warn, allow with flag

#### `validate_order(symbol, side, price, qty, exchange) -> ValidationResult`
- `qty` must be ≥ exchange minimum AND properly tick-rounded using `amount_to_precision`
- `price` must be within 5% of last trade (sanity bound)
- `balance` must be sufficient for the order
- Returns `ValidationResult(valid, errors: List[str])`
- **Never silently round** — surface rounding as an explicit validation step

#### `KillSwitch`
- `KillSwitch.arm()` — raises a file-based flag (`data/KILL_SWITCH`)
- `KillSwitch.is_armed() -> bool` — checks for flag file
- `KillSwitch.disarm()` — removes flag
- All executor entry points check `KillSwitch.is_armed()` before proceeding

#### `EquityCircuitBreaker`
- Tracks start-of-day equity, open P&L
- Fires when (start_equity - current_equity) / start_equity ≥ limit_pct
- Returns `should_halt: bool`

### 7.2 CCXT Failure Matrix (`tests/test_executor_dryrun.py`)
Tests for every CCXT failure path:

| Scenario | Expected behaviour |
|---|---|
| `InsufficientFunds` | Order skipped, error logged, no crash |
| `InvalidOrder` (bad qty) | Retry after tick-rounding, then skip |
| `RateLimitExceeded` | Exponential backoff, max 3 retries |
| `RequestTimeout` | Retry once with idempotency key |
| `AuthenticationError` | **HALT** — no retry, raise critical |
| `ExchangeNotAvailable` | Retry 3× with backoff |
| Partial fill (50%) | Order recorded as partial, position updated |
| Zero quantity after rounding | Skip order, emit warning |
| Price 10% outside bounds | Skip order, emit warning |
| Kill switch armed | Skip all orders immediately |
| Withdraw permission detected | HALT on startup |

### 7.3 Live Trading Gate (`src/deploy/live.py`)
**Behaviour contract:**
- Requires `LIVE_TRADING=true` in env AND typed confirmation `"LIVE"` on stdin
- Default is paper mode (`PAPER_MODE=true`)
- Ghost position check at startup: load DB, verify all "open" trades have matching exchange positions
- Absolute paths for all file I/O (no `./` relative paths that break under cron)

---

## Coverage Targets (per TDD SKILL.md)

| Module | Target |
|--------|--------|
| `backtesting/monte_carlo.py` | 90%+ |
| `backtesting/walk_forward.py` | 85%+ |
| `backtesting/signal_replay.py` | 85%+ |
| `backtesting/polymarket_backtest.py` | 85%+ |
| `agents/executor/safety.py` | 95%+ |
| `agents/executor/agent.py` (with safety) | 80%+ |

---

## TDD Cycle Checkpoints

Each module follows RED → GREEN → REFACTOR with a git commit per phase:
- `test: RED - <module>` — failing tests committed first
- `feat: GREEN - <module>` — minimal implementation committed
- `refactor: REFACTOR - <module>` — cleanup committed

The test is the specification. If you cannot write a test, you do not understand the requirement.
