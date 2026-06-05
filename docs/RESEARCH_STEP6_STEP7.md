# Research Findings: Step 6 & Step 7 Best Practices
*Sources: FreqTrade docs, Polymarket changelog, MacLean-Ziemba-Blazenko (1992),
López de Prado AFML, florinelchis "15 Failure Patterns" (Apr 2026), CCXT wiki,
Bitsgap/BlockBeats/poly-sim field reports, BloFin API safety guide.*

---

## What Changed Our Plan

### 1 — Backtests are necessary but weak predictors
Wiecki et al. (2016) on 888 strategies: Sharpe ratio offers near-zero
predictive value for out-of-sample performance (R² < 0.025). This validated
our multi-stage approach: backtest → paper → tiny-live before scaling.

### 2 — FreqTrade ships two mandatory bias detectors we must run
- `freqtrade lookahead-analysis` — chains backtests, perturbs entries/exits,
  flags future-data leakage.
- `freqtrade recursive-analysis --startup-candle 199 299 399 499` — varies
  warm-up length, detects indicators whose values shift as history grows.
The docs say: *"You should always use both commands."*
**Neither is infallible** (GitHub #11346 shows a strategy that passed both yet
leaked future data via a date filter). They are a floor, not proof.

### 3 — P95 drawdown, not single-path max-DD, is what you size against
Monte Carlo trade-reshuffling consistently shows the 95th-percentile max-DD
running 1.5–3× the backtest figure. Institutional sizing targets P95.
Our existing `monte_carlo.py` only reported the single-path backtest max-DD —
**this was wrong and has been corrected**.

### 4 — Polymarket fee regime changed in 2026
- Jan 7 2026: taker fees live on 15-min crypto markets.
- 5-min launch: taker fees enabled from day one, peaking at **1.56% at 50%
  probability** (fee curve is symmetric around p=0.5).
- Mar 6 2026: fees/rebates extended to all crypto markets.
- Makers pay **zero fees + earn 20% rebate share daily**.
- A **250ms taker delay** is in force (replaced the old 500ms delay ~Feb 18 2026).
- **Impact on our bot:** the 8% edge floor must be measured net of fees + delay.
  The surviving edge is maker/liquidity-provision and well-calibrated late-window
  directional bets. Hardcoding fee rates is wrong — fetch `feeRateBps` live.

### 5 — Production blow-ups are operational, not strategic
Field reports (florinelchis, Apr 2026) documented:
- Fee rates **6–12× higher** than documentation (verify from real fills).
- Rounding that **zeroed order quantities** (68 consecutive ETH rejects).
- **Ghost positions**: DB rows marked open but never closed, blocking trading
  for months (246 ghost rows).
- Relative file paths that break under cron/systemd.
- Silent inactivity — bot "running" but never trading for 28 days.
None were caught by naive unit tests.

### 6 — API key hygiene is non-negotiable
Dec 2024–Jan 2025: >$65M drained via stolen exchange API keys (Coinpaprika).
Rule: **trade-only, no-withdraw, IP-whitelisted keys. Always.**

---

## Revised Thresholds (from research)

| Metric | Before | After research |
|--------|--------|----------------|
| Monte Carlo sims | 1,000 | ≥5,000 |
| Max-DD gate | backtest single-path | Monte Carlo **P95** |
| Polymarket edge floor | 8% gross | 8% **net** (fees + delay) |
| Fee model | hardcoded 0.04% | **dynamic** `feeRateBps` |
| Daily loss limit | balance-based | **equity-based** (open trades count) |
| Brier score gate | none | **<0.20** over ≥50 resolved trades |
| Walk-forward data | 30 days | **≥6 months, multiple regimes** |

---

## Implementation Decisions (drives Step 6 & 7 code)

### Step 6 — What to build
1. `backtesting/walk_forward.py` — purged-embargo walk-forward with P95 gate.
2. `backtesting/monte_carlo.py` — upgraded: 5000 reshuffles, block bootstrap,
   P50/P95/P99 drawdown and Sharpe distributions.
3. `backtesting/signal_replay.py` — save SuperBot signals to JSON; replay
   through FreqTrade backtest engine if available, else homegrown simulator.
4. `backtesting/polymarket_backtest.py` — Brier score calibration, edge net
   of dynamic fees + 250ms delay, maker vs taker comparison.
5. Makefile targets wrapping FreqTrade's `lookahead-analysis` and
   `recursive-analysis` as CI gates.
6. `agents/risk_manager/agent.py` — equity-based daily loss limit.

### Step 7 — What to build
1. `agents/executor/safety.py` — startup permission check (refuse if withdraw
   enabled), order sanity checks (tick-size rounding, balance, bounds), kill
   switch, equity-based circuit breaker, ghost position detector.
2. `agents/executor/agent.py` — wire safety checks before every order.
3. `deploy/live.py` — confirmation gate (`LIVE_TRADING=true` + typed confirm),
   ghost position check at startup, absolute paths.
4. `tests/test_executor_dryrun.py` — CCXT failure matrix (partial fills,
   InvalidOrder, InsufficientFunds, RateLimitExceeded, RequestTimeout,
   AuthenticationError), tick-size rounding tests, idempotency, kill switch.

---

## Three-Stage Gated Rollout (from research synthesis)

```
Stage 1 — GATE: backtest quality
  ✓ lookahead-analysis passes
  ✓ recursive-analysis passes
  ✓ Monte Carlo P95 drawdown < capital tolerance
  ✓ Polymarket Brier score < 0.20 over ≥50 paper trades

Stage 2 — GATE: dry-run operational
  ✓ Paper entry/exit signals match backtest on same candles
  ✓ Executor test suite 100% passing (failure matrix)
  ✓ API keys are trade-only + withdraw disabled (verified programmatically)
  ✓ Kill switch and confirmation gate tested manually

Stage 3 — GATE: burn-in live
  ✓ Live ≈ paper over 100+ trades
  ✓ No ghost positions, no silent inactivity
  ✓ Slippage/fees within 20% of calibrated paper numbers
```
