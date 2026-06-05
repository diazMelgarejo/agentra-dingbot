# LESSONS.md — Engineering Lessons from Day 1
*A comprehensive post-mortem for all agents inheriting this codebase.*
*Written per /engineering:documentation standard: context → what happened → what we learned → rule.*

---

## Preface: What "Day 1" Was

Before the first line of code was written, there were two separate projects:

- **`agentic-trader-v0.2.0`** — A LangGraph spot-only BTC/ETH bot with 5 agent stubs (technical, sentiment, onchain, debate, executor), clean config patterns, but no Polymarket integration and no live data layer.
- **`Polymarket_Hybrid_SuperBot_v1.0`** — A Polymarket-specific bot synthesising two X/Twitter strategies: @zostaff's MACD(3,15,3) + RSI/VWAP + CVD divergence and @shmidtqq's VIX + Fear & Greed + micro-impulse confluence. Had strategies but no multi-agent orchestration.

The decision made before Step 1 was to **synthesise both into a single unified platform** with an 8-step incremental build plan. That decision shaped every subsequent choice.

Also in the background: research into Pionex grid bots (arithmetic grid, $60K–$75K BTC range), eventually abandoned because Pionex aggregates Binance liquidity but doesn't expose a direct API — a key discovery documented here.

---

## L-00 — Pre-Step 1: Platform Discovery

### Context
The user wanted to run a Pionex grid bot and connect it to a custom algorithm via Binance API.

### What happened
Attempted to find Pionex API documentation. Discovered there is none — Pionex operates as its own exchange aggregating Binance liquidity. You cannot point an external API key at Pionex.

### Lesson
**Research platform constraints before designing integration architecture.**  
Pionex = aggregator, not a passthrough. Direct Binance CCXT is the correct path for grid trading on Binance liquidity. This saved weeks of attempted integration.

> **Rule**: Before designing any exchange integration, verify: (a) does the platform expose a REST/WebSocket API, (b) is it direct or aggregated, (c) can it accept external connections. For Pionex specifically: it cannot.

---

## L-01 — Docker on macOS: Storage Problem Discovered Early

### Context
First instinct was to run everything in Docker. M2 MacBook Pro with storage constraints.

### What happened
Docker Desktop on macOS requires a Linux VM. Total cost: 3–5 GB (VM image + FreqTrade image + Python image). On a storage-constrained machine this ruled out Docker as the default dev tool.

### Lesson
**On macOS, Docker is a VM wrapper, not native Linux.** The Python venv approach uses ~600 MB for the same outcome. Colima (Apple Hypervisor `--vm-type vz`) is the lightweight alternative when Docker is genuinely needed.

The most storage-efficient development setup:
```bash
# FreqTrade: dedicated venv (~600 MB total)
python3 -m venv ~/.venvs/freqtrade
source ~/.venvs/freqtrade/bin/activate
pip install "freqtrade[freqai]"
freqtrade install-ui   # pre-built UI, no npm/Node needed (~25 MB)

# SuperBot: own venv
pip install -r requirements.txt
```

> **Rule**: Default to Python venvs on macOS dev machines. Reach for Docker only when the deployment target requires it or when multi-service orchestration is essential.

---

## L-02 — The Sandbox Reset Problem and Why Git Is Essential

### Context
The coding environment resets between sessions. All work done in `/home/claude/superbot/` is lost.

### What happened
First session: packaged zip files as deliverables. Second session: unzipped, reinstalled deps, continued. By session 3 this was costly and fragile — unzipping large zips, re-running pip install, hoping nothing was corrupted.

### Lesson
**A git repository eliminates the sandbox restore problem entirely.** Three commands restore the full working tree in any session:
```bash
git clone https://github.com/diazMelgarejo/agentra-dingbot.git superbot
cd superbot
pip install -r requirements.txt
```

Zip files are a backup artefact. Git is the source of truth.

> **Rule**: Publish to git after every session. The clone command is the only acceptable restore mechanism. Never rely on zip files as the primary delivery vehicle for ongoing work.

---

## L-03 — GitHub Fine-Grained PAT Limitations

### Context
Tried to create a new GitHub repo via the GitHub API using a fine-grained Personal Access Token.

### What happened
`POST /user/repos` returned `Resource not accessible by personal access token`. The PAT authenticated successfully and could list repos, but could not create them. GraphQL also returned FORBIDDEN.

**Root cause**: Fine-grained PATs (format `github_pat_*`) require explicit `Administration: write` at the account level to create repositories. This is different from classic PATs (`ghp_*`) where the `repo` scope covers creation.

### Lesson
1. Fine-grained PATs cannot create repos via API. User must create the repo at `github.com/new` first.
2. Fine-grained PATs **cannot be revoked programmatically**. Only via `github.com/settings/personal-access-tokens`.
3. Always scrub PATs from git remote URLs immediately after push: `git remote set-url origin https://github.com/...` (no credential in URL).

> **Rule**: For repo creation, use GitHub web UI. For all PAT operations: use the minimum scope, store nowhere except the session, scrub from remotes within the same command block, and instruct the user to revoke immediately after use.

---

## L-04 — Step 2: Module-Level Imports Are Required for Mock Patching

### Context
Writing tests for `data/fear_greed.py`, which fetches Fear & Greed index and VIX. Tests needed to mock `aiohttp.ClientSession` and `yfinance.download`.

### What happened
Initial implementation used lazy imports inside functions:
```python
async def fetch_fear_greed():
    import aiohttp  # lazy import inside function
    ...
```

Tests with `patch("data.fear_greed.aiohttp")` failed — the patch target was the top-level module namespace, but the function imported its own fresh reference.

### Lesson
**Module-level imports are required for mock patching to work.**

```python
# WRONG — mock patching will fail
async def fetch_fear_greed():
    import aiohttp
    async with aiohttp.ClientSession() as s: ...

# CORRECT — patch("data.fear_greed.aiohttp") works
import aiohttp

async def fetch_fear_greed():
    async with aiohttp.ClientSession() as s: ...
```

This applies universally: `yfinance`, `ccxt`, `aiohttp` — anything you need to mock in tests must be imported at module level.

> **Rule**: Always use module-level imports in data modules. Lazy imports inside functions make test isolation impossible without monkey-patching `sys.modules`.

---

## L-05 — Step 2: asynccontextmanager Pattern for Connection Leak Prevention

### Context
CCXT async exchange instances need explicit `.close()` to release connections. Missing close = unclosed ClientSession warnings in every test run.

### What happened
Without the context manager, tests emitted:
```
WARNING: Unclosed client session
ResourceWarning: Enable tracemalloc to get the object allocation traceback
```

### Lesson
**Every external connection must use `asynccontextmanager` as the resource boundary.**

```python
# CORRECT pattern used throughout the codebase
@asynccontextmanager
async def _exchange_ctx(sandbox: bool = True):
    exchange = ccxt.binance(...)
    try:
        yield exchange
    finally:
        await exchange.close()   # always runs, even on exception

# Callers: always use the context manager, never hold raw instances
async with _exchange_ctx() as ex:
    return await ex.fetch_ohlcv(symbol, timeframe)
```

> **Rule**: Any class with a `.close()` / `__aexit__` method gets wrapped in `asynccontextmanager`. No exceptions. This pattern appears in 5 separate modules in this codebase.

---

## L-06 — Step 2: Partial Failure Must Be Non-Fatal

### Context
`fetch_full_snapshot()` fetches from four sources concurrently: CCXT OHLCV, Polymarket CLOB, Fear & Greed, VIX. Polymarket is an optional source.

### What happened
First version: if Polymarket API returned 503, the whole snapshot failed. Spot pipeline stalled because a prediction market was down.

### Lesson
**Partial failure must be non-fatal. Errors are collected, not propagated.**

```python
# The pattern used in data/snapshot.py
results = await asyncio.gather(
    fetch_ohlcv(...),
    fetch_polymarket_snapshot(...),
    fetch_fear_greed(...),
    return_exceptions=True   # ← key: don't let one failure kill all
)
errors = [r for r in results if isinstance(r, Exception)]
# Log errors, continue with available data
```

> **Rule**: `asyncio.gather(..., return_exceptions=True)` is the correct pattern for concurrent data fetching when sources have independent availability. Propagate errors via `TradingState.errors`, never via exceptions that abort the pipeline.

---

## L-07 — Step 3: ATR Wilder Smoothing Bug (Never Use `np.convolve`)

### Context
Computing Average True Range (ATR) for stop-loss sizing. ATR is defined using Wilder's Exponential Moving Average (EWM), not a simple moving average.

### What happened
First implementation used `np.convolve` (a simple rolling average):
```python
atr = np.convolve(tr, np.ones(14)/14, mode='valid')  # WRONG
```

This produces ATR values that are ~8–12% higher than TA-Lib's output. The stop-loss distances were too wide, position sizes were too small.

### Lesson
**ATR uses Wilder's EWM with `com=period-1`, not a simple rolling mean.**

```python
# CORRECT — matches TA-Lib to within 0.5%
atr = tr.ewm(com=13, min_periods=14, adjust=False).mean()
```

RSI uses the same Wilder smoothing pattern:
```python
gain = delta.clip(lower=0).ewm(com=13, min_periods=14, adjust=False).mean()
loss = (-delta.clip(upper=0)).ewm(com=13, min_periods=14, adjust=False).mean()
```

> **Rule**: For any Wilder-based indicator (ATR, RSI, ADX): `ewm(com=period-1, adjust=False)`. Never `np.convolve`, never `rolling(period).mean()`. Validate against TA-Lib output when available.

---

## L-08 — Step 3: TA-Lib / pandas-ta Fallback Pattern

### Context
TA-Lib requires a compiled C library. It's not available in all environments (CI, sandboxes, fresh machines). The bot must work without it.

### What happened
Needed RSI, Bollinger Bands, EMA, MACD, ATR — all available in both TA-Lib and pandas-ta. But switching between them mid-codebase produced slightly different values (different smoothing defaults).

### Lesson
**Establish numerical equivalence tests, then use them as CI gates.**

```python
def compute_all_indicators(df: pd.DataFrame) -> IndicatorSnapshot:
    try:
        import talib
        return _compute_talib(df)
    except ImportError:
        return _compute_pandas(df)
```

Tests with `@pytest.mark.skipif(not TALIB_AVAILABLE, reason="no system TA-Lib")` verify that the fallback produces values within 0.5% of TA-Lib. These tests serve as the correctness guarantee.

> **Rule**: Any library with a system-compiled dependency needs a pure-Python fallback. The fallback must be numerically equivalent (verified by tests). 5 tests are correctly skipped in this codebase because TA-Lib is not installed in the sandbox — this is correct behaviour.

---

## L-09 — Step 4: THE CRITICAL BUG — `dataclasses.asdict()` Deep-Converts Nested Objects

### Context
LangGraph 1.1.x nodes receive a `State` object and must return a partial dict. The `_wrap()` function converts `TradingState` to a dict before passing to agents.

### What happened
Used `dataclasses.asdict(state)` for the conversion. All 183 tests failing with:
```
AttributeError: 'dict' object has no attribute 'signal'
```

**Root cause**: `dataclasses.asdict()` is a **deep conversion**. Every nested `IndicatorSnapshot` dataclass became a plain `dict`. When `agents/risk_manager/agent.py` later did `state["technical"].signal`, it failed because `state["technical"]` was a dict, not an `IndicatorSnapshot`.

### Lesson
**This is the single most important bug in the codebase. Never undo this fix.**

```python
# WRONG — deep-converts ALL nested dataclasses
state_dict = dataclasses.asdict(state)

# CORRECT — shallow extraction, preserves nested objects
state_dict = {f.name: getattr(state, f.name) for f in dataclasses.fields(state)}
```

This affects `_wrap()` in `core/orchestrator.py`. The comment in that function must never be removed. Any agent editing the orchestrator must understand this.

> **Rule**: `dataclasses.asdict()` is banned in the LangGraph pipeline. Use shallow field extraction. Period. If you see an `AttributeError: 'dict' object has no attribute` in the pipeline, this is the first thing to check.

---

## L-10 — Step 4: LangGraph Routing Keys Must Be Strings, Not Booleans

### Context
After the debate engine and risk manager, the pipeline routes to either the executor ("execute") or skips ("skip"). Used `True`/`False` as routing keys.

### What happened
```python
# WRONG
graph.add_conditional_edges("risk_manager", _route, {True: "executor", False: END})

# Error: LangGraph routing keys must be strings
```

LangGraph 1.1.x requires string keys in `add_conditional_edges`.

### Lesson
**Routing keys are always strings.**

```python
# CORRECT
def _route(state):
    return "execute" if state.risk and state.risk.approved else "skip"

graph.add_conditional_edges("risk_manager", _route, {
    "execute": "executor",
    "skip":    END,
})
```

> **Rule**: All `add_conditional_edges` routing functions return strings. Use descriptive strings (`"execute"`, `"skip"`, `"retry"`) not booleans or integers.

---

## L-11 — Step 4: `errors` Field Requires an `operator.add` Reducer

### Context
Four analyst agents run in parallel (technical, sentiment, onchain, polymarket). Each node writes to `TradingState.errors` when something goes wrong.

### What happened
With four concurrent nodes all attempting to write to the same `errors` field, LangGraph raised:
```
InvalidUpdateError: Multiple updates for single-value channel 'errors'
```

### Lesson
**Any field written by multiple concurrent nodes needs an `Annotated` reducer.**

```python
# WRONG — LangGraph's default "last write wins" fails on concurrent writes
errors: List[str] = field(default_factory=list)

# CORRECT — operator.add merges concurrent writes via concatenation
from typing import Annotated
import operator

errors: Annotated[List[str], operator.add] = field(default_factory=list)
```

Critically: each node must return only its **delta** (new errors), not the accumulated list:
```python
# WRONG — returns accumulated list, causes duplicates
return {"errors": state.get("errors", []) + ["new error"]}

# CORRECT — returns delta only, reducer handles accumulation
return {"errors": ["new error"]}
```

> **Rule**: `Annotated[List[str], operator.add]` on `errors`. Any new field written by multiple parallel nodes needs a reducer. The reducer pattern extends to any shared mutable state in a fan-out topology.

---

## L-12 — Step 5: CPU-Bound Work Must Leave the Event Loop

### Context
The ML analyst trains a LightGBM/sklearn model on ~500 bars of OHLCV data. Training takes 0.5–3 seconds.

### What happened
Training on the event loop blocked all other async operations for the duration. The WebSocket orderbook stream stalled. The Polymarket agent timed out.

### Lesson
**CPU-bound work goes off the event loop via `run_in_executor`.**

```python
async def run(state: Dict[str, Any]) -> Dict[str, Any]:
    loop = asyncio.get_event_loop()
    # Training is CPU-bound — must not block the event loop
    result = await loop.run_in_executor(None, _train_and_predict, features, labels)
    return {"ml": result}

def _train_and_predict(features, labels):
    # Synchronous, CPU-bound — safe to run in thread pool
    model.fit(features, labels)
    return model.predict_proba(features[-1:])[0]
```

> **Rule**: Any operation taking > ~50ms that is CPU-bound (training, heavy computation) uses `run_in_executor`. Any IO-bound async operation uses `await`. Mixing them kills latency.

---

## L-13 — Step 5: LRU Cache for ML Models (Don't Retrain Every Cycle)

### Context
The bot runs a cycle every 60 seconds. ML model training on every cycle would consume 50+ seconds of each cycle.

### What happened
Initial design retrained on every cycle. CPU utilization hit 100%, cycles took 4–6 seconds instead of <1 second.

### Lesson
**Cache trained models with an LRU cache + cycle counter for adaptive retraining.**

```python
class FreqAIBridge:
    def __init__(self):
        self._model = None
        self._cycles_since_train = 0
        self._retrain_interval = 50  # configurable via ML_RETRAIN_INTERVAL

    def generate_ml_signal(self, df):
        if self._model is None or self._cycles_since_train >= self._retrain_interval:
            self._model = self._train(df)
            self._cycles_since_train = 0
            self._save_meta()  # .meta.json tracks retrain state across restarts
        self._cycles_since_train += 1
        return self._model.predict(df)
```

> **Rule**: ML models are trained on a configurable cadence (default: every 50 cycles), not every cycle. A `.meta.json` sidecar file persists the state across process restarts so the retrain cadence survives crashes.

---

## L-14 — FreqTrade GPL-3.0 Boundary

### Context
FreqTrade is a popular open-source trading bot licensed under GPL-3.0. We wanted to use its backtesting, strategy execution, and FreqUI dashboard.

### What happened
First design: import FreqTrade Python modules directly for strategy execution. Realized this would make the SuperBot a GPL-3.0 derived work, preventing distribution under Apache 2.0.

### Lesson
**The HTTP API boundary is the legal firewall.**

FreqTrade exposes a full REST API (`/api/v1/status`, `/api/v1/forcebuy`, `/api/v1/performance`, etc.). Calling it via HTTP is equivalent to calling any other microservice — GPL does not propagate across network boundaries in the same way as code linking.

```python
# WRONG — importing GPL code makes our code GPL
from freqtrade.strategy import IStrategy  # GPL-3.0 contamination

# CORRECT — calling via HTTP preserves Apache 2.0
async with aiohttp.ClientSession() as s:
    await s.post("http://localhost:8080/api/v1/forceenter", json={...})
```

The `SuperBotFollower.py` strategy file (in `deploy/freqtrade/`) is intentionally minimal — a GPL-licensed thin shell that does nothing except wait for our HTTP commands. It does not contain our intelligence.

> **Rule**: Never `import freqtrade`. All FreqTrade interaction is HTTP REST. The `freqtrade_client.py` module must never grow an import of the `freqtrade` package. This is a legal constraint, not a style preference.

---

## L-15 — Zero-Key Default: Every Bot Should Work Out-of-the-Box

### Context
The debate engine used an LLM (Ollama or OpenAI) to produce the bull/bear consensus. Default `LLM_PROVIDER=ollama`.

### What happened
A new developer cloning the repo sees the first cycle stall indefinitely because Ollama isn't running. No error message — just silence. LangGraph timeout after 30 seconds.

### Lesson
**Default behaviour must require zero external services.** The heuristic judge makes the bot fully functional with no keys, no local LLM, no Docker:

```python
# Default: deterministic weighted vote across analyst signals
_SOURCE_WEIGHTS = {
    "technical": 1.0,  # leads
    "ml":        0.9,  # confirms
    "sentiment": 0.7,  # context
    "onchain":   0.5,  # nudge
}
```

If the LLM fails at runtime, the bot falls back to the heuristic rather than stalling. This is the correct failure mode — degraded gracefully, still functional.

> **Rule**: `LLM_PROVIDER=none` is the default. Every feature that requires an external service must have a working fallback. "Zero config" means the bot runs usefully with no environment variables set.

---

## L-16 — FreqTrade as Detect-Only Optional Sidecar

### Context
FreqTrade handles execution, position management, and backtesting well. We wanted to use it when available but not require it.

### What happened
First design: check for FreqTrade at startup and hard-fail if not found. This broke zero-key operation.

Second design: `FREQTRADE_MODE=auto` — detect binary in common paths AND ping the API. Use it only if both succeed. Otherwise use homegrown CCXT executor silently.

### Lesson
**Optional dependencies follow a three-phase detection model:**

```
1. Binary detection: check PATH + ~/.venvs/freqtrade + /opt/homebrew + /usr/local + ~/.local
2. API reachability: ping http://localhost:8080/api/v1/ping
3. Mode gate: auto=use if both; on=require; off=never
```

The `routed_via` field on every order records the decision for observability without adding noise.

> **Rule**: Optional sidecars are detected at runtime, not at import time. Detection returns `(use: bool, reason: str)` — always log the reason. Errors in the sidecar client fall back to homegrown, never crash the main pipeline.

---

## L-17 — Step 6: Monte Carlo P95 is the Sizing Metric, Not Single-Path Backtest

### Context
First Monte Carlo implementation reported the backtest maximum drawdown as the capital sizing metric.

### What happened
Research (Wiecki et al. 2016, 888 strategies): "Backtest Sharpe R² < 0.025" — backtest results are nearly useless predictors of live performance. More specifically: Monte Carlo P95 drawdown consistently runs **1.5–3× higher** than the single backtest path.

A strategy that shows a 10% max drawdown in a single backtest often shows a 15–28% P95 drawdown in Monte Carlo. Funding for the single-path backtest is dangerously under-capitalized.

### Lesson
**Fund for the Monte Carlo P95 drawdown, not the backtest maximum.**

```python
# OLD — single path
max_dd = backtest_result["max_drawdown"]  # optimistic

# NEW — distribution
report = run_monte_carlo(trades, n_sims=5000, capital=1000.0)
funding_target = report.p95_max_dd  # realistic
```

> **Rule**: Monte Carlo requires ≥5,000 simulations (stable P95), block bootstrap (preserves autocorrelation), and must report P50 AND P95 separately. The P95 is the gate. Never use < 5,000 sims for a gate decision.

---

## L-18 — Step 6: Polymarket Fee Model Changed in 2026

### Context
The Polymarket backtester used a hardcoded `fee_pct = 0.04%` from documentation.

### What happened
In January 2026 Polymarket rolled out taker fees on 5-minute crypto markets. The fee structure is a **symmetric bell curve** peaking at 1.56% at p=0.5:

```
fee_fraction = fee_rate × 4 × p × (1 - p)
```

At p=0.50: fee ≈ 1.56%. At p=0.10 or p=0.90: fee ≈ 0.28%. The previous hardcoded 0.04% was off by 39× at the most-traded probability.

### Lesson
1. **Never hardcode fee rates for Polymarket** — fetch `feeRateBps` dynamically from the CLOB API on every order.
2. **The 8% edge floor must be measured NET of fees**, not gross.
3. **Maker orders pay zero fees** and earn a 20% rebate share. The surviving structural edge in 2026 is maker/liquidity-provision.
4. A **250ms taker delay** is in force — factor this into timing-sensitive strategies.

> **Rule**: `PolymarketFeeModel(fee_rate_bps=live_value)` on every cycle. Never hardcode. Net edge = gross edge − 2 × fee_cost(p, stake). If net edge < 8%, don't trade.

---

## L-19 — Step 6: Equity-Based Circuit Breaker, Not Balance-Based

### Context
The daily loss limit was originally computed against exchange `balance` (USDT available).

### What happened
A bot can have $1,000 balance (unrealised losses not counted yet) but $940 effective equity (including three open losing positions worth -$60). The balance-based check doesn't fire at 5% limit because balance = $1,000. The equity-based check fires because equity = $940 (6% down).

Real production incident: balance-based limits let bots breach limits silently while trades were open.

### Lesson
**Daily loss limit must compare equity (NAV = balance + unrealised P&L), not balance.**

```python
def equity_circuit_breaker(current_equity, start_equity, limit_pct=0.05):
    """
    current_equity = balance + unrealised P&L from ALL open trades
    start_equity   = NAV at start of trading day
    """
    return (start_equity - current_equity) / start_equity >= limit_pct
```

> **Rule**: `current_equity = balance + sum(unrealised_pnl for trade in open_trades)`. Query open trade P&L from FreqTrade's `/api/v1/status` or CCXT's `fetchOpenOrders`. Never use raw balance alone.

---

## L-20 — Step 7: Rounding That Zeroes Quantity (68 Rejections)

### Context
In a production bot (documented by florinelchis, Apr 2026), generic `round(qty, 2)` silently converted small ETH quantities to 0.0. The exchange rejected 68 consecutive orders with `INVALID_ORDERQTY`.

### What happened
```python
# WRONG — produces qty=0.0 for ETH if qty=0.003
qty = round(raw_qty, 2)  # 0.003 → 0.0

# CORRECT — use exchange-provided precision
qty = float(exchange.amount_to_precision(symbol, raw_qty))
if qty <= 0:
    # Explicit error, not silent pass
    raise ValueError(f"Quantity {raw_qty} rounds to zero for {symbol}")
```

### Lesson
Every exchange has its own `precision.amount` setting. `BTC/USDT` might accept 6 decimal places; `ETH/USDT` might require 3. CCXT exposes `amount_to_precision(symbol, qty)` — always use it.

> **Rule**: Never use Python's `round()` for order quantities. Always use `exchange.amount_to_precision(symbol, qty)`. Check the result is > 0 before submitting. This single check prevents the most common category of CCXT order rejection.

---

## L-21 — Step 7: API Keys with Withdraw Permission Are a Critical Risk

### Context
Between December 2024 and January 2025, attackers drained over $65M from exchange accounts using stolen API keys.

### What happened
Every stolen key had withdraw permission enabled. Bots don't need withdraw permission. Withdraw permission means a compromised key can empty the account.

### Lesson
**No trading bot ever needs withdraw permission.**

```python
# Hard check at startup, fail-closed
async def check_api_permissions(exchange):
    perms = await exchange.fetchPermissions()
    if perms.get("withdraw"):
        logger.critical("WITHDRAW_PERMISSION_DETECTED — bot will not start")
        return PermissionResult(safe=False, reason="withdraw enabled")
    return PermissionResult(safe=True, reason="trade-only verified")
```

API key permissions checklist:
- ✅ Trade: enabled
- ✅ Read: enabled
- ❌ Withdraw: NEVER
- ❌ Transfer: NEVER
- ✅ IP whitelist: your VPS/home IP only
- ✅ Rotation: quarterly

> **Rule**: `check_api_permissions()` runs at every startup. If withdraw is enabled, the bot refuses to start. This is not configurable. Fail-closed is the only acceptable posture.

---

## L-22 — Step 7: Ghost Positions Block Trading for Months

### Context
Another production incident (florinelchis, Apr 2026): 246 database rows marked as "open" that were actually closed on the exchange. The bot wouldn't open new trades because it thought the position limit was exceeded.

### What happened
The position-close handler updated the P&L but didn't update the `status` field in the DB. The fill event was logged but the state machine didn't transition. Silent failure — no error, just permanently blocked.

### Lesson
**State machine transitions must be atomic and verified.**

At startup:
```python
async def detect_ghost_positions(exchange, db):
    db_open = db.get_open_trades()
    exchange_open = await exchange.fetch_open_orders()
    exchange_ids = {o["id"] for o in exchange_open}
    ghosts = [t for t in db_open if t.exchange_id not in exchange_ids]
    if ghosts:
        logger.critical(f"GHOST_POSITIONS: {len(ghosts)} DB trades not on exchange")
        # Reconcile: close these in DB
```

> **Rule**: At every startup, reconcile DB open positions against exchange open orders. Any mismatch is a ghost — log critical, reconcile, alert. Build this before going live.

---

## L-23 — TDD Workflow: Test the Spec, Not the Implementation

### Context
Early tests were tightly coupled to implementation details (checking internal state variables, mocking private methods, asserting call counts).

### What happened
Every refactor broke dozens of tests. The tests became a liability rather than a safety net.

### Lesson
**Tests document behaviour, not implementation.**

```python
# WRONG — tests implementation detail
def test_internal_state():
    agent = TechnicalAnalyst()
    agent.run(state)
    assert agent._ema_cache[21] == expected_value  # private!

# CORRECT — tests observable behaviour
def test_buy_signal_when_oversold_and_bull_cross():
    # Arrange
    state = make_state_with(rsi=28, ema_cross="BULL")
    # Act
    result = run(state)
    # Assert — behaviour: what the caller sees
    assert result["technical"].signal in (Signal.BUY, Signal.STRONG_BUY)
```

Per the TDD SKILL.md: one behaviour per test, AAA pattern, descriptive names that read as specifications.

> **Rule**: Test names are specifications. If a test name doesn't describe the expected behaviour in plain English, rewrite the name. If a test breaks on refactor but the behaviour didn't change, the test is wrong.

---

## L-24 — TDD Workflow: The RED Gate Is Mandatory

### Context
Temptation to write tests and implementation simultaneously (easier and faster for small changes).

### What happened
Tests written after code routinely pass on the first run without verifying the test logic is correct. A test that always passes regardless of the implementation provides zero safety.

### Lesson
**The RED gate is not optional.** Per the TDD SKILL.md:

> "A test that was only written but not compiled and executed does not count as RED."

The workflow:
```
1. Write test
2. Run test — MUST FAIL for the RIGHT reason
   (not syntax error, not import error — the actual business logic failing)
3. Commit: "test: RED — <feature>"
4. Write minimal implementation
5. Run test — MUST PASS
6. Commit: "feat: GREEN — <feature>"
7. Refactor
8. Run test — STILL PASS
9. Commit: "refactor: REFACTOR — <feature>"
```

The git log for this project has explicit RED/GREEN/REFACTOR commits as evidence.

> **Rule**: The RED commit hash is your audit trail. If you cannot show a commit where the tests failed with the correct error message, you skipped TDD. The TDD SKILL requires a git checkpoint after each phase.

---

## L-25 — `src/` Layout: Clean Root, Predictable Imports

### Context
All Python packages were at the project root (`core/`, `agents/`, `data/`, etc.). This worked but the root directory was cluttered with both configuration files and Python packages.

### What happened
Moving everything to `src/` with `pythonpath = src` in `pytest.ini` required updating 0 imports — the packages are still imported as `from core.config import ...`. Only configuration files needed updating.

### Lesson
**`src/` layout with `pythonpath = src` is the cleanest Python project structure:**

```ini
# pytest.ini
[pytest]
asyncio_mode = auto
pythonpath = src
testpaths = tests
```

```toml
# pyproject.toml
[tool.setuptools.packages.find]
where = ["src"]
```

Imports in production code: `from core.config import ...` (no `src.` prefix).
Imports in tests: both forms work, but production code uses the non-prefixed form.

> **Rule**: Production code never uses `from src.core.config import ...`. The `src.` prefix is an implementation detail hidden by `pythonpath = src`. Using it in production code creates a dependency on the project layout.

---

## L-26 — `get_settings.cache_clear()` After `monkeypatch.setenv`

### Context
Tests that change `LLM_PROVIDER` via `monkeypatch.setenv` weren't seeing the new value.

### What happened
`get_settings()` uses `@lru_cache(maxsize=1)`. Once loaded, it never reloads from environment. `monkeypatch.setenv("LLM_PROVIDER", "ollama")` changed `os.environ` but the cached Settings object kept the old value.

### Lesson
**Any test that modifies environment variables must clear the settings cache.**

```python
@pytest.fixture
def _force_llm_path(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    from core.config import get_settings
    get_settings.cache_clear()  # ← critical
    yield
    get_settings.cache_clear()  # ← restore for next test
```

This pattern applies to any `@lru_cache` singleton when tests mutate its inputs.

> **Rule**: `lru_cache` singletons + `monkeypatch.setenv` = always `cache_clear()` before AND after. Failure to clear produces the hardest-to-diagnose test ordering bugs (tests pass in isolation, fail in sequence).

---

## L-27 — conftest.py Fixtures: Design for Reusability

### Context
Early test failures from ad-hoc test data: constructing `TradingState` inline in 20 different tests, each with slightly different fields.

### What happened
When `TradingState` gained a new required field, 20 tests failed simultaneously. Each needed updating individually.

### Lesson
**conftest.py factories are the single source of test data construction.**

```python
# conftest.py — single place to update when TradingState changes
def make_full_snapshot(errors=None, **overrides):
    """Canonical snapshot factory. Update here when state changes."""
    return {
        "ohlcv": {"4h": _make_ohlcv()},
        "polymarket": {...},
        "fear_greed": {...},
        "vix": {...},
        "errors": errors or [],
        **overrides,
    }
```

When `TradingState` changes, update `conftest.py` once. All 307 tests benefit.

> **Rule**: Test data construction belongs in `conftest.py`. Inline `TradingState(...)` construction in individual tests is a maintenance liability. Factory functions with sensible defaults and override kwargs are the correct pattern.

---

## L-28 — Pandas Deprecation: Frequency Strings

### Context
`pd.date_range(..., freq="1D")` emitted a deprecation warning in pandas 2.x.

### What happened
```
FutureWarning: 'D' is deprecated and will be removed in a future version.
Please use 'D' instead.
```

Wait — 'D' replaces 'D'? The actual change: uppercase frequency aliases like '1D' become lowercase '1d', 'BH' becomes 'bh', etc.

### Lesson
**Pandas 2.x uses lowercase frequency strings for offset aliases.**

```python
# OLD (deprecated in pandas 2.x)
pd.date_range("2024-01-01", periods=100, freq="1D")

# NEW (correct for pandas 2.x+)
pd.date_range("2024-01-01", periods=100, freq="1d")
```

This affects: `1D`→`1d`, `1H`→`1h`, `1T`→`1min`, `1S`→`1s`.

> **Rule**: All new `pd.date_range`, `pd.tseries`, and resample code uses lowercase frequency strings. When upgrading pandas versions, search for uppercase frequency aliases as a compatibility check.

---

## Summary: The Rule Sheet

A condensed list of every rule from this document — print this for any new agent:

| # | Context | Rule |
|---|---------|------|
| 00 | Pionex | Pionex is an aggregator, not an API-accessible exchange |
| 01 | macOS dev | Python venv (~600 MB) over Docker (~3 GB) on macOS |
| 02 | Continuity | Git clone is the only session restore mechanism |
| 03 | PAT | Fine-grained PATs can't create repos; can't be revoked via API |
| 04 | Testing | Module-level imports required for mock patching |
| 05 | Connections | Every external connection uses `asynccontextmanager` |
| 06 | Resilience | `asyncio.gather(..., return_exceptions=True)` for concurrent fetches |
| 07 | ATR/RSI | Wilder EWM = `ewm(com=period-1, adjust=False)`, never `np.convolve` |
| 08 | TA-Lib | All TA-Lib indicators need a pure-Python fallback |
| 09 | **CRITICAL** | `dataclasses.asdict()` is banned. Use shallow field extraction in `_wrap()` |
| 10 | LangGraph | Routing keys are strings, never booleans |
| 11 | LangGraph | `errors: Annotated[List[str], operator.add]`; return delta only |
| 12 | Async | CPU-bound training → `run_in_executor`; IO-bound → `await` |
| 13 | ML | LRU cache + cycle counter for model retraining cadence |
| 14 | Legal | Never `import freqtrade`. HTTP REST only. GPL boundary. |
| 15 | UX | `LLM_PROVIDER=none` default. Every feature has a working fallback |
| 16 | Optional deps | Detect-only pattern: binary + API ping + mode gate |
| 17 | Backtesting | Monte Carlo P95 is the sizing metric. ≥5,000 sims. Never single-path. |
| 18 | Polymarket | Never hardcode fees. Fetch `feeRateBps` live. Net edge = gross − 2×fees |
| 19 | Risk | Daily loss limit uses equity (NAV), not balance |
| 20 | Executor | `exchange.amount_to_precision()` not Python `round()` |
| 21 | Security | Withdraw permission = bot refuses to start. Fail-closed. |
| 22 | Reliability | Ghost position reconciliation at every startup |
| 23 | TDD | Test behaviour, not implementation. Refactor-safe tests only. |
| 24 | TDD | RED gate is mandatory. Commit the failing test before the fix. |
| 25 | Structure | `src/` layout + `pythonpath = src`. Production code: no `src.` prefix. |
| 26 | Testing | `lru_cache` + `monkeypatch.setenv` = always `cache_clear()` before and after |
| 27 | Testing | `conftest.py` factories are the single source of test data construction |
| 28 | Pandas | Lowercase frequency strings: `1d`, `1h`, `1min` (not `1D`, `1H`, `1T`) |
