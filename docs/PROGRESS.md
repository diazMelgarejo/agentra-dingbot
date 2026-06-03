# Agentic SuperBot — Build Progress

## Current State
**Version**: 0.3.0  
**Test suite**: 225 passed · 6 skipped · 0 failed  
**Last step completed**: Step 5 — FreqAI ML Bridge  

---

## Completed Steps

### Step 1 — Project Scaffold ✅
**Deliverable**: `agentic-superbot-v0_3_0_final.zip`

Everything the remaining steps build on:
- `core/config.py` — pydantic-settings singleton via `@lru_cache get_settings()`. Five config classes: ExchangeConfig, LLMConfig, TradingConfig, PolymarketConfig, AlertConfig (+ MLConfig added in Step 5).
- `core/state.py` — TradingState dataclass with 23 fields; 8 nested snapshot types; Signal/Timeframe/OrderStatus/MarketDirection enums.
- `core/orchestrator.py` — LangGraph 1.1.x StateGraph skeleton (properly filled in Step 4).
- Seven agent stubs with their final module paths established.
- Polymarket hybrid strategies from `@zostaff` + `@shmidtqq` X posts: `technical_signals.py`, `fear_filter.py`, `hybrid_decision.py`.
- Backtesting (walk-forward + Monte Carlo), deploy (async live/paper loop), FastAPI dashboard.
- Docker, Makefile, Apache-2.0 license, pre-commit, pyproject.toml.

---

### Step 2 — Data Ingestion ✅
**Deliverable**: `agentic-superbot-v0_3_0_step2.zip`  
**Tests**: 26 passed

Four independent data sources, all fetched concurrently in `fetch_full_snapshot()`:

| Module | Source | Auth | What it provides |
|--------|--------|------|-----------------|
| `data/fetcher.py` | Binance via CCXT | None (public) | OHLCV for 5m/1h/4h/1d |
| `data/polymarket.py` | Gamma API + CLOB REST | None (public) | BTC/ETH market discovery, YES prices, orderbooks, spread |
| `data/fear_greed.py` | alternative.me + yfinance | None (free) | F&G index + VIX; NORMAL/ELEVATED/EXTREME tiers |
| `data/websocket_stream.py` | Polymarket CLOB WS | None (public) | Live L2 orderbook; is_farmable (spread < 6¢) |

**Key design decisions locked in this step**:
- `asynccontextmanager` for all CCXT connections — zero leaks.
- Module-level imports in `fear_greed.py` so mocks can patch `data.fear_greed.aiohttp` etc. (lazy imports inside functions make mock patching impossible).
- Partial failure is non-fatal: if Polymarket API is down, spot pipeline continues. Errors are collected in `TradingState.errors`, never silently swallowed.

---

### Step 3 — TA Agent ✅
**Deliverable**: `agentic-superbot-v0_3_0_step3.zip`  
**Tests**: 69 passed · 5 skipped (TA-Lib equivalence — no system TA-Lib in CI)

The indicators engine has two compute paths and three outputs:

```
compute_all_indicators(df_4h)     → IndicatorSnapshot (spot pipeline)
compute_fast_indicators(df_5m)    → IndicatorSnapshot (Polymarket pipeline)
```

**Standard indicators (4h)**: RSI(14) Wilder EWM · BB(20,2σ) · EMA 9/21/50/200 · MACD(12,26,9) · ATR(14)  
**Derived fields**: `bb_pct` (price position within band), `atr_pct` (ATR as % of price), `ema_cross` (BULL/BEAR/FLAT)  
**Fast indicators (5m)**: MACD(3,15,3) histogram · RSI(14) · VWAP · CVD (cumulative volume delta)

**Bug fixed**: ATR was using `np.convolve` (simple MA) instead of Wilder's `ewm(com=13)` — now matches TA-Lib to within 0.5%.

**Multi-timeframe confirmation**: `_evaluate_standard(ind_4h, ind_1h)` adds +0.5 alignment bonus when 1h EMA agrees with 4h, and +0.3 RSI divergence bonus when 4h oversold + 1h recovering.

---

### Step 4 — LangGraph Orchestrator ✅
**Deliverable**: `agentic-superbot-v0_3_0_step4.zip`  
**Tests**: 183 passed (26+69+7+10+76 + 5 skipped)

The critical insight discovered during this step: **`dataclasses.asdict()` performs a deep conversion** — every nested `IndicatorSnapshot` object becomes a plain `dict`, causing `AttributeError: 'dict' has no attribute 'signal'` in downstream agents. Fixed with shallow field extraction in `_wrap()`:

```python
# WRONG — deep converts nested objects
state_dict = dataclasses.asdict(state)

# CORRECT — preserves IndicatorSnapshot, SentimentSnapshot, etc.
state_dict = {f.name: getattr(state, f.name) for f in dataclasses.fields(state)}
```

**Pipeline topology** (9 nodes, verified running):
```
ingest_data
    ├─► technical_analyst ─┐
    ├─► sentiment_analyst  ├─► debate_engine ─► risk_manager ─→(execute/skip)─► executor ─► END
    ├─► onchain_analyst    ┘
    └─► polymarket_agent ──────────────────────────────────────────────────────────────────► END
```

**Routing**: returns string keys `"execute"` / `"skip"` (not bool). VIX EXTREME overrides even an approved risk assessment at the routing level as a second circuit breaker.

---

### Step 5 — FreqAI ML Bridge ✅
**Deliverable**: `agentic-superbot-v0_3_0_step5.zip`  
**Tests**: 225 passed · 6 skipped (LightGBM — not in CI env)

A complete FreqAI-style ML layer without requiring FreqTrade:

**Backend fallback chain** (mirrors TA-Lib → pandas-ta pattern):
```
LightGBM LGBMClassifier  ──(if installed)──► primary
sklearn HistGBM           ──(always present)► fallback, NaN-tolerant
momentum heuristic        ──(final backstop)► never fails
```

**Feature matrix** (16 columns, pandas-only):
- Returns: r_1, log_r_1
- Lagged returns: r_lag_1/2/3/5
- Rolling momentum: roll_mean_5/10, roll_std_5/10
- Oscillators: rsi_14, macd_hist
- Trend: price_vs_ema_21, price_vs_ema_50
- Volatility: atr_pct_14
- Volume: vol_z_20

**Adaptive retraining**: sidecar `.meta.json` tracks `cycles_since_train`. When it reaches `ML_RETRAIN_INTERVAL` (default 50), the bridge retrains on fresh history. In-process LRU cache keeps inference fast within a session.

**Critical bug fixed during Step 5**: `Annotated[List[str], operator.add]` reducer on the `errors` field. With 4 parallel analysts all writing to `errors`, LangGraph's default `LastValue` channel raised `InvalidUpdateError`. The reducer merges concurrent writes via concatenation.

---

## Architecture Summary

```
Strategies (from X research)
  @zostaff  → MACD(3,15,3) + RSI/VWAP + CVD   → technical_signals.py
  @shmidtqq → VIX + F&G + micro impulse        → fear_filter.py
  Combined  → Bayesian edge ≥ 8% + Kelly 0.25× → hybrid_decision.py

Data Layer (Step 2)
  CCXT Binance OHLCV + Polymarket CLOB + alternative.me F&G + yfinance VIX

Agent Layer (Steps 3–5)
  technical_analyst  — TA-Lib/pandas 4h + 5m MACD/VWAP/CVD
  sentiment_analyst  — F&G contrarian + VIX muting + micro impulse
  onchain_analyst    — funding rates (bearish when high positive)
  ml_analyst         — FreqAI bridge (LightGBM/sklearn/heuristic)

Debate Layer (Step 4)
  Bull agent vs Bear agent → LLM judge → consensus signal + confidence

Risk Layer (Step 4)
  Mandell rules (STRONG_BUY 25%/1.5×ATR/3.0RR, BUY 15%/2.0×ATR/2.5RR)
  VIX circuit breakers (ELEVATED ≥30 halves size, EXTREME ≥40 blocks)
  Max-loss cap enforced

Execution Layer (Step 4)
  Spot: CCXT limit orders (dry_run=True default)
  Polymarket: CLOB fractional Kelly sizing (edge ≥ 8%)
```
