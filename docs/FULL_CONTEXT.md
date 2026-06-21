# Agentra DingBot — Full Context Document
*Written for continuity in Claude Code CLI and the agent council.*
*Start here if you have no prior context on this project.*

---

## 1. What This Project Is and Why It Exists

**Agentra DingBot** is a production-grade multi-agent algorithmic trading platform
built incrementally over several sessions from June 2026. It synthesises two trading
strategies sourced from X/Twitter:
- **@zostaff's system**: MACD(3,15,3) + RSI + VWAP + CVD divergence, 5-minute signals
- **@shmidtqq's system**: VIX + Fear & Greed Index + micro-impulse confluence

These strategies feed a **LangGraph orchestration** of four parallel analyst agents,
a deterministic debate engine (heuristic by default — zero external keys needed),
a risk manager, and a **dual executor** (spot via CCXT Binance + prediction markets
via Polymarket CLOB).

The platform also integrates a **filing-driven regime allocation framework** (from a
2026 sample trade-plan spec) that extracts institutional factor tilts without
copying individual trades.

**Key design philosophy**: every external dependency is optional. The bot runs
completely offline with `LLM_PROVIDER=none`. FreqTrade is an optional sidecar.
No Docker required. Default mode is always paper/dry-run.

---

## 2. The 8-Step Build Arc (all complete)

### Step 1 — Project Scaffold
- pydantic-settings config (`core/config.py`)
- `TradingState` dataclass (flat, LangGraph-safe)
- 8 agent module stubs
- FastAPI dashboard stub

### Step 2 — Data Ingestion (26 tests)
- CCXT async OHLCV (Binance, multi-timeframe: 5m, 1h, 4h)
- Polymarket Gamma API (market listing) + CLOB API (orderbook, fills)
- alternative.me Fear & Greed Index
- yfinance VIX
- WebSocket orderbook with auto-reconnect and partial-failure non-fatal design

**Critical pattern**: `asyncio.gather(..., return_exceptions=True)` — if Polymarket
is down, spot trading continues unaffected.

### Step 3 — Technical Analyst (74 tests)
- TA-Lib primary / pandas-ta fallback (numerically equivalent, tested)
- Fast 5-min signals: MACD(3,15,3) + RSI + VWAP + CVD for Polymarket
- Slow 4h signals: EMA21/55 + BB + ATR14 + RSI14 for spot sizing
- Derived fields: `bb_pct`, `atr_pct`, `ema_cross`

**Critical bug fixed**: ATR/RSI must use Wilder's EWM `ewm(com=period-1, adjust=False)`
not `np.convolve`. Error was ~8-12% ATR overestimate.

### Step 4 — LangGraph Orchestrator (76 tests)
The most complex step. Critical bugs found and fixed:
1. `dataclasses.asdict()` deep-converts nested objects → broke `.signal` attribute access
   **Fix**: `{f.name: getattr(s,f.name) for f in fields(s)}` (shallow)
2. Routing keys must be strings, not booleans
3. `errors: Annotated[List[str], operator.add]` reducer for concurrent fan-out
4. LangGraph 1.1.x-specific API (nodes receive state object, return partial dict)

**Zero-key mode**: `LLM_PROVIDER=none` → heuristic judge (weighted vote:
technical×1.0, ml×0.9, sentiment×0.7, onchain×0.5).

### Step 5 — FreqAI ML Bridge (42 tests)
- LightGBM primary / sklearn-HGB fallback
- 16-feature pipeline (RSI, MACD, BB, ATR, EMA cross, funding rates, F&G, VIX)
- In-process LRU cache: retrain every 50 cycles (configurable)
- CPU-bound training runs off event loop via `run_in_executor`
- FreqTrade is optional detect-only sidecar (never `import freqtrade` — GPL boundary)

### Step 6 — Backtest Validation
- **Walk-forward**: purged + embargo splits (label_horizon + embargo_bars gap between
  train-end and test-start to prevent leakage)
- **Monte Carlo**: block bootstrap (size=10, preserves autocorrelation), P5/P50/P95/P99
  for drawdown AND equity AND Sharpe. **P95 max-DD is the gate metric** — research
  shows P95 runs 1.5–3× the median. Fund against P95, not P50.
- **Signal replay**: save/load SuperBot signals → replay on OHLCV → compute metrics
- **Polymarket backtest**: Brier score calibration (< 0.20 gate, beats baseline),
  dynamic fee model (peaks 1.56% at p=0.5 in 2026), net edge after fees
- **Key finding**: Backtest Sharpe has R² < 0.025 predictive value (Wiecki 2016)

### Step 7 — Executor Safety
Key lessons from documented production incidents:
- **68 consecutive rejections**: `round(qty, 2)` zeroed ETH quantities → fix:
  `exchange.amount_to_precision(symbol, qty)` always
- **$65M+ drained**: API keys with withdraw permission → fix: refuse to start if
  withdraw enabled (hard fail, no override)
- **Ghost positions**: 246 open DB rows that were closed on exchange → fix:
  startup reconciliation
- **Silent inactivity**: bot "running" but not trading for 28 days

Safety modules built:
- `agents/executor/safety.py`: permission check (fail-closed), order validation,
  KillSwitch (file-based), `is_live_trading_enabled()` (LIVE_TRADING=true opt-in)
- CCXT retry wrapper: bounded retries, auth error halts immediately
- Equity circuit breaker: NAV-based (includes unrealised P&L), not balance-based

### Step 8 — Dashboard + GitHub Pages
- `docs/index.html`: 738-line self-contained static file (no build step)
- Clarence design system: color encodes state (green=bull, red=bear, amber=warn, grey=neutral)
- Light/dark themes via CSS custom properties, persisted in localStorage
- Lightweight Charts v4 (Apache-2.0) candlestick, fed by CCXT data
- **Three-tier fallback**: live WebSocket (`ws://localhost:8000/ws/signals`) →
  snapshot.json poll (`docs/data/snapshot.json`) → demo mode
- GitHub Pages deployed via `actions/configure-pages` + `actions/upload-pages-artifact`
  + `actions/deploy-pages` (no third-party action)

---

## 3. Portfolio / Regime Allocation Framework

From the 2026 sample filing trade-plan. **This is NOT copy-trading.** Edge is in
regime alignment, not ticker imitation.

```yaml
# config/portfolio.yaml — four sleeves, weights sum to 1.0
core_beta:         60%  # S&P 500 + Nasdaq/AI + Bitcoin + cash
ai_infrastructure: 25%  # NVDA, MSFT, AVGO, AMZN, ORCL
bitcoin_sleeve:    10%  # spot BTC in tranches (ETF flow + trend required)
tactical_swing:     5%  # Mandell system (4 conditions must ALL be met)

Risk caps: 25% per name · 2%/trade · 6% heat · ≥3:1 R:R
```

Decision logic (from the spec):
- Add risk **after** event-driven uncertainty resolves, never before
- Scale in on weakness / post-rebalance dislocation; don't chase breakouts
- Bitcoin as macro hedge, not another tech proxy

**Council runner**: `python scripts/run_council.py --backend ollama --model llama3.1:8b`
See `docs/COUNCIL_PROMPT.md` for the full template with 4-role structure and YAML output contract.

---

## 4. Dashboard Data Flow

```
TradingState
    ↓
to_dashboard_view()   ← src/dashboard/state_view.py (single source of truth)
    ↓
┌─────────────────────────────────────────────────────┐
│  Live mode (backend running)                         │
│  WebSocket /ws/signals → push every 60s             │
└─────────────────────────────────────────────────────┘
    ↓ (if backend unreachable)
┌─────────────────────────────────────────────────────┐
│  Snapshot mode (GitHub Pages)                        │
│  Poll docs/data/snapshot.json (cache-busted)        │
│  Updated daily by .github/workflows/snapshot.yml    │
│  Admin: make snapshot-live OR trigger workflow_dispatch │
└─────────────────────────────────────────────────────┘
    ↓ (if snapshot 404)
┌─────────────────────────────────────────────────────┐
│  Demo mode — simulated live-updating data           │
│  No backend, no network, works offline              │
└─────────────────────────────────────────────────────┘
```

**Snapshot loop-safety**: The cron workflow uses the default `GITHUB_TOKEN`.
GitHub does NOT re-trigger workflows from `GITHUB_TOKEN` commits, so there is
no infinite loop. A `git diff --staged --quiet` guard prevents empty commits.
`[skip ci]` is added as defense-in-depth.

---

## 5. Research Findings (from June 2026 session)

### Snapshot workflow — key patterns
- **Use default GITHUB_TOKEN, not PAT**: PAT commits re-trigger workflows → infinite loop
- **Commit-bot identity**: `github-actions[bot]` / `41898282+github-actions[bot]@users.noreply.github.com`
- **Change guard**: `git diff --staged --quiet || git commit …`
- **keep-alive**: scheduled workflows auto-disabled after 60 days with no repo activity
  (only commits/PRs/issues reset the clock)

### GitHub Pages polling — key patterns
- **`res.ok` check is mandatory**: `fetch()` does NOT reject on 404; `res.ok` must be
  checked explicitly or demo fallback never fires
- **Cache-busting**: both `?t=${Date.now()}` (unique URL) AND `{cache:'no-store'}` needed
  (browser + Pages CDN serve stale aggressively)
- **`setTimeout` over `setInterval`**: prevents request pile-up under latency

### LangGraph council patterns
- One markdown file, one `##` header per role, shared output contract
- Supervisor routes with `llm.with_structured_output(Router)` → `Command(goto=...)`
- `reasoning` field in Router improves routing accuracy (lightweight CoT)
- Parse YAML output with `yaml.safe_load` (never `yaml.load`) + pydantic validate

### pydantic-settings YAML config
- `YamlConfigSettingsSource` requires overriding `settings_customise_sources` (not just `yaml_file=`)
- Source order = precedence (put `env_settings` before YAML if env should override)
- `@model_validator(mode="after")` for cross-field rules (weights-sum, cap checks)
- `@field_validator` for single-field range checks

---

## 6. Full Test Suite Inventory

| Test file | Tests | What it covers |
|-----------|-------|----------------|
| `test_data_ingestion.py` | 26 | CCXT OHLCV, Polymarket, F&G, VIX, partial failure |
| `test_ta_agent.py` | 64 | TA indicators, TA-Lib/pandas fallback, MACD fast |
| `test_technical_analyst.py` | 10 | Multi-timeframe signals |
| `test_langgraph_pipeline.py` | 76 | Full LangGraph cycle, agent routing, zero-key |
| `test_ml_bridge.py` | 42 | FreqAI bridge, sklearn-HGB, cache, feature parity |
| `test_risk_manager.py` | 7 | ATR stops, VIX gates, Kelly sizing |
| `test_freqtrade_optional.py` | 21 | FreqTrade optional, HTTP client, zero-key pipeline |
| `test_step6_backtest.py` | 40 | Monte Carlo P95, walk-forward, signal replay, Brier |
| `test_step7_executor.py` | 21 | Safety, CCXT matrix, kill switch, permission gate |
| `test_portfolio_config.py` | 6 | Portfolio loader, weights sum, concentration cap |
| `test_dashboard_snapshot.py` | 8 | State view mapper, snapshot exporter envelope |
| **Total** | **321** | 6 correctly skipped (no TA-Lib / LightGBM) |

---

## 7. Remaining Work (Prioritised)

### P0 — One click away
- **Enable GitHub Pages** in repo Settings → Pages → Source: GitHub Actions *(already done per screenshot)*
- **Open PR for feature branch** `2026-06-20-snapshot-council-livebind`

### P1 — Dashboard completion
- Wire real Polymarket markets into WebSocket payload (field `polymarket_markets` is
  populated by `polymarket_agent` but not yet extracted in `_serialize_state`)
- Playwright smoke test: assert KPIs render, theme toggle works, chart mounts
- Snapshot freshness badge: warn when `meta.generated_at` > 48h old

### P2 — Portfolio / council
- `portfolio_agent` LangGraph node: reads `config/portfolio.yaml` → emits sleeve
  targets into `TradingState`; blocks execution when regime is `risk_off`
- Walk-forward backtest of the regime allocation vs. buy-and-hold 2024–2026
- `scripts/run_council.py` integration test (mock LLM backend)
- Correlation-cluster guard: enforce `max_correlated_positions: 3` at order time

### P3 — Production hardening
- Real-device responsive QA (sidebar collapse @1100px, single-col @560px)
- Docs: "How to extend the dashboard" guide (adding a panel, KPI card)
- Optional privacy-friendly analytics for public Pages site

### P4 — Larger horizon (from `docs/FUTURE_PLANS.md`)
- LLaVA visual agent (5th analyst on chart screenshots via Ollama)
- On-chain data expansion (MVRV, exchange flows, OI from Glassnode/CryptoQuant)
- ETH/USDT multi-symbol support end-to-end
- PostgreSQL + TimescaleDB for persistent trade history
- Grafana monitoring overlay
- FreqAI hyperparameter tuning (optuna search over feature window + tree depth)

---

## 8. Files Modified Most Recently (branch: `2026-06-20-snapshot-council-livebind`)

```
CLAUDE.md                              ← NEW: Claude Code entry point
scripts/run_council.py                 ← NEW: council runner (Ollama/Anthropic/etc.)
src/dashboard/state_view.py            ← NEW: dashboard shape single source of truth
src/dashboard/snapshot_export.py       ← NEW: writes docs/data/snapshot.json
src/dashboard/app.py                   ← MODIFIED: WS uses dashboard shape, POLL_SECONDS
src/agents/risk_manager/agent.py       ← MODIFIED: min_reward_to_risk gate from portfolio
src/strategies/portfolio_config.py     ← NEW: pydantic portfolio YAML loader
config/portfolio.yaml                  ← NEW: four-sleeve allocation config
docs/index.html                        ← MODIFIED: 3-tier fallback (WS→snapshot→demo)
docs/data/snapshot.json                ← NEW: initial demo snapshot for Pages
.github/workflows/snapshot.yml         ← NEW: daily cron snapshot export
.github/workflows/pages.yml            ← EXISTS: GitHub Actions Pages deploy
```

---

## 9. Key Operational Commands for Claude Code CLI

```bash
# Check test health
python -m pytest tests/ -q

# Run a single test file
python -m pytest tests/test_step6_backtest.py -v

# TDD cycle — write RED test first
python -m pytest tests/test_new_feature.py -v  # must fail
# ... implement ...
python -m pytest tests/test_new_feature.py -v  # must pass

# Generate a snapshot (demo, no network)
PYTHONPATH=src python -m dashboard.snapshot_export --mode demo

# Generate a snapshot (real cycle, needs CCXT)
PYTHONPATH=src LLM_PROVIDER=none python -m dashboard.snapshot_export --mode cycle

# Run council (demo — no LLM needed)
python scripts/run_council.py --backend demo

# Validate portfolio config
PYTHONPATH=src python -m strategies.portfolio_config

# Start the dashboard backend
PYTHONPATH=src LLM_PROVIDER=none python src/dashboard/app.py

# Open dashboard (static, instant)
open docs/index.html

# Push to branch (use GITHUB_PAT env var — never hardcode)
git remote set-url origin "https://diazMelgarejo:${GITHUB_PAT}@github.com/diazMelgarejo/agentra-dingbot.git"
git push origin <branch>
git remote set-url origin https://github.com/diazMelgarejo/agentra-dingbot.git
```

---

## 10. Important Warnings

1. **Revoke GitHub PATs immediately** after use. Fine-grained PATs cannot be revoked
   via API — only at https://github.com/settings/personal-access-tokens
2. **`LIVE_TRADING=true`** triggers real money. Default is paper. Requires typed "LIVE"
   confirmation + startup permission check.
3. **Never `import freqtrade`** — GPL-3.0 contamination. HTTP REST only.
4. **Never `dataclasses.asdict()`** in the LangGraph `_wrap()` function.
5. **Portfolio council output requires human approval** before any implementation.
6. **PAT commits made with `GITHUB_TOKEN` don't re-trigger workflows** (safe).
   PAT commits DO re-trigger — always use default token in GitHub Actions.
