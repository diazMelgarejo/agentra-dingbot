# CLAUDE.md — Agentra DingBot
*Claude Code reads this file automatically at the start of every session.*
*Keep it current. It is the single source of truth for any agent picking up this project.*

---

## What This Is

A **multi-agent algorithmic trading platform** for BTC/ETH spot (Binance/CCXT) and
Polymarket 5-minute BTC Up/Down prediction markets. Four parallel analyst agents
(technical, sentiment, on-chain, ML) feed a heuristic debate engine → risk manager →
dual executor. Fully functional with zero external keys (`LLM_PROVIDER=none`).

**GitHub:** https://github.com/diazMelgarejo/agentra-dingbot
**Live dashboard:** https://diazmelgarejo.github.io/agentra-dingbot/

---

## Project State (as of June 2026)

| Step | What | Status |
|------|------|--------|
| 1 | Scaffold, config, state | ✅ |
| 2 | Data ingestion (CCXT, Polymarket, F&G, VIX) | ✅ |
| 3 | TA agent (TA-Lib/pandas, MACD/RSI/VWAP/CVD) | ✅ |
| 4 | LangGraph orchestrator (dual pipeline) | ✅ |
| 5 | FreqAI ML bridge (LightGBM/sklearn-HGB) | ✅ |
| 6 | Backtest validation (walk-forward, Monte Carlo P95) | ✅ |
| 7 | Executor safety (kill switch, permission check, LIVE gate) | ✅ |
| 8 | Static dashboard + GitHub Pages + snapshot export | ✅ |
| — | Portfolio config + council prompt | ✅ |

**Tests:** 321 passing, 6 skipped (correct — no system TA-Lib / LightGBM)

---

## How to Resume in Any Session

```bash
# 1. Clone (the only restore mechanism — no zips)
git clone https://github.com/diazMelgarejo/agentra-dingbot.git
cd agentra-dingbot

# 2. Install (all in one)
pip install pytest pytest-asyncio structlog pandas numpy pydantic pydantic-settings \
    aiohttp ccxt yfinance pandas-ta langgraph scikit-learn joblib fastapi uvicorn pyyaml

# 3. Verify green (must be 321 passed, 6 skipped)
python -m pytest tests/

# 4. Check portfolio config
make portfolio

# 5. Start demo (no backend, no keys)
open docs/index.html
```

---

## Source Layout

```
src/                     ← ALL Python (pythonpath=src in pytest.ini)
  core/                  config.py, state.py, orchestrator.py, cli.py
  data/                  fetcher.py, polymarket.py, fear_greed.py, websocket_stream.py
  agents/
    technical_analyst/   TA-Lib/pandas indicators (4h+5m), MACD/RSI/VWAP/CVD
    sentiment_analyst/   Fear & Greed + VIX + micro-impulse
    onchain_analyst/     Binance perpetual funding rates
    ml_analyst/          FreqAI/sklearn-HGB signal
    debate_engine/       heuristic vote (LLM_PROVIDER=none) or LLM bull/bear/judge
    risk_manager/        ATR stops, VIX breakers, equity circuit, Kelly sizing
    executor/            CCXT spot + Polymarket CLOB + safety.py
    polymarket_agent/    Hybrid decision + fractional Kelly
  ml/                    features.py, labels.py, model.py, freqai_bridge.py
  strategies/            technical_signals.py, fear_filter.py, portfolio_config.py
  backtesting/           monte_carlo.py, walk_forward.py, signal_replay.py, polymarket_backtest.py
  deploy/                live.py, paper_broker.py
  dashboard/             app.py (FastAPI/WS), state_view.py, snapshot_export.py
  utils/                 logger.py, telegram_alerts.py, orderbook.py
config/portfolio.yaml    ← multi-sleeve allocation config
docs/index.html          ← self-contained static dashboard (GitHub Pages)
docs/data/snapshot.json  ← committed daily by snapshot.yml workflow
scripts/run_council.py   ← council runner (Ollama/LM Studio/Anthropic/OpenAI)
```

---

## Import Convention (critical)

```python
# CORRECT — no src. prefix (pythonpath = src adds src/ to sys.path)
from core.config import get_settings
from agents.executor.safety import KillSwitch

# WRONG — don't use src. prefix in production code
from src.core.config import get_settings   # breaks when installed
```

---

## TDD Workflow (mandatory for new code)

```
1. SPECS.md entry → 2. failing tests (🔴 RED, commit) → 3. minimal code (🟢 GREEN, commit) → 4. refactor (🔵, commit)
```

Never write production code before a failing test. The RED commit is the audit trail.
Git checkpoint commits: `test: RED — <feature>` / `feat: GREEN — <feature>` / `refactor: REFACTOR — <feature>`

---

## Critical Architectural Rules (never change without reading LESSONS.md)

| Rule | Why |
|------|-----|
| `_wrap()` shallow dict: `{f.name: getattr(s,f.name) for f in fields(s)}` | `dataclasses.asdict()` deep-converts nested objects, breaks `.signal` access |
| `errors: Annotated[List[str], operator.add]` — return delta only | Concurrent fan-out causes `InvalidUpdateError` without reducer |
| Routing keys are strings: `"execute"` / `"skip"` not booleans | LangGraph 1.1.x requires string keys |
| `LLM_PROVIDER=none` is the default (heuristic judge) | Zero-config must work without any external service |
| Never `import freqtrade` — HTTP only | GPL-3.0 boundary; Apache-2.0 preserved |
| `LIVE_TRADING=true` + typed "LIVE" confirmation required | Default is always paper; going live is explicit |
| P95 drawdown (not median, not single-path) for capital sizing | P95 runs 1.5–3× higher in Monte Carlo — the actual risk |
| Equity-based circuit breaker (NAV = balance + open P&L) | Balance-based silently ignores open-trade losses |
| `exchange.amount_to_precision()` not `round(qty, n)` | Generic rounding caused 68 consecutive order rejections |
| API keys: trade-only, no-withdraw, IP-whitelisted | Withdraw keys caused $65M+ in documented drain incidents |

---

## Make Targets

```bash
make test              # run 321 tests
make coverage          # 80%+ coverage gate
make backtest          # 30-day walk-forward + Monte Carlo
make paper             # paper trading loop (no real orders)
make dashboard         # start FastAPI backend (WS at localhost:8000)
make snapshot          # write docs/data/snapshot.json (demo, no network)
make snapshot-live     # write real cycle snapshot (needs CCXT access)
make portfolio         # validate + print config/portfolio.yaml
make freqtrade-start   # start optional FreqTrade sidecar
```

---

## Environment Variables

```bash
# Defaults — no .env file needed for dev
LLM_PROVIDER=none           # heuristic judge (zero-key default)
FREQTRADE_MODE=auto         # auto-detect FreqTrade binary
LIVE_TRADING=false          # paper mode (must be 'true' + confirmation for live)
DASHBOARD_PUSH_SECONDS=60   # WebSocket push cadence
EXCHANGE_SANDBOX=true       # Binance testnet

# Upgrade incrementally
LLM_PROVIDER=ollama         # local free LLM (run: ollama pull llama3.1:8b)
LLM_PROVIDER=anthropic      # cloud (needs ANTHROPIC_API_KEY)
```

---

## Open PRs / Active Branches

| Branch | Purpose | Status |
|--------|---------|--------|
| `main` | Stable (Steps 1–8 + dashboard) | ✅ deployed to Pages |
| `2026-06-20-snapshot-council-livebind` | Live-binding, snapshot, portfolio config | 🔄 PR open |

---

## Docs Index (all under `docs/`)

| File | Contents |
|------|----------|
| `LESSONS.md` | 28 engineering lessons from Day 1 (must-read before changing anything) |
| `SPECS.md` | TDD behaviour contracts |
| `HANDOFF.md` | Full context transfer for the agent council |
| `COUNCIL_PROMPT.md` | Council prompt template + output contract |
| `DASHBOARD.md` | Dashboard architecture, demo, remaining work |
| `FREQTRADE_SETUP.md` | Install + security checklist |
| `FUTURE_PLANS_NEXT.md` | Refinement backlog |
| `RESEARCH_NOTES.md` | Research on snapshot workflows, Pages, LangGraph, pydantic |
| `FULL_CONTEXT.md` | ← **Start here if you're new to this project** |
| `PROGRESS.md` | Build log per step |
| `ARCHITECTURE.md` | System design |
| `ZERO_KEY_SETUP.md` | Run with no API keys |
| `LICENSE_RATIONALE.md` | Why Apache 2.0, FreqTrade GPL boundary |

---

## Council Prompt (for filing-driven regime allocation)

```bash
# Demo (shows prompt, no LLM)
python scripts/run_council.py --backend demo

# Ollama (free local)
python scripts/run_council.py --backend ollama --model llama3.1:8b

# Anthropic
python scripts/run_council.py --backend anthropic --model claude-sonnet-4-6

# With custom filing context
python scripts/run_council.py --backend ollama --context my_filing.md
```

The council framework is **regime/factor allocation, not copy-trading.**
Human approval is required before implementing any plan. See `docs/COUNCIL_PROMPT.md`.

---

## What's Next (from `docs/FUTURE_PLANS_NEXT.md`)

1. Wire real Polymarket markets into the live WebSocket path
2. Playwright smoke test on the built dashboard
3. `portfolio_agent` LangGraph node (reads `portfolio.yaml` → sleeve targets)
4. Walk-forward backtest of the regime allocation
5. Correlation-cluster guard (≤3 correlated positions enforced at order time)
6. Real-device responsive QA (sidebar collapse @1100px, single-col @560px)
