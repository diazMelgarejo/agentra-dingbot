# AGENTS.md — Agentra DingBot
> Primary handoff document for Claude Code, Cursor, GitHub Copilot, Codex, Ollama, LMStudio.
> Read this file first. Everything else is downstream of the decisions here.

---

## What This Project Is

A multi-agent algorithmic trading platform for **BTC/ETH spot** (via Binance/CCXT) and
**Polymarket 5-minute BTC Up/Down prediction markets** simultaneously. The intelligence
layer is a LangGraph orchestration of four parallel analyst agents, a deterministic
heuristic debate engine (zero keys required), a risk manager, and a dual executor.

**Key design principle**: every external dependency is optional.
- No LLM key required (heuristic judge runs by default, `LLM_PROVIDER=none`)
- No FreqTrade required (auto-detected sidecar, `FREQTRADE_MODE=auto`)
- No Docker required (runs in a plain Python venv)
- No Polymarket key for research/backtesting
- Default mode: dry-run paper trading only

---

## Repository Layout

```
agentra-dingbot/
├── src/                    ← ALL Python source code lives here
│   ├── core/               ← Config (pydantic-settings), state dataclasses, orchestrator
│   ├── data/               ← Async data ingestion (CCXT, Polymarket, Fear&Greed, VIX)
│   ├── agents/             ← 8 LangGraph agent nodes
│   │   ├── technical_analyst/    ← TA-Lib/pandas indicators (4h + 5m)
│   │   ├── sentiment_analyst/    ← F&G + VIX + micro-impulse
│   │   ├── onchain_analyst/      ← Funding rates (Binance futures public)
│   │   ├── ml_analyst/           ← FreqAI-style LightGBM/sklearn signal
│   │   ├── debate_engine/        ← Heuristic vote OR LLM bull/bear/judge
│   │   ├── risk_manager/         ← ATR stops, VIX circuit breakers, Kelly
│   │   ├── executor/             ← CCXT spot + Polymarket CLOB + safety layer
│   │   └── polymarket_agent/     ← Hybrid decision + fractional Kelly sizing
│   ├── ml/                 ← Feature engineering, labels, model, FreqAI bridge
│   ├── strategies/         ← @zostaff + @shmidtqq strategy implementations
│   ├── backtesting/        ← Walk-forward, Monte Carlo (P95), signal replay
│   ├── deploy/             ← Async live/paper trading loop
│   ├── utils/              ← SQLite logger, Telegram alerts
│   └── dashboard/          ← FastAPI + TradingView webhook receiver
├── tests/                  ← pytest test suite (307 passing, 6 skipped)
├── docs/                   ← All documentation (see below)
├── config/                 ← strategies.yaml
├── SPECS.md                ← Behaviour contracts (TDD spec, read before coding)
├── AGENTS.md               ← This file
├── pytest.ini              ← pythonpath = src (critical for imports)
└── requirements.txt
```

---

## How to Resume (Any Agent)

```bash
# Clone
git clone https://github.com/diazMelgarejo/agentra-dingbot.git
cd agentra-dingbot

# Install
pip install pytest pytest-asyncio structlog pandas numpy pydantic pydantic-settings \
    aiohttp ccxt yfinance pandas-ta langgraph scikit-learn joblib fastapi uvicorn \
    --break-system-packages

# Verify (must be 307 passed, 6 skipped)
python -m pytest tests/

# Run one cycle (zero keys, dry-run)
LLM_PROVIDER=none FREQTRADE_MODE=off python -c "
import asyncio
from src.core.orchestrator import run_one_cycle
asyncio.run(run_one_cycle('BTC/USDT', dry_run=True))
"
```

---

## Import Convention

**All imports use the package name without `src.` prefix.**
pytest.ini adds `src/` to PYTHONPATH, so:

```python
# CORRECT
from core.config import get_settings
from agents.technical_analyst.agent import run

# WRONG (don't use src. prefix in production code)
from src.core.config import get_settings
```

Tests may use either form — both work. Production code uses the non-prefixed form.

---

## TDD Workflow (Required for New Code)

This project follows TDD strictly per `tdd.md` and `SKILL.md`:

```
1. Write SPECS.md entry for the feature
2. Write failing tests (🔴 RED) — commit: "test: RED — <feature>"
3. Implement minimal code (🟢 GREEN) — commit: "feat: GREEN — <feature>"
4. Refactor (🔵 REFACTOR) — commit: "refactor: REFACTOR — <feature>"
```

**Never write production code before a failing test.**
**Never skip the RED phase** — tests must be verified failing before implementation.

---

## Build Status

| Step | Module | Tests | Status |
|------|--------|-------|--------|
| 1 | Project scaffold | — | ✅ Done |
| 2 | Data ingestion | 26 | ✅ Done |
| 3 | TA Agent | 74 | ✅ Done (5 skip: no system TA-Lib) |
| 4 | LangGraph orchestrator | 76 | ✅ Done |
| 5 | FreqAI ML bridge | 42 | ✅ Done (1 skip: no LightGBM) |
| 6 | Backtest validation | 61 | ✅ Done |
| 7 | Executor safety | 28 | ✅ Done |
| 8 | Static dashboard + WS token + snapshot + UI tests | 27 | ✅ Done |

---

## Next Step: Step 8 — React Dashboard

See `docs/FUTURE_PLANS.md` for full spec. Key deliverables:
- `dashboard/frontend/` — React + Vite + TypeScript + Tailwind
- `Lightweight Charts™` (Apache-2.0) fed by CCXT data (NOT TradingView)
- `/ws/signals` WebSocket pushing cycle state
- Buy/sell signal markers overlaid on candlestick chart
- FreqUI embed link (if FreqTrade running)

---

## Key Architectural Decisions (Don't Undo These)

1. **`_wrap()` shallow dict** — `{f.name: getattr(state,f.name) for f in fields(state)}`
   NOT `dataclasses.asdict()`. Deep conversion breaks nested dataclass objects.

2. **`errors` reducer** — `Annotated[List[str], operator.add]`
   Concurrent fan-out nodes all write `errors`; reducer merges them.

3. **Routing string keys** — `"execute"` / `"skip"` (not bool) in conditional edges.

4. **LLM default `none`** — heuristic judge is the zero-config path.

5. **FreqTrade HTTP only** — never `import freqtrade` (GPL-3.0). HTTP boundary preserves Apache-2.0.

6. **`LIVE_TRADING=true` gate** — default is always paper. Explicit opt-in required.

7. **P95 drawdown gate** — Monte Carlo P95, not single-path backtest max-DD.

---

## Docs Index

| File | Contents |
|------|----------|
| `docs/PROGRESS.md` | Build log, per-step decisions and bugs |
| `docs/FUTURE_PLANS.md` | Steps 8+ roadmap, enhancement ideas |
| `docs/RESEARCH_STEP6_STEP7.md` | Research findings driving Step 6/7 |
| `docs/STEP6_FREQTRADE_PLAN.md` | FreqTrade sidecar architecture |
| `docs/TRADINGVIEW_WEBHOOK.md` | TradingView webhook implementation |
| `docs/TRADINGVIEW_INTEGRATION.md` | Full TV integration guide |
| `docs/ZERO_KEY_SETUP.md` | Run with no API keys |
| `docs/LICENSE_RATIONALE.md` | Why Apache 2.0 + GPL boundary analysis |
| `docs/SPECS.md` | TDD behaviour contracts for all modules |

---

## Environment Variables

```bash
# Zero-config defaults (no .env needed for dev)
LLM_PROVIDER=none              # heuristic judge (no key)
FREQTRADE_MODE=auto            # detect if installed
PAPER_MODE=true                # never real money
EXCHANGE_SANDBOX=true          # testnet

# Unlock features incrementally
LLM_PROVIDER=ollama            # free local LLM (run: ollama pull llama3.1:8b)
LLM_PROVIDER=openai            # paid cloud LLM (needs OPENAI_API_KEY)
LIVE_TRADING=true              # explicit live opt-in (after thorough testing)
```
