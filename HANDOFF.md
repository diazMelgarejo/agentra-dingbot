# HANDOFF.md — Agent Council Transfer Document
*For: antigravity, codex, cursor, ollama, LMStudio via orama-system*

---

## Context in One Paragraph

We are building **Agentra DingBot** — a multi-agent BTC/ETH + Polymarket hybrid trading platform. The intelligence layer uses LangGraph to orchestrate four parallel analyst agents (technical TA, sentiment F&G/VIX, on-chain funding rates, ML LightGBM/sklearn). These feed a debate engine that produces a consensus signal, which passes through a risk manager and dual executor (spot via CCXT, prediction markets via Polymarket CLOB). Steps 1–7 are complete with 307 tests green. Step 8 (React dashboard) is next.

---

## What Is Complete (Steps 1–7)

### Completed modules under `src/`
```
src/core/            config.py, state.py, orchestrator.py, cli.py
src/data/            fetcher.py, polymarket.py, fear_greed.py,
                     websocket_stream.py, snapshot.py
src/agents/          technical_analyst/, sentiment_analyst/, onchain_analyst/,
                     ml_analyst/, debate_engine/, risk_manager/,
                     executor/ (with safety.py), polymarket_agent/
src/ml/              features.py, labels.py, model.py, freqai_bridge.py
src/strategies/      technical_signals.py, fear_filter.py, hybrid_decision.py
src/backtesting/     backtest.py, monte_carlo.py, walk_forward.py,
                     signal_replay.py, polymarket_backtest.py
src/deploy/          live.py, paper_broker.py
src/utils/           logger.py, telegram_alerts.py, orderbook.py
src/dashboard/       app.py (FastAPI + TradingView webhook)
```

### Test suite
- 307 passed, 6 skipped (TA-Lib and LightGBM — correct skips, fallbacks work)
- Zero failures
- Run: `python -m pytest tests/`

---

## What Step 8 Needs (React Dashboard)

### User journey
```
As a trader, I want a live web dashboard that shows my bot's current signal,
confidence, open positions, and candlestick charts, so I can monitor it
without reading log files.
```

### Tech stack
- **Framework**: React + Vite + TypeScript + Tailwind CSS
- **Charts**: [Lightweight Charts™](https://github.com/tradingview/lightweight-charts)
  (Apache-2.0, open-source, TradingView-style candlesticks fed by our CCXT data)
- **Live data**: WebSocket to `/ws/signals` (stub already in `src/dashboard/app.py`)
- **Build tool**: Vite (fast, minimal config)

### Directory structure to create
```
src/dashboard/
├── app.py              ← FastAPI backend (exists, add /ws/signals)
└── frontend/
    ├── package.json
    ├── vite.config.ts
    ├── index.html
    └── src/
        ├── main.tsx
        ├── App.tsx
        └── components/
            ├── SignalCard.tsx    ← current consensus + confidence
            ├── AgentCards.tsx   ← technical/sentiment/onchain/ml snapshots
            ├── CandlestickChart.tsx  ← Lightweight Charts fed by CCXT
            ├── FearGreedGauge.tsx
            └── PositionTable.tsx
```

### WebSocket message shape (from backend)
```json
{
  "type": "cycle_update",
  "timestamp": "2026-01-01T00:00:00+00:00",
  "data": {
    "symbol": "BTC/USDT",
    "debate_consensus": "BUY",
    "debate_confidence": 0.72,
    "technical": { "signal": "BUY", "rsi_14": 38.2, "ema_cross": "BULL" },
    "sentiment": { "signal": "BUY", "fear_greed_index": 28, "vix": 19.5 },
    "onchain": { "signal": "NEUTRAL", "funding_rate": 0.001 },
    "ml": { "signal": "BUY", "prob_up": 0.67, "model_type": "sklearn_hgb" },
    "ohlcv_4h": [{"time": 1704067200, "open": 42000, "high": 43000, "low": 41800, "close": 42800 }]
  }
}
```

### TDD cycle for Step 8

**RED first** — write these tests before any frontend code:
```
tests/test_dashboard.py:
  test_health_endpoint_returns_200
  test_ws_signals_connects_and_receives_heartbeat
  test_state_serialization_handles_enums_and_datetimes
  test_tradingview_webhook_queues_signal
```

**GREEN** — implement minimal code to pass.
**REFACTOR** — clean up.

---

## Critical Rules (Never Violate)

| Rule | Why |
|------|-----|
| Tests before code (TDD) | See SPECS.md + SKILL.md |
| `from core.config import ...` (no `src.` prefix in production code) | pythonpath=src in pytest.ini |
| Shallow dict in `_wrap()` — `{f.name: getattr(s,f.name) for f in fields(s)}` | dataclasses.asdict() deep-converts nested objects, breaks agents |
| `errors: Annotated[List[str], operator.add]` — return delta only (new errors) | Concurrent fan-out writes cause InvalidUpdateError without reducer |
| Never `import freqtrade` | GPL-3.0 boundary — HTTP calls only |
| `LIVE_TRADING=true` requires explicit env + confirmation | Research: most bot disasters are missing safety gates |
| P95 drawdown (not P50, not single-path) as sizing metric | Research: P95 runs 1.5-3x higher in real Monte Carlo |
| Polymarket fee = dynamic `feeRateBps` from CLOB, NOT hardcoded | Fee regime changed Jan-Mar 2026; hardcoding breaks edge calc |
| API keys: trade-only, no-withdraw | $65M+ drained Dec 2024-Jan 2025 via keys with withdraw |

---

## How to Reproduce the Dev Environment

```bash
# 1. Clone
git clone https://github.com/diazMelgarejo/agentra-dingbot.git && cd agentra-dingbot

# 2. Install Python deps
pip install pytest pytest-asyncio structlog pandas numpy pydantic pydantic-settings \
    aiohttp ccxt yfinance pandas-ta langgraph scikit-learn joblib fastapi uvicorn \
    --break-system-packages

# 3. Green suite (must be 307 passed, 6 skipped)
python -m pytest tests/

# 4. Zero-key cycle test
python -m pytest tests/test_freqtrade_optional.py::TestZeroKeyPipeline -v

# 5. Run dashboard locally
LLM_PROVIDER=none python src/dashboard/app.py
# → http://localhost:8000/api/health
```

---

## Useful Commands

```bash
# Run just one test file
python -m pytest tests/test_step6_backtest.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing

# Check which tests cover a module
python -m pytest tests/ --cov=src/agents/executor/safety --cov-report=term-missing

# Lint
ruff check src/ tests/

# Push (needs PAT in URL or SSH key)
git push origin main
```

---

## Open Questions for the Council

1. **Step 8 React framework**: Vite + React or Next.js? (Vite recommended for lighter, no SSR needed)
2. **WebSocket push frequency**: every cycle (~60s default) or on-demand? (currently 60s stub)
3. **FreqUI embed**: iframe embed or separate tab link? (iframe has CSP issues)
4. **Auth for dashboard**: unauthenticated (local only) or JWT? (JWT already in FreqTrade config)
5. **Mobile-responsive**: priority or desktop-first for now?

---

## File Ownership by Agent Capability

| Agent | Best fit for |
|-------|-------------|
| **Cursor** | React/TypeScript frontend (Step 8), IDE-integrated |
| **Claude Code** | Python backend, LangGraph, testing, orchestration |
| **GitHub Codex** | Boilerplate generation, scaffolding |
| **Ollama / LMStudio** | Local inference for the debate engine (LLM_PROVIDER=ollama) |
| **antigravity** | Code review, REFACTOR phase, architecture critique |

---

## Version

`v0.3.0` — Apache 2.0 — Python ≥3.11
GitHub: https://github.com/diazMelgarejo/agentra-dingbot
