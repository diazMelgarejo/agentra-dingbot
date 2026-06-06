# 🤖 agAIntra
## agentic AI SuperBot Meta-Platform v0.3.0

> we are moving to oramasys 2.0 as againtra plugin
> 
> [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
> [![Python](https://img.shields.io/badge/python-3.11+-green.svg)](https://python.org)
> [![Build](https://img.shields.io/badge/build-Steps_1--7_complete-brightgreen)](docs/PROGRESS.md)
> [![Tests](https://img.shields.io/badge/tests-307_passing-brightgreen)](tests/)
> [![Dashboard](https://img.shields.io/badge/dashboard-live-2540ff)](https://diazmelgarejo.github.io/agentra-dingbot/)

---

**Multi-agent hybrid trading platform** — BTC/ETH spot 5-min prediction markets via fractional Kelly.

Merges two battle-tested X bot strategies:
- **Strategy A** (@zostaff): MACD(3,15,3) + RSI(14)/VWAP + CVD order-flow divergence
- **Strategy B** (@shmidtqq): VIX + CNN Fear & Greed + micro-impulse confluence filter

> ⚠️ **Disclaimer:** Educational/research only. Always start in `--paper` mode. Never risk capital you cannot afford to lose.

---

## Project Status — Steps 1–7 Complete ✅

**307 tests passing, 6 skipped** · TDD throughout (RED → GREEN → REFACTOR) · zero-key by default.

| Step | Status | What's Inside |
|------|--------|--------------|
| **Step 1** — Scaffold | ✅ | 8 agents, LangGraph orchestrator, FastAPI dashboard |
| **Step 2** — Data Ingestion | ✅ | CCXT (Binance OHLCV) + Polymarket REST + Fear & Greed/VIX + WebSocket orderbook |
| **Step 3** — TA Agent | ✅ | TA-Lib primary / pandas-ta fallback · MACD/RSI/VWAP/CVD · 5-min fast signals |
| **Step 4** — LangGraph | ✅ | Dual-pipeline orchestrator · shallow-dict state bridge · concurrent fan-out |
| **Step 5** — FreqAI ML | ✅ | LightGBM / sklearn-HGB fallback · 16-feature pipeline · cached retraining |
| **Step 6** — Backtest Validation | ✅ | Walk-forward (purged/embargo) · block-bootstrap Monte Carlo (P5/P50/**P95**/P99) · Brier score · dynamic Polymarket fees |
| **Step 7** — Executor Safety | ✅ | Fail-closed permission check · order validation · KillSwitch · LIVE opt-in gate · NAV-based circuit breaker · CCXT retry matrix |
| **Step 8** — Dashboard | ✅ | Self-contained static dashboard · light/dark themes · live WebSocket + demo fallback · [**live demo →**](https://diazmelgarejo.github.io/agentra-dingbot/) |

### Highlights
- **Backtesting that doesn't lie**: capital is sized against the Monte Carlo **P95** drawdown (which runs 1.5–3× the single-path backtest), not the optimistic backtest figure.
- **Safety-first executor**: refuses to start if API keys carry withdraw permission; paper mode is the default and going live requires both `LIVE_TRADING=true` and a typed confirmation.
- **Zero external dependencies required**: runs with no LLM key (`LLM_PROVIDER=none` heuristic judge), no FreqTrade, no Docker. Each is an optional upgrade.
- **Agent-council ready**: `AGENTS.md`, `docs/HANDOFF.md`, `docs/SPECS.md`, `docs/LESSONS.md` (28 lessons) document every decision for the next contributor.

---

## Architecture

```
ingest_data ─► technical_analyst ─┐
            ─► sentiment_analyst  ├─► debate_engine ─► risk_manager ─► executor (CCXT)
            ─► onchain_analyst    ┘
            ─► polymarket_agent ──────────────────────────────────────► END (CLOB)
```

The two pipelines share all data (OHLCV, Fear & Greed, VIX) but produce independent outputs:
- Spot: `final_signal` + `TradeOrder` (CCXT Binance)
- Polymarket: `PolymarketDecision` (py-clob-client)

---

## Quick Start

### Prerequisites

```bash
# System dependency for TA-Lib (optional — pandas-ta fallback works without it)
brew install ta-lib            # macOS
sudo apt install libta-lib-dev # Ubuntu/Debian
```

### Install

```bash
git clone https://github.com/diazMelgarejo/agentra-dingbot.git
cd agentra-dingbot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env — set POLYMARKET_PRIVATE_KEY + TELEGRAM_BOT_TOKEN at minimum
```

### Backtest (no API keys needed)

```bash
make backtest                                   # 30-day walk-forward + Monte Carlo
PYTHONPATH=src python src/backtesting/backtest.py --days 90 --sims 5000
```

### Paper Trade (safe default)

```bash
make paper                                      # paper mode, no real orders
PYTHONPATH=src LLM_PROVIDER=none python src/deploy/live.py --paper
```

### Run One Cycle (dry-run)

```bash
PYTHONPATH=src LLM_PROVIDER=none python src/core/cli.py run --symbol BTC/USDT
```

### Tests

```bash
make test                                       # all 307 tests
make coverage                                   # tests + 80% coverage gate
python -m pytest tests/test_step6_backtest.py -v   # Step 6 backtest suite
ruff check src tests && mypy src                # lint + type-check
```

### Dashboard

```bash
# Option 1 — instant static demo (no backend, no build)
open docs/index.html

# Option 2 — live data: start backend, then open the dashboard
LLM_PROVIDER=none python src/dashboard/app.py     # WebSocket at /ws/signals
python -m http.server 8080 --directory docs       # → http://localhost:8080
```

The dashboard auto-connects to `ws://localhost:8000/ws/signals`; if no backend is
running it falls back to a live-updating demo. Public demo:
**https://diazmelgarejo.github.io/agentra-dingbot/** · details in [docs/DASHBOARD.md](docs/DASHBOARD.md).

---

## Data Sources (Step 2)

| Source | Provider | Auth | Used For |
|--------|----------|------|----------|
| BTC/ETH OHLCV | Binance via CCXT | None (public) | Technical signals |
| Polymarket markets | Gamma API | None (public) | Market discovery |
| Polymarket prices | CLOB REST API | None (public) | YES token pricing |
| Polymarket L2 book | CLOB WebSocket | None (public) | Spread/farming |
| Fear & Greed Index | alternative.me | None (free) | Sentiment regime |
| VIX | yfinance ^VIX | None (free) | Volatility filter |

---

## Configuration

Key `.env` variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `EXCHANGE_SANDBOX` | `true` | **Always start on testnet** |
| `PAPER_MODE` | `true` | **Always start in paper mode** |
| `BANKROLL_USDC` | `100.0` | Polymarket trading capital |
| `KELLY_FRACTION` | `0.25` | Fractional Kelly (25% of optimal) |
| `MIN_EDGE_PCT` | `8.0` | Min Bayesian edge to trade |
| `DAILY_DRAWDOWN_LIMIT_PCT` | `5.0` | Circuit breaker threshold |
| `LLM_PROVIDER` | `none` | `none` (heuristic, zero-key) · `ollama` (local) · `openai` |
| `LIVE_TRADING` | `false` | Must be `true` **and** typed-confirmed to place real orders |

---

## License

[Apache 2.0](LICENSE)