# 🤖 Agentic SuperBot v0.3.0

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-green.svg)](https://python.org)
[![Step](https://img.shields.io/badge/build-Step_2_Complete-brightgreen)](docs/ARCHITECTURE.md)

**Multi-agent hybrid trading platform** — BTC/ETH spot via LangGraph + Polymarket 5-min prediction markets via fractional Kelly.

Merges two battle-tested X bot strategies:
- **Strategy A** (@zostaff): MACD(3,15,3) + RSI(14)/VWAP + CVD order-flow divergence
- **Strategy B** (@shmidtqq): VIX + CNN Fear & Greed + micro-impulse confluence filter

> ⚠️ **Disclaimer:** Educational/research only. Always start in `--paper` mode. Never risk capital you cannot afford to lose.

---

## What's Built (Steps 1 + 2)

| Step | Status | What's Inside |
|------|--------|--------------|
| **Step 1** — Scaffold | ✅ | All 6 agents, LangGraph orchestrator, FastAPI dashboard |
| **Step 2** — Data Ingestion | ✅ | CCXT (Binance OHLCV) + Polymarket REST + Fear & Greed/VIX + WebSocket orderbook |
| **Step 3** — TA Agent | 🔜 | TA-Lib validation + indicator tests |
| **Step 4** — LangGraph | 🔜 | Integration test end-to-end |
| **Step 5** — FreqAI | 🔜 | ML signal bridge |
| **Step 6** — Risk tuning | 🔜 | Backtest validation |
| **Step 7** — Executor | 🔜 | Dry-run test suite |
| **Step 8** — Dashboard | 🔜 | React + WebSocket UI |

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
git clone https://github.com/yourusername/agentic-superbot.git
cd agentic-superbot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env — set POLYMARKET_PRIVATE_KEY + TELEGRAM_BOT_TOKEN at minimum
```

### Backtest (no API keys needed)

```bash
python backtesting/backtest.py --days 30
python backtesting/monte_carlo.py --simulations 5000
```

### Paper Trade (Polymarket)

```bash
python deploy/live.py --paper
```

### Run Spot Cycle (dry-run)

```bash
superbot run --symbol BTC/USDT
```

### Tests

```bash
pytest                          # all tests
pytest tests/test_data_ingestion.py -v   # Step 2 data layer only
```

### Dashboard

```bash
superbot dashboard --port 8000
# Open http://localhost:8000/docs
```

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
| `LLM_PROVIDER` | `ollama` | `ollama` (local) or `openai` |

---

## License

[Apache 2.0](LICENSE)
