# Agentic SuperBot v0.3.0 — Architecture

## Overview

SuperBot merges two trading strategies into one unified agentic platform:

- **Strategy A** (`strategies/technical_signals.py`) — MACD(3,15,3) + RSI(14)/VWAP + CVD for 5m directional signals
- **Strategy B** (`strategies/fear_filter.py`) — CNN Fear & Greed + VIX + micro-impulse as a confluence filter

They run through a shared risk engine and execute on two surfaces simultaneously:
1. **Spot BTC/ETH** via CCXT + Binance (LangGraph pipeline)
2. **Polymarket 5-min BTC Up/Down** via py-clob-client CLOB (independent pipeline)

---

## Data Ingestion Layer (Step 2)

```
fetch_full_snapshot("BTC/USDT")
       │
       ├── CCXT Binance (async)  ─────────► {"5m": df, "1h": df, "4h": df, "1d": df}
       │   data/fetcher.py
       │   asynccontextmanager pattern — zero connection leaks
       │
       ├── Polymarket REST (async) ────────► {markets, enriched_markets, farmable_markets}
       │   data/polymarket.py
       │   Gamma API (market discovery) + CLOB API (prices, spreads)
       │
       └── Sentiment (concurrent) ─────────► {fear_greed, vix, vix_risk_level, size_multiplier}
           data/fear_greed.py
           alternative.me (free) + yfinance ^VIX (free)
```

WebSocket orderbook (`data/websocket_stream.py`) runs as a separate background task
for live L2 data and liquidity farming spread detection.

---

## Dual Pipeline Architecture

```
                    ┌──────────────────────┐
                    │     ingest_data      │ ← fetch_full_snapshot() [all 4 sources]
                    └──────┬───────────────┘
                           │
          ┌────────────────┼────────────────────┐
          │                │                    │
          ▼                ▼                    ▼
  technical_analyst  sentiment_analyst   onchain_analyst
  (4h + 5m MACD/     (F&G + VIX +       (funding rate,
   RSI/CVD/VWAP)      micro impulse)     open interest)
          │                │                    │
          └────────────────┼────────────────────┘
                           │
                           ▼
                     debate_engine         ← LLM Bull vs Bear
                           │
                           ▼
                     risk_manager          ← ATR stops + VIX circuit breaker
                           │
                    approved?
                    /        \
                  YES         NO
                   │           │
                   ▼           ▼
               executor       END       ← CCXT spot order (dry_run by default)

  ingest_data ──► polymarket_agent ──► END
                  (hybrid decision:
                   tech × fear → Bayesian edge → fractional Kelly)
```

---

## Module Map

```
superbot/
├── core/
│   ├── config.py         — pydantic-settings singleton (ExchangeConfig + PolymarketConfig)
│   ├── state.py          — TradingState + PolymarketDecision dataclasses
│   ├── orchestrator.py   — LangGraph dual-pipeline graph builder
│   └── cli.py            — CLI: run | dashboard | backtest
│
├── data/                 ★ STEP 2 — DATA INGESTION LAYER
│   ├── fetcher.py        — CCXT async OHLCV (asynccontextmanager, no leaks)
│   ├── polymarket.py     — Gamma + CLOB REST (market discovery, prices, spreads, farming)
│   ├── fear_greed.py     — CNN Fear & Greed (alternative.me) + VIX (yfinance)
│   ├── websocket_stream.py — L2 orderbook via Polymarket WebSocket (auto-reconnect)
│   ├── snapshot.py       — Unified 4-source concurrent snapshot (one call)
│   └── __init__.py       — Public API exports
│
├── agents/
│   ├── technical_analyst/ — Standard 4h TA + fast 5m MACD(3,15,3)/VWAP/CVD signals
│   ├── sentiment_analyst/ — F&G + VIX + micro impulse → Signal
│   ├── onchain_analyst/   — Funding rates, open interest
│   ├── debate_engine/     — LLM Bull vs Bear → consensus (ollama or openai)
│   ├── risk_manager/      — ATR stops + VIX circuit breaker + max loss cap
│   ├── executor/          — Spot CCXT + Polymarket CLOB execution
│   └── polymarket_agent/  — Hybrid decision bridge (tech × fear → Kelly)
│
├── strategies/            ← From Polymarket SuperBot
│   ├── technical_signals.py  — MACD(3,15,3) + RSI/VWAP + CVD
│   ├── fear_filter.py        — F&G + VIX regime + micro impulse + confluence gate
│   └── hybrid_decision.py    — Bayesian edge (≥8%) + fractional Kelly sizing
│
├── backtesting/
│   ├── backtest.py        — Walk-forward on real Binance 5m OHLCV
│   └── monte_carlo.py     — 5,000 simulation P&L distribution
│
├── deploy/
│   ├── live.py            — Async event loop: HybridTrader + midnight reset
│   └── paper_broker.py    — Paper trading simulator
│
├── utils/
│   ├── logger.py          — SQLite trade logging + daily P&L
│   ├── telegram_alerts.py — Async Telegram notifications
│   └── orderbook.py       — LocalOrderbook (legacy REST version)
│
├── dashboard/
│   └── app.py             — FastAPI: /api/signals + /api/run + /api/health
│
├── config/
│   └── strategies.yaml    — Risk rules + indicator parameters
│
└── tests/
    ├── test_data_ingestion.py   — 26 tests: CCXT, F&G, VIX, Polymarket, snapshot
    ├── test_technical_analyst.py
    └── test_risk_manager.py
```

---

## Risk Parameters

### Spot Trading (Mandell Backtest — 14 months)
| Signal | Position | SL Mult | RR |
|--------|----------|---------|-----|
| STRONG_BUY | 25% | 1.5× ATR | 3.0 |
| BUY | 15% | 2.0× ATR | 2.5 |

### Polymarket (Fractional Kelly)
| Parameter | Value |
|-----------|-------|
| Min Bayesian edge | 8% |
| Kelly fraction | 0.25× full Kelly |
| Max concurrent | 3 |
| Daily drawdown stop | −5% |
| VIX elevated (≥30) | 50% size reduction |
| VIX extreme (≥40) | Circuit breaker — no trades |

---

## Confluence Logic (Hybrid Decision)

```
generate_technical_signal(df_5m)     → direction + confidence (0–1)
         │
         └── if NEUTRAL → skip
         
generate_fear_signal(fg, vix, df_5m) → regime + size_multiplier
         │
         └── if VIX EXTREME → skip
         
fear_confirms_direction(fear, tech)   → (confirmed, boost ×1.0–1.5)
         │
         └── if not confirmed → skip
         
bayesian_update(market_price, LR)    → posterior_prob
         │
edge = |posterior - market_price| × 100
         │
         └── if edge < 8% → skip
         
fractional_kelly(posterior, price)   × size_multiplier × 0.25
         │
         └── position_usdc = bankroll × kelly_fraction
```

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| ✅ 1 | Done | Scaffold, config, state, 6 agents |
| ✅ 2 | Done | Data ingestion: CCXT + Polymarket + F&G + VIX + WebSocket |
| ⬜ 3 | Next | TA Agent: EMA/BB/ATR + MACD(3,15,3) validation tests |
| ⬜ 4 | Next | LangGraph integration test end-to-end |
| ⬜ 5 | Next | FreqAI ML bridge |
| ⬜ 6 | Next | Risk manager tuning + backtest validation |
| ⬜ 7 | Next | Executor dry-run test suite |
| ⬜ 8 | Next | Dashboard React UI + WebSocket push |
