# Agentic SuperBot — Future Plans

## Immediate Roadmap (Steps 6–8)

---

### Step 6 — Risk Manager Tuning + Backtest Validation ⬜

**Goal**: Validate the risk rules against real historical data and tighten the parameters.

**What to build**:
- `backtesting/backtest.py`: extend to run the full LangGraph pipeline on historical OHLCV slices (walk-forward, not look-ahead). Currently uses simplified signal replay; upgrade to run real agents.
- Backtest against real Binance 5m OHLCV (30-day minimum, 90-day preferred).
- Output: Sharpe ratio, win rate, profit factor, max drawdown, equity curve PNG.
- `backtesting/monte_carlo.py`: 5,000 P&L shuffle simulations on backtest results.
- `agents/risk_manager/agent.py`: tune `_RISK_RULES` table thresholds based on backtest results. Currently using Mandell signal rules as defaults.

**Key design question**: Should the Polymarket Kelly position sizing (8% edge threshold, 0.25× fraction) be validated separately against Polymarket historical resolution data?  
**Recommendation**: Yes — pull a 90-day window of resolved BTC Up/Down markets from Gamma API and backtest the hybrid decision model against actual outcomes.

**Tests to add** (`tests/test_backtest.py`):
- Walk-forward slice produces consistent metrics across overlapping windows.
- Position sizing never exceeds max allowed capital per trade.
- Drawdown circuit breaker fires at correct threshold.
- Monte Carlo: 5th/95th percentile bands of P&L distribution.

---

### Step 7 — Executor Dry-Run Test Suite ⬜

**Goal**: Full coverage of the execution layer before any live money touches it.

**What to build**:
- Expand `tests/test_executor.py` to cover all order types, failure modes, and edge cases.
- Mock CCXT order placement (`create_order`) and verify the right parameters are passed (side, price, amount, type="limit").
- Mock Polymarket CLOB order submission (`py-clob-client`) and verify Kelly-sized amounts.
- Slippage simulation: test that dry-run prices include the configured slippage.
- Partial fills: test that the position tracker handles partially-filled limit orders.

**Dry-run gate**: The `--live` CLI flag should be hard to set accidentally. Add a confirmation prompt: "You are enabling LIVE trading with real funds. Type 'LIVE' to confirm:".

**Tests to add** (expand `tests/test_langgraph_pipeline.py::TestExecutorNode`):
- `test_limit_order_uses_current_price` — price from technical snapshot, not market order.
- `test_dry_run_never_calls_exchange` — CCXT mock never called when dry_run=True.
- `test_polymarket_kelly_amount_within_bounds` — position never exceeds bankroll × kelly_fraction.
- `test_stop_loss_above_entry_for_shorts` — executor SL/TP geometry for sell side.

---

### Step 8 — React Dashboard + WebSocket Push ⬜

**Goal**: Live monitoring UI showing agent signals, debate consensus, and open positions.

**What to build**:
- `dashboard/app.py`: upgrade FastAPI to serve WebSocket connections at `/ws/signals`.
- Background task: runs `run_one_cycle()` on a configurable interval and pushes results to all connected clients.
- `dashboard/frontend/`: React app (Vite + TypeScript + Tailwind).

**Dashboard panels**:

| Panel | Data Source | Update Rate |
|-------|-------------|-------------|
| Signal Consensus | debate_engine output | Per cycle |
| Agent Cards | technical/sentiment/onchain/ml snapshots | Per cycle |
| Probability Gauge | ml_analyst prob_up | Per cycle |
| Fear & Greed + VIX | sentiment_raw | Per cycle |
| Open Positions | executor order log | Real-time |
| Equity Curve | SQLite trade log | Per cycle |
| Polymarket Markets | polymarket_snapshot | Per cycle |
| Liquidity Farming | farmable_markets | Per cycle |

**Charts**: Use [Lightweight Charts™](https://github.com/tradingview/lightweight-charts) (open-source, Apache-2.0, TradingView-style candlestick charts) fed by the bot's own CCXT data — **not** TradingView's hosted service. See `docs/TRADINGVIEW_INTEGRATION.md` for details.

**Tests to add** (`tests/test_dashboard.py`):
- FastAPI health endpoint returns 200.
- WebSocket connection establishes and receives first cycle update.
- State serialization: TradingState → JSON for WebSocket (needs custom serializer for Enums and datetimes).

---

## Medium-Term Enhancements

### TradingView Webhook Receiver
Add a `/webhook/tradingview` endpoint to `dashboard/app.py` so TradingView alerts (Pine Script) can inject directional signals into the pipeline. See `docs/TRADINGVIEW_INTEGRATION.md` for the complete implementation plan.

### LLaVA Visual Agent (Fenix-style)
From the architecture doc: a vision LLM analyzing TradingView chart screenshots for candlestick pattern recognition.

```python
# Future: agents/visual_analyst/agent.py
# Screenshot TradingView chart → encode as base64 → Claude or LLaVA vision model
# Returns: detected patterns, support/resistance levels, trend lines
```

This becomes the 5th parallel analyst feeding the debate engine. The Polymarket visual context (screenshot of the 5-min BTC Up/Down market price chart) is a strong signal source that no pure-quantitative agent can replicate.

### FreqAI Enhancements
- **LightGBM hyperparameter tuning**: Optuna study on backtested data.
- **Feature importance drift detection**: alert when the top-5 features change significantly between retrains (regime shift indicator).
- **Multi-class labels**: extend from binary up/down to three-class up/flat/down for the debate engine.
- **Lookback adaptation**: shorten the training window during high-volatility regimes (detected via VIX).

### On-Chain Data Expansion
Currently limited to funding rates. Add:
- MVRV ratio (Glassnode or CryptoQuant API)
- Exchange net flows (large deposits → bearish)
- Open interest (CoinGlass API — free tier available)
- Liquidation heatmap snapshots

### Multi-Symbol Support
The architecture is designed for this: add `ETH/USDT` as a second symbol running its own LangGraph cycle. The `ml_analyst` already keys models by symbol. The main change: a cycle scheduler in `deploy/live.py` that interleaves BTC and ETH cycles with rate-limit awareness.

### PostgreSQL + TimescaleDB
Replace SQLite trade logging with PostgreSQL + TimescaleDB for:
- Efficient time-series queries on trade history.
- Granular per-agent signal logging for post-trade analysis.
- Grafana integration for historical equity curves.

### Grafana Monitoring
- Agent confidence over time (is the ML model drifting?).
- Fear & Greed + VIX correlation with trade outcomes.
- Polymarket P&L vs edge distribution.

---

## Architecture Decisions Still Open

| Decision | Options | Recommendation |
|----------|---------|----------------|
| LLM provider for debate | Ollama (local, free) vs OpenAI (API cost) | Ollama (llama3.1:8b) for development; OpenAI gpt-4o for production if budget allows |
| Polymarket execution | py-clob-client (current) vs custom CLOB client | Keep py-clob-client until it breaks |
| ML refresh cadence | Every N cycles (current) vs time-based vs accuracy-triggered | Add accuracy monitoring in Step 6; keep cycle-based for now |
| Dashboard deploy | Local only vs VPS | Start local; add nginx + domain in Step 8 |
| Secrets management | .env file (current) vs HashiCorp Vault vs AWS SSM | .env is fine for personal use; add Vault if deploying to VPS for multiple users |

---

## Risk Warnings (Permanent)

1. **Always start in paper mode** (`PAPER_MODE=true`, `EXCHANGE_SANDBOX=true`).
2. **Backtest at least 30 days before live** — the strategies have not been validated in all market regimes.
3. **The ML model will overfit on short histories** — use `ML_LOOKBACK_BARS=500` minimum.
4. **Polymarket is zero-sum** — 14 of the top 20 traders are bots (March 2026 data). Speed and structure matter more than prediction accuracy alone.
5. **The daily drawdown circuit breaker** (`DAILY_DRAWDOWN_LIMIT_PCT=5.0`) is a last resort, not a strategy. If it fires regularly, the edge doesn't exist.
6. **VIX ≥ 40** blocks all trading. Don't override this.
