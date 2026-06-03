# Step 6: Risk Manager Tuning + Backtest Validation
## Revised Plan — FreqTrade as Execution Backend

**Lesson from Steps 1–5**: We built the full intelligence stack ourselves (good).
We also re-built parts of the execution stack that FreqTrade already does well
(order management, position tracking, live OHLCV, backtesting). Step 6 course-
corrects by wiring FreqTrade as the **execution sidecar** while keeping the
SuperBot's brain (LangGraph + ML + debate) doing what it does best.

---

## Architecture After Step 6

```
┌─────────────────────────────────────────────────────────┐
│                  Agentic SuperBot                        │
│                                                         │
│  data/fetcher.py ──► LangGraph Pipeline                 │
│  (CCXT / Polymarket)    ↓                               │
│                    technical_analyst                     │
│                    sentiment_analyst  ──► debate_engine  │
│                    onchain_analyst        ↓             │
│                    ml_analyst         risk_manager       │
│                                           ↓             │
│                    ┌──────────────────────┘             │
│                    │  FreqTrade REST Client              │
│                    └──────────────────────┐             │
└───────────────────────────────────────────┼─────────────┘
                                            │  HTTP REST
                           ┌────────────────▼─────────────┐
                           │   FreqTrade Container         │
                           │   (GPL-3.0, separate service) │
                           │                               │
                           │  • CCXT order execution       │
                           │  • Position management        │
                           │  • Stop-loss/take-profit      │
                           │  • Trade history / DB         │
                           │  • FreqAI (optional)          │
                           │  • Backtesting engine         │
                           └───────────────────────────────┘
```

The SuperBot generates the intelligence. FreqTrade handles the plumbing. Clean
separation. Apache 2.0 preserved (no FreqTrade source imported).

---

## What FreqTrade Gives Us for Free

| FreqTrade Capability | What It Replaces in Our Code |
|---------------------|------------------------------|
| Order placement + retry logic | `agents/executor/agent.py` CCXT calls |
| Position tracking + open trades | Manual state management |
| Stop-loss / take-profit automation | `_calc_sl` / `_calc_tp` in executor |
| Trade history + P&L | `utils/logger.py` SQLite implementation |
| Walk-forward backtesting | `backtesting/backtest.py` (our simpler version) |
| FreqAI with LightGBM/XGBoost | Supplements our `ml/` package |
| Telegram notifications | `utils/telegram_alerts.py` |
| Hyperopt parameter tuning | Not yet built |
| Live dashboard (FreqUI) | Supplements our React dashboard (Step 8) |

---

## Step 6 Build Plan

### 6a. FreqTrade Docker Setup

**File: `deploy/freqtrade/docker-compose.override.yml`**

```yaml
version: "3.9"
services:
  freqtrade:
    image: freqtradeorg/freqtrade:stable
    restart: unless-stopped
    volumes:
      - ./user_data:/freqtrade/user_data
    ports:
      - "8080:8080"
    command: >
      trade
      --config /freqtrade/user_data/config.json
      --strategy SuperBotFollower
      --logfile /freqtrade/user_data/logs/freqtrade.log
```

**File: `deploy/freqtrade/user_data/config.json`**

```json
{
  "max_open_trades": 3,
  "stake_currency": "USDT",
  "stake_amount": "unlimited",
  "tradable_balance_ratio": 0.99,
  "fiat_display_currency": "USD",
  "dry_run": true,
  "dry_run_wallet": 1000,
  "cancel_open_orders_on_exit": false,
  "trading_mode": "spot",
  "margin_mode": "",
  "unfilledtimeout": {
    "entry": 10,
    "exit": 10,
    "exit_timeout_count": 0,
    "unit": "minutes"
  },
  "entry_pricing": {
    "price_side": "same",
    "use_order_book": true,
    "order_book_top": 1,
    "price_last_balance": 0.0,
    "check_depth_of_market": {"enabled": false, "bids_to_ask_delta": 1}
  },
  "exit_pricing": {
    "price_side": "same",
    "use_order_book": true,
    "order_book_top": 1
  },
  "exchange": {
    "name": "binance",
    "key": "",
    "secret": "",
    "ccxt_config": {},
    "ccxt_async_config": {},
    "pair_whitelist": ["BTC/USDT", "ETH/USDT"],
    "pair_blacklist": []
  },
  "pairlists": [
    {"method": "StaticPairList"}
  ],
  "telegram": {
    "enabled": false,
    "token": "",
    "chat_id": ""
  },
  "api_server": {
    "enabled": true,
    "listen_ip_address": "0.0.0.0",
    "listen_port": 8080,
    "verbosity": "error",
    "enable_openapi": false,
    "jwt_secret_key": "change_me_in_production",
    "CORS_origins": ["http://localhost:3000"],
    "username": "superbot",
    "password": "superbot_password"
  },
  "bot_name": "superbot-freqtrade",
  "initial_state": "running",
  "force_entry_enable": true,
  "internals": {"process_throttle_secs": 5}
}
```

**File: `deploy/freqtrade/user_data/strategies/SuperBotFollower.py`**

```python
"""
SuperBotFollower strategy — a thin FreqTrade strategy that does nothing on its
own. All entry/exit decisions come from the SuperBot via force_entry / force_exit
REST calls. FreqTrade handles execution, position management, and stop-losses.
"""
from freqtrade.strategy import IStrategy
import pandas as pd

class SuperBotFollower(IStrategy):
    # Minimal config — the SuperBot controls entry/exit via REST
    INTERFACE_VERSION = 3
    can_short = False
    minimal_roi = {"0": 0.10}  # 10% safety net — SuperBot manages exits normally
    stoploss = -0.05            # FreqTrade hard stop at -5% as backstop
    trailing_stop = False
    timeframe = "5m"

    def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return df  # SuperBot does all TA

    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df["enter_long"] = 0   # SuperBot uses force_entry, not trend signals
        return df

    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df["exit_long"] = 0    # SuperBot uses force_exit
        return df
```

---

### 6b. FreqTrade REST Client in SuperBot

**New file: `agents/executor/freqtrade_client.py`**

```python
"""
FreqTrade REST API client. Wraps the bot's API with async calls.
Reference: https://www.freqtrade.io/en/stable/rest-api/
"""
from __future__ import annotations
import asyncio
from typing import Any, Dict, List, Optional
import aiohttp
import structlog

logger = structlog.get_logger(__name__)

class FreqTradeClient:
    """Async client for the FreqTrade REST API."""

    def __init__(self, base_url: str = "http://localhost:8080",
                 username: str = "superbot",
                 password: str = "superbot_password"):
        self.base = base_url.rstrip("/")
        self._auth = aiohttp.BasicAuth(username, password)

    async def _get(self, path: str) -> Dict[str, Any]:
        async with aiohttp.ClientSession(auth=self._auth) as s:
            async with s.get(f"{self.base}/api/v1{path}",
                             timeout=aiohttp.ClientTimeout(total=10)) as r:
                r.raise_for_status()
                return await r.json()

    async def _post(self, path: str, data: dict = None) -> Dict[str, Any]:
        async with aiohttp.ClientSession(auth=self._auth) as s:
            async with s.post(f"{self.base}/api/v1{path}",
                              json=data or {},
                              timeout=aiohttp.ClientTimeout(total=10)) as r:
                r.raise_for_status()
                return await r.json()

    async def ping(self) -> bool:
        """Check if FreqTrade is reachable."""
        try:
            r = await self._get("/ping")
            return r.get("status") == "pong"
        except Exception:
            return False

    async def status(self) -> List[Dict[str, Any]]:
        """Get all open trades."""
        return await self._get("/status")

    async def performance(self) -> List[Dict[str, Any]]:
        """Get trade performance stats."""
        return await self._get("/performance")

    async def profit(self) -> Dict[str, Any]:
        """Get overall profit summary."""
        return await self._get("/profit")

    async def force_entry(self, pair: str, side: str = "long",
                          stake_amount: Optional[float] = None,
                          price: Optional[float] = None,
                          stoploss: Optional[float] = None) -> Dict[str, Any]:
        """
        Force an entry trade.
        This is the primary call from the executor when risk_manager approves.
        """
        payload: Dict[str, Any] = {"pair": pair, "side": side}
        if stake_amount:
            payload["stakeamount"] = stake_amount
        if price:
            payload["price"] = price
        if stoploss:
            payload["stoploss"] = stoploss
        logger.info("freqtrade_force_entry", pair=pair, side=side,
                    stake=stake_amount, sl=stoploss)
        return await self._post("/forcebuy", payload)

    async def force_exit(self, trade_id: int,
                         ordertype: str = "limit") -> Dict[str, Any]:
        """Force-exit an open trade."""
        logger.info("freqtrade_force_exit", trade_id=trade_id)
        return await self._post("/forcesell",
                                {"tradeid": str(trade_id), "ordertype": ordertype})

    async def count(self) -> Dict[str, Any]:
        """Get count of open/allowed trades."""
        return await self._get("/count")

    async def locks(self) -> List[Dict[str, Any]]:
        """Get pair locks (prevent re-entry)."""
        return await self._get("/locks")
```

---

### 6c. Upgrade executor/agent.py to Use FreqTrade When Available

```python
# In agents/executor/agent.py — updated routing
async def run(state: Dict[str, Any]) -> Dict[str, Any]:
    from core.config import get_settings
    cfg = get_settings()

    # Try FreqTrade first (preferred execution backend)
    if cfg.exchange.use_freqtrade:
        try:
            return await _execute_via_freqtrade(state)
        except Exception as exc:
            logger.warning("freqtrade_unavailable_falling_back", error=str(exc))
            # Fall back to direct CCXT

    # Direct CCXT (dry_run=True by default)
    return await _execute_via_ccxt(state)
```

---

### 6d. Backtesting via FreqTrade

FreqTrade's backtesting is production-grade and handles:
- Accurate fill simulation with slippage
- Proper position sizing (stake_amount)
- Stop-loss/ROI exit simulation
- Walk-forward optimization (Hyperopt)
- Multi-timeframe data automatically

```bash
# Run backtest with our strategy on 30 days of real Binance data
docker-compose run --rm freqtrade backtesting \
  --config user_data/config.json \
  --strategy SuperBotFollower \
  --timerange 20240901-20241201 \
  --export trades \
  --export-filename user_data/backtest_results/step6.json

# Generate report
docker-compose run --rm freqtrade backtesting-show \
  --export-filename user_data/backtest_results/step6.json
```

For the SuperBot's signal-driven backtest, we need to **replay the SuperBot's
signals** through FreqTrade's backtesting engine. Implementation approach:

1. Run the SuperBot in `--dry-analysis` mode over historical data → saves signals
   to a JSON file with timestamps and directions.
2. Write a FreqTrade strategy that reads those signals: if `signal_file[timestamp]
   == BUY` → enter_long, if `SELL` → exit_long.
3. FreqTrade backtests the signal file with real slippage and order simulation.

This gives us a **much more realistic backtest** than our custom
`backtesting/backtest.py` (which doesn't simulate order fills, slippage, or
partial fills). We keep our backtest.py for rapid signal-only evaluation.

---

### 6e. Risk Parameter Tuning Process

After the FreqTrade integration is running in dry-run mode for ≥ 1 week:

1. **Pull performance data** from `/api/v1/profit` and `/api/v1/performance`.
2. **Analyse per-signal-strength results**: did STRONG_BUY outperform BUY? Was
   the confidence threshold of 0.3 too low?
3. **Adjust the risk table** in `agents/risk_manager/agent.py`:

```python
# Current Mandell rules (from 14-month backtest on MSTR/BTC)
_RISK_RULES = {
    Signal.STRONG_BUY:  RiskRule(pos_pct=25.0, sl_atr=1.5, rr=3.0),
    Signal.BUY:         RiskRule(pos_pct=15.0, sl_atr=2.0, rr=2.5),
    Signal.STRONG_SELL: RiskRule(pos_pct=25.0, sl_atr=1.5, rr=3.0),
    Signal.SELL:        RiskRule(pos_pct=15.0, sl_atr=2.0, rr=2.5),
}
# Tune based on Step 6 live dry-run results:
# - If win_rate < 45% on BUY: raise confidence threshold or lower pos_pct
# - If max_dd > 15%: tighten sl_atr from 2.0 to 1.5
# - If profit_factor < 1.3: revisit label_horizon in ml/labels.py
```

4. **Monte Carlo validation** on the tuned rules using `backtesting/monte_carlo.py`
   (already built).

---

## Step 6 Deliverables

- [ ] `deploy/freqtrade/` directory with Docker config + SuperBotFollower strategy
- [ ] `agents/executor/freqtrade_client.py` — async FreqTrade REST client
- [ ] `agents/executor/agent.py` — upgraded to route through FreqTrade when running
- [ ] `core/config.py` — `FreqTradeConfig` (base_url, username, password, enabled)
- [ ] `backtesting/signal_replay.py` — replay SuperBot signals through FreqTrade backtest
- [ ] `docs/FREQTRADE_SETUP.md` — setup guide (Docker, config, API keys)
- [ ] Tests: FreqTrade client unit tests (mock HTTP), integration test with real
  FreqTrade container (pytest-mark slow, opt-in)
- [ ] 1-week dry-run → performance report → updated `_RISK_RULES` table

**Not doing in Step 6**: FreqAI from FreqTrade. We already have `ml/freqai_bridge.py`
which is lighter and test-friendly. FreqAI adds complexity without proportional value
at this stage. Revisit in Step 7 or post-launch.

---

## Revised Overall Build Plan

| Step | Original Plan | Revised Plan | Rationale |
|------|--------------|--------------|-----------|
| 6 | Risk tuning + backtest | **FreqTrade sidecar + risk tuning** | Avoid reinventing execution; get production-grade backtesting free |
| 7 | Executor dry-run tests | **FreqTrade integration tests + paper trading 2 weeks** | Real dry-run beats unit tests for execution validation |
| 8 | React dashboard | **React dashboard + FreqUI embed + WebSocket** | Supplement with FreqUI for trade history; Lightweight Charts for signals |

### New Step 6.5 (inserted between 6 and 7)
**TradingView Webhook Integration** (see `docs/TRADINGVIEW_INTEGRATION.md`)  
- 2 hours of work, already specced.
- Adds a 5th signal source feeding the debate engine.
- High ROI: TradingView has a huge Pine Script library of battle-tested signals.

---

## Why FreqTrade Instead of Reinventing

| Capability | DIY Cost | FreqTrade (free) |
|------------|----------|-----------------|
| Slippage simulation in backtest | 2-3 days | Built-in |
| Partial fill simulation | 1-2 days | Built-in |
| Walk-forward optimization (Hyperopt) | 1 week | Built-in |
| Position tracking across restarts | 2-3 days | Built-in (SQLite) |
| Multi-exchange support | 1 week | 300+ exchanges via CCXT |
| Graceful order cancellation on crash | 1-2 days | Built-in |
| FreqUI live trade monitoring | 1 week | Built-in web UI |
| Active community / bug fixes | ∞ | 27k GitHub stars |

Total: ~3-4 weeks of work that FreqTrade already solved. The SuperBot adds value
in the **intelligence layer** (LangGraph + ML + debate). FreqTrade adds value in
the **execution layer**. Play to each system's strengths.
