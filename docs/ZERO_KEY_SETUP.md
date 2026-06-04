# Zero-Key Setup — Run Entirely on Free, No-Key Sources

The SuperBot runs **fully functional with no API keys, no local LLM, and no
Docker**. Every data source is a free public endpoint, the debate engine has a
deterministic heuristic judge that needs no LLM, and FreqTrade is an optional
sidecar that's auto-detected only if you already have it.

---

## What Works With Zero Configuration

| Component | Source | Key required? |
|-----------|--------|--------------|
| BTC/ETH OHLCV | Binance public API via CCXT | ❌ None |
| Perp funding rate | Binance futures public API via CCXT | ❌ None |
| Polymarket markets | Gamma API (public) | ❌ None |
| Polymarket prices/spreads | CLOB REST (public) | ❌ None |
| Polymarket orderbook | CLOB WebSocket (public) | ❌ None |
| Fear & Greed Index | alternative.me (free) | ❌ None |
| VIX | yfinance / Yahoo Finance (free) | ❌ None |
| ML signal | scikit-learn HistGBM (local) | ❌ None |
| **Debate consensus** | **heuristic judge (local, deterministic)** | ❌ None |
| Spot execution | CCXT dry-run | ❌ None |
| Polymarket execution (dry-run) | local simulation | ❌ None |

The **only** things that need keys are entirely optional:
- **Live** spot trading (Binance API key) — not needed for paper/dry-run.
- **Live** Polymarket trading (Polygon wallet key) — not needed for dry-run.
- **LLM debate** (OpenAI key) — only if you want LLM reasoning instead of the
  free heuristic judge; Ollama is the free local alternative.
- **Telegram alerts** (bot token) — optional notifications.

---

## The Heuristic Judge (replaces the LLM for free)

With `LLM_PROVIDER=none` (the default), the debate engine skips the LLM entirely
and uses a deterministic weighted vote across all analyst signals:

```
consensus = weighted_average(
    technical.signal × technical.confidence × 1.0
  + ml.signal        × ml.confidence        × 0.9
  + sentiment.signal × sentiment.confidence × 0.7
  + onchain.signal   × onchain.confidence   × 0.5
)
```

The numeric average (range −2 … +2) maps back to STRONG_BUY / BUY / NEUTRAL /
SELL / STRONG_SELL, and its magnitude becomes the confidence. This is fully
reproducible, instant, and needs no network or keys.

If you later set `LLM_PROVIDER=ollama` or `openai`, the engine uses the LLM
bull/bear/judge debate instead — and if that LLM is unreachable, it
automatically falls back to the heuristic judge. You never get a stall.

---

## Run It Right Now (3 commands, no keys)

```bash
# 1. Install (all free, pip-only — no Docker, no system services)
pip install pandas numpy pydantic pydantic-settings aiohttp ccxt yfinance \
            pandas-ta langgraph scikit-learn joblib fastapi uvicorn structlog \
            --break-system-packages

# 2. Run one analysis cycle (dry-run, BTC/USDT)
python -m core.cli run --symbol BTC/USDT

# 3. Or start the dashboard + paper-trading loop
python deploy/live.py --paper
```

No `.env` file is required. The defaults are:
```
LLM_PROVIDER=none          # heuristic judge, no key
FREQTRADE_MODE=auto        # use FreqTrade only if already installed + running
PAPER_MODE=true            # never places real orders
EXCHANGE_SANDBOX=true      # testnet
```

---

## FreqTrade Is Optional and Auto-Detected

The bot checks for an existing FreqTrade install on startup:

```
FREQTRADE_MODE=auto  (default)
   ├── FreqTrade binary found AND API reachable → route execution through it
   ├── binary found but API not running         → use homegrown (logs a hint)
   └── not installed                            → use homegrown CCXT executor

FREQTRADE_MODE=off   → never use FreqTrade, always homegrown
FREQTRADE_MODE=on    → require FreqTrade (error if API unreachable)
```

Detection looks for the `freqtrade` binary in: anything on `PATH`,
`~/.venvs/freqtrade/bin/`, `/opt/homebrew/bin/`, `/usr/local/bin/`,
`~/.local/bin/`. It then probes `http://localhost:8080/api/v1/ping`.

You will see the routing decision in every order log:
```
spot_order_built ... routed_via=homegrown ft="FreqTrade not installed / not running — using homegrown executor"
```
or, if you have it:
```
spot_order_built ... routed_via=freqtrade ft="detected binary=/opt/homebrew/bin/freqtrade + API up"
```

Nothing imports the `freqtrade` Python package (it's GPL-3.0). The bot only
speaks to it over HTTP, so this project stays Apache-2.0 either way. See
`docs/LICENSE_RATIONALE.md`.

---

## Optional Upgrades (each independent, each still mostly free)

| Want this? | Set this | Cost |
|------------|----------|------|
| LLM reasoning, local & free | `LLM_PROVIDER=ollama` + run Ollama | Free (local GPU/CPU) |
| LLM reasoning, cloud | `LLM_PROVIDER=openai` + `OPENAI_API_KEY` | Paid API |
| TradingView signals | start dashboard + set webhook | TradingView Pro (~$15/mo) |
| FreqTrade execution backend | `pip install freqtrade` in a venv | Free (OSS) |
| Telegram alerts | `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` | Free |
| Live spot trading | `EXCHANGE_API_KEY/SECRET`, `--live` | Free (your funds at risk) |

---

## Verify Zero-Key Operation

```bash
# Confirm the full pipeline runs with no keys and no LLM
LLM_PROVIDER=none FREQTRADE_MODE=off python -m pytest \
  tests/test_freqtrade_optional.py::TestZeroKeyPipeline -v

# → test_full_pipeline_no_keys_no_llm PASSED
```
