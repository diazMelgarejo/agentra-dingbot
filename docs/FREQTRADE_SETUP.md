# FreqTrade Setup Guide — Agentra DingBot Sidecar

FreqTrade is an **optional execution sidecar**. The SuperBot auto-detects it
(`FREQTRADE_MODE=auto`) and uses it when both the binary and API are available.
Nothing breaks if FreqTrade is absent.

---

## Why Use FreqTrade at All?

FreqTrade provides things that would take weeks to build cleanly:
- CCXT order execution with retry logic and exchange quirks handled
- Position tracking across restarts (SQLite)
- Stop-loss enforcement even when the SuperBot is offline
- Walk-forward backtesting with realistic fill simulation
- FreqUI — a live monitoring web dashboard on `localhost:8080/ui/`

The SuperBot's intelligence (LangGraph agents, ML, debate) calls FreqTrade via
REST API (`force_entry`, `force_exit`). FreqTrade handles the plumbing.

---

## Installation (macOS / Linux, no Docker)

```bash
# 1. Create a dedicated venv (~600 MB — much less than Docker)
python3 -m venv ~/.venvs/freqtrade
source ~/.venvs/freqtrade/bin/activate

# 2. Install FreqTrade + FreqAI
pip install "freqtrade[freqai]"

# 3. Install FreqUI (pre-built, no npm/Node needed, ~25 MB)
freqtrade install-ui

# 4. Verify
freqtrade --version
# → freqtrade 2024.x ...
```

---

## Configuration

Copy the provided config and strategy into your FreqTrade data directory:

```bash
mkdir -p ~/freqtrade-data/user_data/strategies

# Config file (set your exchange keys here — leave empty for dry_run)
cp deploy/freqtrade/user_data/config.json \
   ~/freqtrade-data/user_data/config.json

# Strategy (thin shell — SuperBot sends all signals via REST)
cp deploy/freqtrade/user_data/strategies/SuperBotFollower.py \
   ~/freqtrade-data/user_data/strategies/
```

**Important config settings to review:**

```json
{
  "dry_run": true,          // ← ALWAYS start here. Switch to false only after ≥1 week paper
  "dry_run_wallet": 1000,   // Paper wallet size in USDT
  "api_server": {
    "listen_ip_address": "127.0.0.1",   // ← localhost only, never 0.0.0.0 in production
    "username": "superbot",
    "password": "superbot_password",    // ← CHANGE THIS before going live
    "jwt_secret_key": "CHANGE_ME_IN_PRODUCTION"   // ← CHANGE THIS
  }
}
```

---

## Running FreqTrade

```bash
source ~/.venvs/freqtrade/bin/activate

freqtrade trade \
  --config ~/freqtrade-data/user_data/config.json \
  --strategy SuperBotFollower \
  --userdir ~/freqtrade-data/user_data \
  --logfile ~/freqtrade-data/user_data/logs/freqtrade.log
```

FreqUI is now available at `http://localhost:8080/ui/` — default login:
- Username: `superbot`
- Password: `superbot_password` (change this in config.json)

---

## Running Both Together

```bash
# Terminal 1 — FreqTrade
source ~/.venvs/freqtrade/bin/activate
freqtrade trade --config ~/freqtrade-data/user_data/config.json \
  --strategy SuperBotFollower --userdir ~/freqtrade-data/user_data

# Terminal 2 — SuperBot
LLM_PROVIDER=none FREQTRADE_MODE=auto python src/deploy/live.py
```

Or via Make:
```bash
make freqtrade-start   # starts FreqTrade in background
make paper             # starts SuperBot in paper mode
```

---

## Verifying the Connection

```python
import asyncio
from src.agents.executor.freqtrade_client import FreqTradeClient

async def check():
    use, reason = await FreqTradeClient.detect(
        "http://localhost:8080", "superbot", "superbot_password"
    )
    print(f"FreqTrade available: {use}")
    print(f"Reason: {reason}")

asyncio.run(check())
```

Expected output when running:
```
FreqTrade available: True
Reason: detected binary=/home/user/.venvs/freqtrade/bin/freqtrade + API up
```

---

## Anti-Bias Analysis (Required Before Dry-Run)

Before any dry-run or live trading, run FreqTrade's built-in bias detection:

```bash
source ~/.venvs/freqtrade/bin/activate

# Download 90 days of data
freqtrade download-data \
  --config ~/freqtrade-data/user_data/config.json \
  --timerange 20240901-20241201 \
  --timeframes 5m 1h 4h

# 1. Lookahead analysis (detects future data leakage)
freqtrade lookahead-analysis \
  --config ~/freqtrade-data/user_data/config.json \
  --strategy SuperBotFollower \
  --timerange 20240901-20241201

# 2. Recursive analysis (detects indicator instability)
freqtrade recursive-analysis \
  --config ~/freqtrade-data/user_data/config.json \
  --strategy SuperBotFollower \
  --startup-candle 199 299 399 499

# 3. Full backtest
freqtrade backtesting \
  --config ~/freqtrade-data/user_data/config.json \
  --strategy SuperBotFollower \
  --timerange 20240901-20241201 \
  --export trades
```

**Both bias checks must pass before going to dry-run.**
See `docs/RESEARCH_STEP6_STEP7.md` for why these matter.

---

## Makefile Targets

```bash
make freqtrade-start      # Start FreqTrade trade mode
make freqtrade-stop       # Stop FreqTrade
make freqtrade-backtest   # Run 30-day backtest
make freqtrade-lookahead  # Run lookahead-analysis (CI gate)
make freqtrade-recursive  # Run recursive-analysis (CI gate)
```

---

## Security Checklist

Before switching `dry_run: false`:

```
[ ] Change api_server.password from "superbot_password"
[ ] Change api_server.jwt_secret_key from "CHANGE_ME_IN_PRODUCTION"
[ ] Verify exchange.key has trade-only permission (no withdraw)
[ ] Verify exchange.key is IP-whitelisted to your machine
[ ] Run lookahead-analysis → no bias detected
[ ] Run recursive-analysis → no instability detected
[ ] Paper trade for ≥1 week
[ ] Review equity curve + Sharpe in FreqUI
```

---

## Troubleshooting

**"FreqTrade not detected" in SuperBot logs**
- Is FreqTrade running? Check `freqtrade trade ...` is active.
- Is the port open? `curl http://localhost:8080/api/v1/ping`
- Is the binary on a known path? Run `which freqtrade` or check `~/.venvs/freqtrade/bin/freqtrade`.

**"Authentication failed" from FreqTradeClient**
- Check `username` and `password` in config.json match what FreqTradeClient uses.
- The SuperBot reads `FREQTRADE_USERNAME` / `FREQTRADE_PASSWORD` from `.env`.

**"No module named freqtrade" in SuperBot**
- Expected — the SuperBot never imports FreqTrade (GPL boundary).
- The connection is HTTP-only via `freqtrade_client.py`.
