# TradingView Webhook Integration — Implementation Guide

## Status: ✅ Implemented in `dashboard/app.py`

The webhook endpoint `/webhook/tradingview` is live in the codebase right now.
This document explains what was built, how to activate it today, and what the
5-step rollout looks like from dev to production.

---

## How It Helps Right Now (Current Stage)

The SuperBot at Step 5 produces signals from four internal sources:
technical indicators, sentiment (F&G + VIX), on-chain funding rates, and ML.
These all read the same Binance OHLCV.

**The gap**: TradingView's Pine Script community has thousands of battle-tested
strategies — momentum, volume profile, Wyckoff, SMC, options flow correlation.
You cannot easily replicate that library from scratch.

**The webhook** adds a 5th signal source: **any Pine Script alert from any
strategy you trust**. The alert fires when your TradingView condition is met,
hits your FastAPI endpoint, gets validated, queued, and passed into the debate
engine as additional evidence. The debate engine's LLM judge then weighs all
five sources and produces a more informed consensus.

Concrete example:
- Your Binance BTC/USDT chart has a [LuxAlgo Oscillator](https://www.tradingview.com/script/q4y9sPCF/) alert set on bullish divergence.
- TradingView fires the webhook → `{"symbol":"BINANCE:BTCUSDT","action":"BUY","strategy":"luxalgo_div"}`.
- The debate engine sees `[TRADINGVIEW] Signal: BUY | strategy: luxalgo_div`.
- Combined with technical BULL + sentiment FEAR (contrarian buy) → consensus likely upgrades from BUY to STRONG_BUY with higher confidence.

---

## What Was Built

### `/webhook/tradingview` (POST)

```
TradingView alert fires
       │
       ▼
IP check (log unexpected IPs; hard-block if TV_STRICT_IP_CHECK=true)
       │
       ▼
HMAC-SHA256 signature verification (if TRADINGVIEW_WEBHOOK_SECRET is set)
       │
       ▼
JSON parse + symbol normalisation (BINANCE:BTCUSDT → BTC/USDT)
       │
       ▼
Append to _external_signals queue (last 50 signals, in-memory)
       │
       ▼
If TV_AUTO_CYCLE=true → asyncio.create_task(run_one_cycle(symbol))
       │
       ▼
Return {"status":"received","symbol":"BTC/USDT","action":"BUY"}
```

### `/webhook/tradingview/signals` (GET)

Returns the last 10 queued signals for debugging. Open in your browser or
`curl http://localhost:8000/webhook/tradingview/signals` to inspect.

### `get_latest_tv_signal(symbol)` (function)

Importable by any agent. Called like:
```python
from dashboard.app import get_latest_tv_signal
tv = get_latest_tv_signal("BTC/USDT")
if tv and tv.get("action") == "BUY":
    score += 1.5
    reasons.append(f"TradingView: {tv.get('strategy','unknown')}")
```

---

## Activate Today (3 Steps)

### Step 1 — Start the dashboard

```bash
cd superbot
pip install fastapi uvicorn pydantic --break-system-packages
python dashboard/app.py
# → Uvicorn running on http://0.0.0.0:8000
```

Or via Make:
```bash
make dashboard
```

### Step 2 — Expose a public URL (dev)

```bash
# Option A: ngrok (free, 1 tunnel)
ngrok http 8000
# → https://abc123.ngrok-free.app

# Option B: localtunnel (free, no account)
npx localtunnel --port 8000
# → https://quick-deer-42.loca.lt

# Option C: Cloudflare tunnel (free, stable)
cloudflared tunnel --url http://localhost:8000
```

Copy the HTTPS URL. You'll paste it into TradingView next.

### Step 3 — Create the TradingView alert

1. Open your BTC/USDT chart on TradingView.
2. Add **any indicator** that has arrow/alert signals (or use the built-in
   script below as a starter).
3. Click the **Alert** icon (clock icon) → **Create Alert**.
4. Under **Condition**: set to your indicator signal.
5. Under **Notifications**: enable **Webhook URL**.
6. Paste: `https://your-ngrok-url.ngrok-free.app/webhook/tradingview`
7. Under **Message**, paste:

```json
{"symbol":"{{ticker}}","action":"BUY","price":{{close}},"rsi":{{plot_0}},"timeframe":"{{interval}}","strategy":"my_indicator"}
```

(Change `"BUY"` to `"SELL"` for bearish conditions, or use a variable.)

8. Click **Create**.

Test it: manually trigger the alert by right-clicking on the indicator →
"Add Alert" → "Fire once" → check `GET /webhook/tradingview/signals`.

---

## Pine Script Starter — EMA Cross + RSI Filter

This is a minimal Pine Script you can add to any chart. It fires BUY/SELL
webhooks on EMA crossovers confirmed by RSI levels.

```pine
//@version=5
indicator("SuperBot Signal Feeder", overlay=true)

// Parameters
ema_fast = input.int(9,  "Fast EMA")
ema_slow = input.int(21, "Slow EMA")
rsi_period = input.int(14, "RSI Period")
rsi_ob = input.int(65, "Overbought threshold")
rsi_os = input.int(35, "Oversold threshold")

// Indicators
fast  = ta.ema(close, ema_fast)
slow  = ta.ema(close, ema_slow)
rsi   = ta.rsi(close, rsi_period)
[ml, ms, mh] = ta.macd(close, 12, 26, 9)

// Conditions
bull_cross = ta.crossover(fast, slow) and rsi < rsi_ob
bear_cross = ta.crossunder(fast, slow) and rsi > rsi_os

// Visuals
plotshape(bull_cross, style=shape.triangleup,   location=location.belowbar, color=color.green, size=size.small)
plotshape(bear_cross, style=shape.triangledown, location=location.abovebar, color=color.red,   size=size.small)

// Alerts — these fire the webhook
if bull_cross
    alert('{"symbol":"' + syminfo.ticker + '","action":"BUY","price":' +
          str.tostring(close, "#.##") + ',"rsi":' + str.tostring(rsi, "#.##") +
          ',"macd_hist":' + str.tostring(mh, "#.####") + ',"timeframe":"' +
          timeframe.period + '","strategy":"ema_cross_rsi"}',
          alert.freq_once_per_bar_close)

if bear_cross
    alert('{"symbol":"' + syminfo.ticker + '","action":"SELL","price":' +
          str.tostring(close, "#.##") + ',"rsi":' + str.tostring(rsi, "#.##") +
          ',"macd_hist":' + str.tostring(mh, "#.####") + ',"timeframe":"' +
          timeframe.period + '","strategy":"ema_cross_rsi"}',
          alert.freq_once_per_bar_close)
```

Add this script to your chart, then create an alert on either `bull_cross` or
`bear_cross` with the webhook URL.

---

## Wire TV Signals into the Debate Engine (Code)

The webhook stores signals in memory. The debate engine needs to read them.
Add this to `agents/debate_engine/agent.py` in `_compile_evidence()`:

```python
def _compile_evidence(state: Dict[str, Any]) -> str:
    parts = []

    # ... existing technical / sentiment / on-chain / ML blocks ...

    # TradingView external signal (optional 5th source)
    symbol = state.get("symbol", "BTC/USDT")
    try:
        from dashboard.app import get_latest_tv_signal
        tv = get_latest_tv_signal(symbol)
        if tv:
            age_note = ""
            received = tv.get("received_at")
            if received:
                from datetime import datetime, timezone
                dt = datetime.fromisoformat(received)
                age_s = (datetime.now(timezone.utc) - dt).total_seconds()
                if age_s > 3600:
                    age_note = f" (⚠️ {int(age_s/60)}m old)"
            parts.append(
                f"[TRADINGVIEW] Signal: {tv.get('action','?')}{age_note}\n"
                f"  → strategy={tv.get('strategy','?')} "
                f"price={tv.get('price','?')} "
                f"rsi={tv.get('rsi','?')} "
                f"tf={tv.get('timeframe','?')}"
            )
    except ImportError:
        pass  # dashboard not running in test mode — skip gracefully

    return "\n\n".join(parts) if parts else "No evidence available."
```

Note the **age check**: a TradingView signal older than 60 minutes is flagged
as stale. The LLM judge will naturally discount it, but the visual indicator
helps you debug missed-signal scenarios.

---

## Environment Variables

Add to `.env`:

```bash
# TradingView webhook security
TRADINGVIEW_WEBHOOK_SECRET=your_random_32_char_string_here

# Hard-block non-TradingView IPs (set true in production on a VPS)
TV_STRICT_IP_CHECK=false

# Automatically trigger a full pipeline cycle when a BUY/SELL webhook arrives
# Set true for reactive mode; false for purely timed cycles
TV_AUTO_CYCLE=false
```

---

## Production Checklist (VPS Deployment)

```
[ ] Deploy to VPS with a fixed IP
[ ] Set up nginx + Let's Encrypt SSL (letsencrypt.org — free)
[ ] Set TV_STRICT_IP_CHECK=true
[ ] Set TRADINGVIEW_WEBHOOK_SECRET to a 32+ char random string
[ ] Add TradingView's 4 IPs to UFW/iptables allowlist on port 443
[ ] Test the webhook with curl before pointing TradingView at it
[ ] Monitor /webhook/tradingview/signals endpoint for dropped signals
```

---

## Testing the Webhook Without TradingView

```bash
# Simulate a TradingView BUY alert locally
curl -X POST http://localhost:8000/webhook/tradingview \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BINANCE:BTCUSDT","action":"BUY","price":67432.50,
       "rsi":28.4,"timeframe":"4h","strategy":"ema_cross_rsi"}'

# Expected response:
# {"status":"received","symbol":"BTC/USDT","action":"BUY","queued":1}

# Check the queue
curl http://localhost:8000/webhook/tradingview/signals
```

---

## What TradingView Still Cannot Do

| Desire | Reality |
|--------|---------|
| Pull OHLCV data FROM TradingView | No official API. Use CCXT → Binance (already in Step 2) |
| Push signals INTO TradingView charts | Not possible — webhooks only go outward |
| Programmatically trigger alerts | Pine Script runs server-side; you can't call it |
| Get live chart data in real-time via webhook | Webhooks fire on close of bar, not tick |

The bot's Binance OHLCV (Step 2) is more reliable and lower-latency than
anything TradingView can provide. TradingView's value here is the **Pine Script
signal logic** and the **alert engine** — not the data.
