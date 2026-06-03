# TradingView ↔ Claude Integration Guide

## The Honest Picture First

TradingView does **not** have an official public REST API for market data. This is the most common misconception — people assume "TradingView API" means you can pull OHLCV data on demand. You can't. Their Terms of Service prohibit scraping, and unofficial libraries like `tvdatafeed` break frequently and violate ToS.

But there are **four legitimate, production-ready integration paths**, and two of them slot directly into the SuperBot architecture you already have.

---

## Integration Map

```
TradingView                         Your SuperBot
─────────────────                   ──────────────────────────────────
                                    CCXT → Binance OHLCV (your data)
  Pine Script alert() ─────────►   /webhook/tradingview (FastAPI)
                                         ↓
                                    TradingState.signals injection
                                         ↓
                                    LangGraph pipeline cycle

  Chart screenshot ────────────►   Claude vision API
                                    (LLaVA / visual_analyst agent)

                                    dashboard/frontend/
  Lightweight Charts library  ◄───  (your own React UI, TradingView-style
  (open source, you host it)         charts fed by your CCXT data)

  Advanced Charting Library   ◄───  UDF datafeed endpoint (Step 8+)
  (requires application)             /api/udf/history
```

---

## Path 1: TradingView Alerts → Webhook → Your Pipeline (BEST ROI)

This is the primary integration. TradingView Pro/Pro+/Premium plans let you create **alert webhooks**: when a Pine Script condition fires, TradingView sends a POST request to any URL you control.

### What you need
- TradingView Pro or higher (Pro = ~$15/month; Pro+ = ~$30/month).
- A public URL for your dashboard (ngrok for local dev; VPS for production).

### Step 1: Add the webhook endpoint to `dashboard/app.py`

```python
import hashlib
import hmac
from fastapi import Request, HTTPException

TRADINGVIEW_SECRET = os.getenv("TRADINGVIEW_WEBHOOK_SECRET", "changeme")

# TradingView's published webhook source IPs (allowlist these at firewall level)
TRADINGVIEW_IPS = {
    "52.89.214.238",
    "34.212.75.30",
    "54.218.53.128",
    "52.32.178.7",
}

@app.post("/webhook/tradingview")
async def tradingview_webhook(request: Request):
    # IP allowlist check
    client_ip = request.client.host
    if client_ip not in TRADINGVIEW_IPS:
        # Log but don't hard-block in dev (IPs change rarely)
        logger.warning("webhook_unexpected_ip", ip=client_ip)

    # Signature validation via shared secret
    body = await request.body()
    sig_header = request.headers.get("X-Webhook-Secret", "")
    expected = hmac.new(
        TRADINGVIEW_SECRET.encode(), body, hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(sig_header, expected):
        raise HTTPException(status_code=401, detail="Invalid signature")

    payload = await request.json()
    # Expected payload shape (you define this in Pine Script):
    # {
    #   "symbol": "BINANCE:BTCUSDT",
    #   "action": "BUY" | "SELL" | "NEUTRAL",
    #   "rsi": 28.4,
    #   "macd_hist": 0.0023,
    #   "price": 67432.50,
    #   "timeframe": "4h",
    #   "strategy": "ema_cross_rsi"
    # }

    logger.info("webhook_received", symbol=payload.get("symbol"),
                action=payload.get("action"))

    # Option A: Store as external signal for next cycle (non-blocking)
    await signal_store.put(payload)

    # Option B: Immediately trigger a pipeline cycle with the signal injected
    # asyncio.create_task(run_one_cycle_with_external_signal(payload))

    return {"status": "received", "action": payload.get("action")}
```

### Step 2: Pine Script alert configuration

In TradingView, write your indicator/strategy in Pine Script:

```pine
//@version=5
indicator("SuperBot Signal Feed", overlay=true)

// Your existing signals
rsi = ta.rsi(close, 14)
[macd_line, signal_line, hist] = ta.macd(close, 12, 26, 9)
ema9  = ta.ema(close, 9)
ema21 = ta.ema(close, 21)

// Condition: Bull stack + oversold RSI
bull_signal = ema9 > ema21 and rsi < 35 and hist > 0
bear_signal = ema9 < ema21 and rsi > 65 and hist < 0

// Fire the alert with a JSON payload
if bull_signal
    alert('{"symbol":"' + syminfo.ticker + '","action":"BUY","rsi":' +
          str.tostring(rsi, "#.##") + ',"macd_hist":' +
          str.tostring(hist, "#.####") + ',"price":' +
          str.tostring(close, "#.##") + ',"timeframe":"' +
          timeframe.period + '","strategy":"ema_rsi_bull"}',
          alert.freq_once_per_bar_close)

if bear_signal
    alert('{"symbol":"' + syminfo.ticker + '","action":"SELL","rsi":' +
          str.tostring(rsi, "#.##") + ',"macd_hist":' +
          str.tostring(hist, "#.####") + ',"price":' +
          str.tostring(close, "#.##") + ',"timeframe":"' +
          timeframe.period + '","strategy":"ema_rsi_bear"}',
          alert.freq_once_per_bar_close)
```

In the TradingView alert dialog:
- **Condition**: your indicator signal
- **Webhook URL**: `https://your-domain.com/webhook/tradingview`
- **Message**: the JSON above (TradingView fills the `{{close}}` etc. placeholders)
- Add your secret in the Alert Message or as a custom header (TradingView Pro+ supports custom headers)

### Step 3: Inject the webhook signal into the SuperBot pipeline

```python
# In agents/technical_analyst/agent.py — add external signal blending
async def run(state: Dict[str, Any]) -> Dict[str, Any]:
    # ... existing TA logic ...

    # Check if a TradingView webhook signal exists for this symbol
    tv_signal = await signal_store.get(state.get("symbol"))
    if tv_signal:
        tv_action = tv_signal.get("action", "NEUTRAL")
        if tv_action == "BUY":
            score += 1.5   # TradingView confirmation bonus
            reasons.append(f"TradingView alert: {tv_signal.get('strategy')}")
        elif tv_action == "SELL":
            score -= 1.5
            reasons.append(f"TradingView alert: {tv_signal.get('strategy')}")
```

### Dev setup with ngrok

```bash
# Install ngrok (free tier: 1 tunnel, random URL)
brew install ngrok   # or download from ngrok.com

# Expose local FastAPI
ngrok http 8000

# Copy the HTTPS URL (e.g. https://abc123.ngrok.io) → paste into TradingView webhook field
```

For production, use a VPS with nginx + let's encrypt SSL.

---

## Path 2: Claude Analyzes TradingView Chart Screenshots (Vision Agent)

You screenshot a TradingView chart → pass it to Claude or LLaVA → get a natural-language + structured analysis back. This is the "Fenix v2.0 visual agent" pattern from the architecture doc.

### Claude API vision call

```python
import base64
import httpx

async def analyze_chart_screenshot(image_path: str) -> dict:
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    response = await httpx.AsyncClient().post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": """Analyze this BTC/USD candlestick chart and return JSON:
{
  "trend": "up|down|sideways",
  "key_levels": [{"type": "support|resistance", "price": number}],
  "patterns": ["hammer", "engulfing", etc.],
  "rsi_visible": number_or_null,
  "ema_cross": "bull|bear|flat|unknown",
  "signal": "BUY|SELL|NEUTRAL",
  "confidence": 0.0_to_1.0,
  "reasoning": "brief explanation"
}
Return only valid JSON, no markdown."""
                    }
                ],
            }],
        },
        timeout=30,
    )
    result = response.json()
    text = result["content"][0]["text"]
    return json.loads(text)
```

### Automated screenshot capture (for local setup)

```python
import pyautogui  # or playwright for headless browser

async def capture_tradingview_chart(symbol: str = "BTCUSDT",
                                     timeframe: str = "240") -> str:
    """
    Opens TradingView chart URL in a headless browser and takes a screenshot.
    Requires playwright: pip install playwright && playwright install chromium
    """
    from playwright.async_api import async_playwright
    url = f"https://www.tradingview.com/chart/?symbol={symbol}&interval={timeframe}"
    path = f"/tmp/tv_chart_{symbol}_{timeframe}.png"

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 720})
        await page.goto(url)
        await page.wait_for_timeout(3000)  # wait for chart to render
        await page.screenshot(path=path)
        await browser.close()
    return path
```

### Wire into the LangGraph pipeline as `visual_analyst` agent

```python
# Future: agents/visual_analyst/agent.py
async def run(state: Dict[str, Any]) -> Dict[str, Any]:
    symbol = state.get("symbol", "BTC/USDT")
    tf_map = {"4h": "240", "1h": "60", "5m": "5"}
    timeframe = tf_map.get(state.get("config", {}).get("primary_tf", "4h"), "240")

    try:
        img_path = await capture_tradingview_chart(symbol.replace("/",""), timeframe)
        analysis = await analyze_chart_screenshot(img_path)
        # ... create VisualSnapshot, return {"visual": snap}
    except Exception as exc:
        logger.warning("visual_agent_failed", error=str(exc))
        return {"visual": None}
```

---

## Path 3: Lightweight Charts in Your Dashboard (RECOMMENDED for Step 8)

[TradingView Lightweight Charts™](https://github.com/tradingview/lightweight-charts) is open-source (Apache-2.0) and gives you TradingView-quality candlestick charts in your React dashboard, **fed by your own CCXT data**.

This is better than embedding TradingView's hosted chart for your use case because:
- Your data (from CCXT/Binance) is already clean and available.
- You control what overlays, signals, and annotations appear.
- No third-party iframe → no CORS/embedding restrictions.
- You can plot your bot's buy/sell signals directly on the chart.

### Step 8 React component

```tsx
// dashboard/frontend/src/components/CandlestickChart.tsx
import { useEffect, useRef } from 'react';
import { createChart, CandlestickSeries, LineSeries } from 'lightweight-charts';

interface OHLCVBar {
  time: number;   // Unix timestamp (seconds)
  open: number;
  high: number;
  low: number;
  close: number;
}

export function CandlestickChart({
  data,
  signals,
}: {
  data: OHLCVBar[];
  signals: Array<{ time: number; action: 'BUY' | 'SELL' }>;
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      width: 800,
      height: 400,
      layout: { background: { color: '#1a1a2e' }, textColor: '#e0e0e0' },
      grid: { vertLines: { color: '#2a2a3e' }, horzLines: { color: '#2a2a3e' } },
    });

    const candleSeries = chart.addSeries(CandlestickSeries);
    candleSeries.setData(data);

    // Overlay your bot's buy/sell signals as markers
    const markers = signals.map(s => ({
      time: s.time,
      position: s.action === 'BUY' ? 'belowBar' : 'aboveBar',
      color: s.action === 'BUY' ? '#26a69a' : '#ef5350',
      shape: s.action === 'BUY' ? 'arrowUp' : 'arrowDown',
      text: s.action,
    }));
    candleSeries.setMarkers(markers);

    chart.timeScale().fitContent();
    return () => chart.remove();
  }, [data, signals]);

  return <div ref={containerRef} />;
}
```

### Feed it from your FastAPI WebSocket

```python
# dashboard/app.py — add WebSocket endpoint (Step 8)
@app.websocket("/ws/signals")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        result = await run_one_cycle("BTC/USDT", dry_run=True)
        # Convert OHLCV to lightweight-charts format
        ohlcv_4h = result.get("ohlcv", {}).get("4h")
        if ohlcv_4h is not None:
            bars = [
                {"time": int(ts.timestamp()), "open": float(row.open),
                 "high": float(row.high), "low": float(row.low), "close": float(row.close)}
                for ts, row in ohlcv_4h.iterrows()
            ]
            await websocket.send_json({
                "type": "cycle_update",
                "ohlcv": bars[-200:],   # last 200 bars
                "signal": result.get("debate_consensus", "NEUTRAL"),
                "confidence": result.get("debate_confidence", 0.0),
                "ml_prob_up": result.get("ml", {}).get("prob_up") if result.get("ml") else None,
                "fear_greed": result.get("sentiment_raw", {}).get("fear_greed", {}).get("value"),
                "vix": result.get("sentiment_raw", {}).get("vix"),
            })
        await asyncio.sleep(60)   # update every minute
```

---

## Path 4: Advanced Charting Library + Custom Datafeed (Advanced)

TradingView's [Advanced Charting Library](https://www.tradingview.com/HTML5-stock-forex-bitcoin-charting-library/) is the same library used on TradingView.com itself. It's free to use but **requires an application** (for non-broker partners, approval can take weeks).

If approved, you serve your own data via the **UDF (Universal Data Feed) protocol**:

```python
# dashboard/app.py — UDF datafeed endpoints
@app.get("/api/udf/config")
async def udf_config():
    return {
        "supported_resolutions": ["5", "15", "60", "240", "D"],
        "supports_group_request": False,
        "supports_marks": True,
        "supports_search": True,
        "supports_timescale_marks": True,
    }

@app.get("/api/udf/history")
async def udf_history(symbol: str, resolution: str,
                       from_ts: int = Query(alias="from"),
                       to_ts: int = Query(alias="to")):
    # Map resolution to CCXT timeframe
    tf_map = {"5": "5m", "15": "15m", "60": "1h", "240": "4h", "D": "1d"}
    tf = tf_map.get(resolution, "1h")
    df = await fetch_ohlcv(symbol.replace("_", "/"), tf, 500)
    # Filter by from/to timestamps and return UDF format
    filtered = df[(df.index.astype(int) // 1e9 >= from_ts) &
                   (df.index.astype(int) // 1e9 <= to_ts)]
    return {
        "s": "ok",
        "t": [int(ts.timestamp()) for ts in filtered.index],
        "o": filtered["open"].tolist(),
        "h": filtered["high"].tolist(),
        "l": filtered["low"].tolist(),
        "c": filtered["close"].tolist(),
        "v": filtered["volume"].tolist(),
    }
```

**Recommendation**: Unless you need the exact TradingView look-and-feel and have the time to get approved, use Lightweight Charts (Path 3) instead. It covers 95% of use cases with zero gatekeeping.

---

## What TradingView CANNOT Do (Limitations)

| Capability | Status | Alternative |
|------------|--------|-------------|
| Pull OHLCV data from TradingView via API | ❌ No official API | Use CCXT → Binance (already in Step 2) |
| Push signals INTO TradingView charts | ❌ Not possible | Render in your own Lightweight Charts dashboard |
| Trigger Pine Script from external code | ❌ Not possible | Webhooks go OUT from TradingView, not in |
| Access TradingView's social/ideas feed | ❌ No API | Scraping violates ToS |
| tvdatafeed (unofficial library) | ⚠️ ToS violation | CCXT + Binance (same data, official) |

---

## Implementation Priority for SuperBot

| Priority | Task | Effort | Value |
|----------|------|--------|-------|
| **1** | Webhook receiver endpoint in FastAPI | 2 hours | Immediate: TradingView Pine signals feed directly into your pipeline |
| **2** | Lightweight Charts in Step 8 React dashboard | 4 hours | Visualize your own bot's signals on professional charts |
| **3** | Claude vision for chart screenshots | 4 hours | Pattern recognition layer (visual_analyst agent) |
| **4** | Advanced Charting Library UDF datafeed | 1-2 weeks | Only if you need the exact TradingView UI embedded |

**Start with Priority 1** — it's the highest-value integration and uses infrastructure you already have (FastAPI dashboard).

---

## Environment Variables to Add

```bash
# .env additions for TradingView integration
TRADINGVIEW_WEBHOOK_SECRET=your_random_secret_here   # shared with TV alert
ANTHROPIC_API_KEY=sk-ant-...                          # for vision chart analysis
```

---

## Testing the Webhook Locally

```bash
# Start your dashboard
uvicorn dashboard.app:app --port 8000

# Expose via ngrok
ngrok http 8000
# → https://abc123.ngrok.io

# Test the webhook manually (simulate TradingView)
curl -X POST https://abc123.ngrok.io/webhook/tradingview \
  -H "Content-Type: application/json" \
  -H "X-Webhook-Secret: $(echo -n '{"symbol":"BTCUSDT","action":"BUY","price":67500}' | hmac256 your_random_secret_here)" \
  -d '{"symbol":"BTCUSDT","action":"BUY","rsi":28.4,"price":67500,"timeframe":"4h","strategy":"ema_rsi"}'

# Expected response:
# {"status": "received", "action": "BUY"}
```
