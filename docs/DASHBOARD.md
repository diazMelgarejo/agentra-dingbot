# Dashboard (Step 8) — Plan, Demo & Deployment

The Agentra DingBot dashboard is a **self-contained static page** (`docs/index.html`).
It shows live signal consensus, the four analyst agents, sentiment gauges, a
candlestick chart, Polymarket markets, and the risk assessment.

**Live demo:** https://diazmelgarejo.github.io/agentra-dingbot/

---

## Design Decision: Static Single-File, Not a Build Pipeline

The initial plan (see git history / review notes) proposed a React/Vue/Svelte
app with a Vite build step that outputs to `/docs`. **We deliberately diverged**
to a single self-contained `docs/index.html` for these reasons:

1. **Immediate demo after download.** The core requirement was that someone can
   clone the repo and open the dashboard with zero build step. A static file
   satisfies this — `open docs/index.html` just works.
2. **GitHub Pages with no CI build.** Serving `/docs` directly means Pages needs
   no Node toolchain, no `npm ci`, no build cache. Fewer moving parts, fewer
   ways for the deploy to break.
3. **Matches the zero-key philosophy.** The whole project runs with no required
   external services. The dashboard mirrors that: no build, no bundler, no
   framework lock-in. It degrades gracefully offline.
4. **Single artifact is auditable.** One 700-line file with inline CSS/JS is
   easier for the agent council to read and modify than a bundled dist.

If the dashboard grows beyond what a single file can cleanly hold, the migration
path is documented under "Future: Build Pipeline" below.

---

## What's in the Dashboard

| Panel | Data source (live mode) | Demo mode |
|-------|------------------------|-----------|
| KPI row — consensus, confidence, ML prob, F&G, VIX | `debate_consensus`, `ml.prob_up`, `sentiment.*` | Simulated, realistic ranges |
| Candlestick chart (1H) | `ohlcv_4h` from CCXT | Random-walk OHLCV around $66–67k |
| Sentiment gauges (F&G + ML) | `sentiment.fear_greed_index`, `ml.prob_up` | Simulated |
| Technical agent | `technical.{signal,rsi_14,ema_cross}` | BUY / RSI 38 / BULL |
| Sentiment agent | `sentiment.{signal,fear_greed_index,vix}` | BUY / 28 / 19.5 |
| On-chain agent | `onchain.{signal,funding_rate}` | NEUTRAL / +0.01% |
| ML agent | `ml.{signal,prob_up,model_type}` | BUY / 0.67 / sklearn_hgb |
| Polymarket markets | Gamma + CLOB (planned bind) | 3 sample 5-min markets |
| Risk assessment | risk manager output (planned bind) | 10.8% / 2% SL / 5% TP |

### Design system (Clarence)
- **Color encodes state, not decoration**: green=bullish, red=bearish, amber=warning, grey=neutral
- **KPI values 2–3× larger than labels** for instant scannability
- Left-sidebar + top-header + card-grid SaaS layout
- Pill status badges, candlestick + line charts (never pie)
- Explicitly designed light **and** dark surfaces; toggle persists in `localStorage`
- Fonts: Sora (display) · Manrope (body) · JetBrains Mono (prices/data)

---

## Local Demo (Three Ways)

### 1. Instant — just open the file
```bash
git clone https://github.com/diazMelgarejo/agentra-dingbot.git
cd agentra-dingbot
open docs/index.html        # macOS
xdg-open docs/index.html    # Linux
start docs/index.html       # Windows
```
Opens in **demo mode** with simulated live-updating data. No backend needed.

### 2. Served locally (avoids file:// quirks)
```bash
# Python (no install)
python -m http.server 8000 --directory docs
# → http://localhost:8000

# or Node
npx serve docs
```

### 3. Fully live — connect to the real backend
```bash
# Terminal 1 — start the FastAPI backend (pushes real cycle state over WebSocket)
LLM_PROVIDER=none python src/dashboard/app.py
# → serves on http://localhost:8000, WebSocket at /ws/signals

# Terminal 2 — serve the dashboard (or just open docs/index.html)
python -m http.server 8080 --directory docs
```
The dashboard auto-detects the backend: it attempts `ws://localhost:8000/ws/signals`
on load (2.5s timeout). On success the badge flips to **Live (green)** and real
data flows in. On failure it stays in **Demo (amber)** mode.

---

## GitHub Pages Deployment

We use **GitHub-native Pages actions** (`actions/configure-pages`,
`actions/upload-pages-artifact`, `actions/deploy-pages`) — *not* third-party
deploy actions. The workflow lives at `.github/workflows/pages.yml`.

Because the site is already static, the workflow does **no build** — it uploads
`/docs` as the Pages artifact and deploys it. This is the orama/perpetua-style
GitHub-native pattern.

### One-time setup (repo owner)
1. Repo **Settings → Pages → Build and deployment → Source: GitHub Actions**
2. Push to `main` — the `pages.yml` workflow deploys automatically
3. Site goes live at `https://diazmelgarejo.github.io/agentra-dingbot/`

(Alternatively: Settings → Pages → Source: *Deploy from a branch* → `main` → `/docs`.)

---

## Remaining Work (Tracked)

Adopted from the review's "Remaining Work" list, refined for the static approach:

- [x] **Live data binding** — `render(d)` now binds `d.polymarket_markets` (list →
      pm-body rows) and `d.risk` (position_size_pct / stop_loss_pct / take_profit_pct /
      risk_reward_ratio) and `d.vix_risk_level` into the Risk Assessment table.
      Falls back to demo HTML when no live data. XSS-safe via `escHtml()`.
- [x] **Snapshot export mode** — `_run_cycle()` calls `_write_snapshot()` after each
      cycle, writing `docs/data/latest.json`. The dashboard's `tryLive()` tries the
      JSON as a 30-second polled fallback between WebSocket and demo mode. Shows
      "Snapshot" badge. `docs/data/` created and tracked with `.gitkeep`.
- [x] **Responsive QA** — breakpoints tested by Playwright at 1280px, 1099px, 560px,
      and 375px (iPhone SE). All pass: sidebar visible at desktop, content accessible
      at narrow viewports, KPIs visible at 375px. (`make test-ui`)
- [x] **Read-only API gating** — `WS_READ_TOKEN` env var gates `/ws/signals`. Empty
      (default) = no auth required. When set, connections without `?token=<value>` or
      with wrong token receive `{"type":"error","code":4401}` and are closed. Token
      captured at `create_app()` time so it survives process lifetime.
- [x] **FreqUI link** — implemented as link-out (not iframe). JS sets
      `href='http://localhost:8080/ui/'` in live mode and `href='#'` in demo/snapshot
      mode. CSP blocks cross-origin iframes by default; link-out is the right choice.
- [x] **Playwright smoke test** — `tests/ui/test_dashboard_ui.py` (17 tests, `make
      test-ui`). Covers: KPI rendering, demo fallback badge, theme toggle round-trip +
      localStorage persistence, chart canvas mount, responsive breakpoints at 4 sizes.
      Isolated from asyncio suite in `tests/ui/` (`norecursedirs` in pytest.ini).
- [ ] **"How to extend the dashboard" guide** — short `docs/README.md` section for the
      council on adding a panel (where the render() map lives, how to add a KPI card).
- [ ] **Lightweight analytics** — optional privacy-friendly pageview beacon for the
      public Pages site (or omit entirely to stay dependency-free).

---

## Future: Build Pipeline (Only If Needed)

If the dashboard outgrows a single file, migrate to:
```
src/dashboard/frontend/     ← Vite + React + TypeScript + Tailwind
  src/components/           ← SignalCard, AgentCards, CandlestickChart, ...
  vite.config.ts            ← base: '/agentra-dingbot/', outDir: '../../../docs'
```
Then `pages.yml` gains a `setup-node` + `npm ci && npm run build` step before the
upload-artifact step. Until then, the static file is the right tool.
