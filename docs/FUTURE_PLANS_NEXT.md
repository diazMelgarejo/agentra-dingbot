# Future Plans & Refinement Backlog

*Tracked for manual review. Updated after the snapshot/council/live-binding session.*

---

## Just completed (this branch)
- ✅ Live-data binding (WebSocket emits dashboard shape via `state_view.py`)
- ✅ Snapshot export (`snapshot_export.py`) + daily cron (`snapshot.yml`)
- ✅ Three-tier dashboard fallback (live → snapshot → demo)
- ✅ Portfolio config (`portfolio.yaml` + validated loader)
- ✅ Council prompt template (`docs/COUNCIL_PROMPT.md`)

---

## Near-term refinements (dashboard / Pages)

| Item | Why | Effort |
|------|-----|--------|
| **Enable GitHub Pages** (manual click — PAT can't) | Public dashboard goes live | 1 min |
| **Wire real Polymarket markets into `polymarket_markets`** | Snapshot/live PM panel currently sample data on the live path until the agent populates the field | S |
| **Snapshot freshness indicator** | Warn if `meta.generated_at` is > 48h old (stale snapshot) | S |
| **Playwright smoke test** on built `/docs` | Assert KPIs render, theme toggle works, chart mounts, 3-tier fallback fires | M |
| **Responsive QA on real devices** | Sidebar collapse @1100px + single-col @560px are coded but untested | S |
| **FreqUI embed decision** | iframe (CSP issues) vs link-out from the sidebar | S |

---

## Portfolio / council (item 4 follow-ups)

| Item | Why | Effort |
|------|-----|--------|
| **`portfolio_agent` node** | Make a LangGraph node that loads `portfolio.yaml` and emits sleeve targets into `TradingState` | M |
| **Council runner script** | `scripts/run_council.py` that feeds a filing blob through `COUNCIL_PROMPT.md` to Ollama/LM Studio and validates the YAML | M |
| **Wire `min_reward_to_risk` into risk_manager** | Tactical sleeve should enforce ≥3:1 at order time | S |
| **Backtest the regime allocation** | Walk-forward the sleeve weights vs. buy-and-hold over 2024–2026 | L |
| **Correlation-cluster guard** | Enforce "max 3 correlated positions" (mega-cap tech + BTC proxies cluster) | M |

---

## Carried over from prior sessions

- **Live data binding for snapshot's `cycle` mode** — verify the real Polymarket + risk
  fields populate when run against live APIs (works structurally; needs a network run).
- **Read-only API gating** if the backend is ever exposed beyond localhost.
- **"How to extend the dashboard" guide** — short section on adding a panel / KPI.
- **Lightweight privacy-friendly analytics** for the public Pages site (or omit).

---

## Larger horizon (from docs/FUTURE_PLANS.md)
- LLaVA visual agent (5th analyst on chart screenshots)
- On-chain expansion (MVRV, exchange flows, OI)
- Multi-symbol ETH support end-to-end
- PostgreSQL + TimescaleDB for trade history
- Grafana monitoring alongside the static dashboard
- FreqAI hyperparameter tuning

---

## Operational reminders
- ⚠️ **Revoke any exposed GitHub PAT** at https://github.com/settings/personal-access-tokens
- Snapshot cron runs daily ~13:17 UTC; admin can trigger on demand via the Actions tab
  (`workflow_dispatch`, mode=cycle for real data).
- Scheduled workflows can lag 10–30 min (GitHub platform behavior) — not time-critical.
