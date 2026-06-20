# Session Progress — Snapshot, Council & Live-Binding Branch

**Branch:** `2026-06-20-snapshot-council-livebind`
**Base:** `main` @ `8c17752` (Step 8 dashboard + Pages)
**Tests:** 307 → **321 passing**, 6 skipped (+14 new)

This session closed the remaining Step 8 follow-ups and added the filing-driven
portfolio framework requested from the sample trade-plan spec.

---

## What shipped

### 1. Live-data binding (dashboard ↔ backend)
- New `src/dashboard/state_view.py` — `to_dashboard_view()` is the **single source
  of truth** for the dashboard JSON shape. Both the WebSocket push and the snapshot
  exporter call it, so live and Pages render identical structures.
- `src/dashboard/app.py` WebSocket now emits the dashboard shape (consensus, all four
  agents, risk assessment, Polymarket markets, OHLCV) instead of a raw state dump.
  Push cadence is env-tunable via `DASHBOARD_PUSH_SECONDS`.
- Maps the real `TradingState` fields: `technical/sentiment/onchain/ml` snapshots,
  `RiskAssessment` (position size, SL, TP, VIX level), `polymarket_markets`.

### 2. Snapshot export → GitHub Pages shows real data
- New `src/dashboard/snapshot_export.py` — writes `docs/data/snapshot.json` with a
  `meta` envelope (`generated_at`, `mode`, `symbol`, `git_sha`, `schema`).
  - `--mode cycle` runs one real LangGraph cycle (zero-key, dry-run).
  - `--mode demo` writes a realistic no-network sample (CI-safe default).
- New `.github/workflows/snapshot.yml` — daily cron + `workflow_dispatch`. Commits the
  snapshot back using the default `GITHUB_TOKEN` (no infinite loop — token commits
  don't re-trigger workflows), with a `git diff --staged --quiet` guard and `[skip ci]`.
- Dashboard upgraded to a **three-tier fallback**:
  `live WebSocket → snapshot.json poll → demo`. The snapshot poll uses cache-busting
  (`?t=Date.now()` + `cache:'no-store'`) and an `res.ok` check, per fetch() semantics.
  New blue **Snapshot** badge + banner showing the export timestamp. It keeps polling,
  so an admin re-running the exporter upgrades the public page automatically.

### 3. Filing-driven portfolio framework (from the sample trade-plan)
- New `config/portfolio.yaml` — multi-sleeve regime/factor allocation:
  core_beta 60% · ai_infrastructure 25% · bitcoin_sleeve 10% · tactical_swing 5%,
  plus hard risk caps (25% per name, 2% per trade, 6% heat, ≥3:1 R:R) and the
  June-22 rebalance / July-14 CPI catalyst calendar.
- New `src/strategies/portfolio_config.py` — pydantic loader with **cross-field
  validation**: weights sum to 1.0, sub-mix sums to sleeve weight, and a
  concentration cap that correctly applies to **individual holdings** (not diversified
  sleeves — a bug the validator caught during the build).
- New `docs/COUNCIL_PROMPT.md` — reusable council prompt template (regime detector →
  AI-tilt detector → allocator → swing/risk validator) with a machine-readable YAML
  output contract and a `yaml.safe_load` + pydantic parsing snippet.

---

## TDD evidence
- `tests/test_portfolio_config.py` — written RED first (6 failing), then GREEN.
  The concentration-cap test surfaced a real logic bug (sleeve vs. holding), fixed
  before GREEN.
- `tests/test_dashboard_snapshot.py` — 8 tests for the view mapper + exporter envelope.

---

## New Make targets
```bash
make snapshot        # write demo snapshot.json
make snapshot-live   # run a real cycle and export
make portfolio       # validate + print portfolio.yaml
```

---

## Not copy-trading — explicit
The portfolio framework is **regime/factor allocation**, derived from the sample
spec's own guidance: *extract factor exposures and regime posture; do not mirror
disclosed trades or chase delayed filings.* Human approval + audit logging are
required before any implementation (encoded in the council prompt's governance block).
