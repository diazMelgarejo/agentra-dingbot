# agAIntra — Council Handoff Guide

> This doc wires the multi-agent council (antigravity, codex, cursor, ollama, LM Studio)
> to the orama-system universal-skill-protocol. Read `docs/ARCHITECTURE.md` first.
>
> Status: v1 playground (`diazMelgarejo/agentra-dingbot`). Target: `oramasys/againtra-platform`.
> **IMPORTANT: ALL trading work runs `--paper` / `dry_run` / backtest ONLY. Never live trades.**

---

## 1 — Stack roles and who does what

| Agent | Role | Canonical tool |
|-------|------|---------------|
| **Antigravity** | Design + UX specs, brainstorming, spec writing | `antigravity` CLI or IDE |
| **Codex** | Fast inline code generation, boilerplate, migrations | `codex` CLI |
| **Cursor** | IDE editing, multi-file refactors, PR review | Cursor IDE |
| **Ollama (Mac)** | Local inference for multi-light tasks (`qwen3.5:9b-nvfp4`) | `ollama` on `localhost:11434` |
| **LM Studio (Win)** | Heavy inference, code review, plan comparison | LM Studio on `192.168.254.104:1234` |
| **Claude (Opus/Sonnet)** | PLAN-mode orchestration, adversarial review, architecture decisions | Claude Code |

**Inference routing rule (orama universal-skill-protocol):**
- Code plans → GPT-5.5 (outsource)
- Code reviews → Gemini 3.1 Thinking (outsource)
- Plan merge/harmonize → Opus 4.8 PLAN-mode
- Heavy local inference → Win RTX3080 LM Studio first, Mac Ollama second
- Light inference / multi-agent parallel → Mac Ollama (`qwen3.5:9b-nvfp4`)

---

## 2 — Hard guardrails (ALL agents must enforce these)

| Guard | Rule |
|-------|------|
| **PAPER MODE ONLY** | All trading code runs `--paper` or `dry_run`. No live Binance or Polymarket orders. Never. |
| **No financial advice** | Role = platform migration + code review + backtest. Not investment advice. |
| **oramasys/* REFERENCE-ONLY** | `oramasys/againtra-platform` is a future planning scaffold. Push only to `diazMelgarejo/agentra-dingbot`. |
| **TDD first** | Write the failing test before any implementation (docs/v2/26 RED→GREEN→REFACTOR). |
| **Git identity** | Approved: `cyre <Lawrence@cyre.me>`, `cyre <diazMelgarejo@gmail.com>`, `Codex <codex@openai.com>`, `cursoragent@cursor.com`. |
| **No secrets in repo** | Never commit API keys, private keys, or `.env` files. Use GitHub Secrets or env vars. |

---

## 3 — Project state and what's done

Steps 1–2 complete per `README.md`:

| What | Files | Status |
|------|-------|--------|
| LangGraph dual pipeline (spot + polymarket) | `src/core/orchestrator.py` | ✅ |
| PerpetuaState / TradingState types | `src/core/state.py` | ✅ |
| 6 agents (technical, sentiment, onchain, debate, risk, executor) | `src/agents/*/agent.py` | ✅ |
| Data ingestion (CCXT, Polymarket, F&G, VIX, WebSocket) | `src/data/` | ✅ |
| Backtesting + Monte Carlo | `src/backtesting/` | ✅ |
| Paper broker + deploy | `deploy/` | ✅ |
| Safety gate (`safety.py`, KillSwitch, circuit breakers) | `src/agents/executor/safety.py` | ✅ |
| CI pipeline | `.github/workflows/ci.yml` | ✅ (this PR) |
| Security scan (pip-audit/safety) | `.github/workflows/ci.yml` | ✅ (this PR) |
| TA agent (Step 3) | `src/agents/technical_analyst/` | 🔜 |
| LangGraph integration test (Step 4) | `tests/test_langgraph_pipeline.py` | 🔜 |
| FreqAI bridge (Step 5) | `src/agents/ml_analyst/` | 🔜 |
| Risk tuning (Step 6) | Backtest validation | 🔜 |
| Executor dry-run suite (Step 7) | `tests/` | 🔜 |
| React dashboard (Step 8) | `dashboard/` | 🔜 |

---

## 4 — How to pick up a task (TDD protocol per docs/v2/26)

```bash
# 1. Confirm which step you're targeting (see table above)
# 2. Write the failing test FIRST
pytest tests/test_<step>.py -v  # must FAIL before you write code

# 3. Write minimal implementation
# 4. Run test again — must PASS
pytest tests/test_<step>.py -v

# 5. Run full suite to check regressions (paper mode only)
pytest --ignore=tests/live -v

# 6. Run CI locally before pushing
ruff check src tests
mypy src --ignore-missing-imports
pip-audit
```

**For Steps 3–8 in order:** TA agent validation → LangGraph E2E test → FreqAI bridge →
risk tuning backtest → executor dry-run → dashboard.

---

## 5 — Connecting to orama-system orchestration

When handing off to the orama-system autoresearcher heartbeat (docs/v2/25 §3):
- The heartbeat reads `docs/COUNCIL_HANDOFF.md` (this file) + `docs/PROGRESS.md` to pick
  the next task.
- It commits only to the `diazMelgarejo/agentra-dingbot` playground (never `oramasys/*`).
- It posts a gbrain log entry and updates `docs/PROGRESS.md` after each iteration.
- All financial/live-trade actions are blocked in heartbeat pulses (G2 guardrail).

**To trigger the next autoresearcher pulse manually:**
```bash
# From orama-system checkout:
gbrain put "againtra-heartbeat-trigger" <<'EOF'
{
  "type": "heartbeat_trigger",
  "repo": "diazMelgarejo/agentra-dingbot",
  "next_step": "Step 3 — TA agent validation",
  "mode": "paper"
}
EOF
```

---

## 6 — File map for the council

```
src/
├── core/
│   ├── orchestrator.py   ← LangGraph graph builder (future: MiniGraph swap target)
│   ├── state.py          ← TradingState + PolymarketDecision (future: PerpetuaState)
│   ├── config.py         ← Pydantic-settings singleton
│   └── cli.py            ← CLI entry point
├── agents/               ← one subdir per agent, agent.py in each
├── data/                 ← fetcher.py, polymarket.py, fear_greed.py, websocket_stream.py
├── backtesting/          ← backtest.py, monte_carlo.py, walk_forward.py
├── strategies/           ← technical_signals.py, fear_filter.py, hybrid_decision.py
└── agents/executor/
    └── safety.py         ← KillSwitch, equity circuit breaker, dry_run gate

tests/
├── test_data_ingestion.py   ← 26 passing (Step 2)
├── test_technical_analyst.py
└── test_risk_manager.py

docs/                        ← architecture, specs, handoff, lessons, progress
config/strategies.yaml       ← risk params (Kelly fraction, VIX levels, ATR)
deploy/live.py               ← async event loop (paper only)
deploy/paper_broker.py       ← paper trading simulator
```
