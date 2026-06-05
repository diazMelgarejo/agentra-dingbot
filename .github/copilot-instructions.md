# Copilot Instructions — Agentra DingBot

## Project context
Multi-agent crypto + Polymarket trading bot. Python 3.12. All source under `src/`.
pytest.ini sets `pythonpath = src` — import as `from core.config import ...` not `from src.core.config import ...`.

## TDD required
ALWAYS write failing tests before writing code.
1. Write test → confirm it fails → commit `test: RED — <feature>`
2. Write minimal code → confirm tests pass → commit `feat: GREEN — <feature>`
3. Refactor → tests stay green → commit `refactor: REFACTOR — <feature>`

## Key patterns
- LangGraph nodes receive STATE OBJECT, return partial dict
- Use shallow dict in _wrap(): `{f.name: getattr(state,f.name) for f in fields(state)}`
- NEVER use `dataclasses.asdict()` — it deep-converts nested objects
- `errors` field uses `Annotated[List[str], operator.add]` reducer (concurrent fan-out)
- Routing keys are strings: `"execute"` / `"skip"` not bool
- Default `LLM_PROVIDER=none` → heuristic judge (no key required)
- Safety: ALWAYS check `KillSwitch.is_armed()` before orders
- Safety: ALWAYS use `validate_order()` before CCXT calls
- Safety: `is_live_trading_enabled()` must return True for any live order

## Never do
- `import freqtrade` — GPL-3.0, HTTP only (see docs/LICENSE_RATIONALE.md)
- `LIVE_TRADING=true` without explicit user confirmation
- Hardcode Polymarket fee rates (fetch feeRateBps dynamically)
- Use `dataclasses.asdict()` in the orchestrator _wrap function
- Write code before failing tests
