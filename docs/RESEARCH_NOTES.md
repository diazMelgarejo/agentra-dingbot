# Research Notes — Best Practices (June 2026)
*Saved from the extended research session before the snapshot/council build.*
*Non-code answers and technical findings for continuity across sessions.*

---

## 1. GitHub Actions Snapshot Export — Loop-Safe Pattern

### The core question
How do you commit a file back to the repo from a GitHub Action (cron job) without
creating an infinite push → trigger → push loop?

### The answer
**Use the default `GITHUB_TOKEN` — its commits never re-trigger workflows.**

GitHub's platform guarantees: "commits made by this Action do not trigger new
Workflow runs" when using the default token. The infinite loop is caused by PAT
commits, not GITHUB_TOKEN commits.

### Correct workflow structure

```yaml
permissions:
  contents: write         # REQUIRED — otherwise push is rejected

jobs:
  export:
    steps:
      - uses: actions/checkout@v4   # uses GITHUB_TOKEN by default

      # ... generate the file ...

      - name: Commit if changed
        run: |
          git config user.name  "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add docs/data/snapshot.json
          git diff --staged --quiet || git commit -m "chore: update snapshot [skip ci]"
          git push
```

### If you must use a PAT (e.g., to push to protected branches)
The loop protection is gone. Add ALL three defenses:
1. `[skip ci]` in the commit message (GitHub natively skips for all `[skip ci]` variants)
2. `paths-ignore: ['docs/data/**']` on the `on: push` trigger
3. `if: github.actor != 'github-actions[bot]'` job condition

`paths-ignore` alone is NOT reliable — it inspects only the current push's files
and has edge cases. Always use `[skip ci]` as primary + `paths-ignore` as backup.

### Keep-alive warning
Scheduled workflows are **auto-disabled after 60 days with no repo activity**.
Only commits, PRs, and issues reset the clock (not releases). The snapshot commit
itself resets the clock, so this workflow is self-sustaining.

---

## 2. GitHub Pages Static Polling Pattern

### The fetch() behavior quirk (critical)
`fetch()` does **NOT** throw or reject on HTTP errors (404, 403, 500).
It resolves with a `Response` object where `response.ok === false`.
**You must check `res.ok` explicitly — otherwise a 404 for a missing `snapshot.json`
silently produces a resolved promise that your code treats as valid data.**

```javascript
async function pollSnapshot() {
  try {
    // cache-bust: unique URL per poll so browser AND CDN serve fresh data
    const res = await fetch(`data/snapshot.json?t=${Date.now()}`, {
      cache: 'no-store'    // instructs browser to "totally ignore HTTP-cache"
    });
    if (!res.ok) {         // 404 → false, not an exception
      return false;        // fall back to demo data
    }
    const data = await res.json();
    render(data.data || data);
    return true;
  } catch (err) {
    // ONLY network errors land here (DNS failure, CORS blocked, etc.)
    return false;
  }
}
```

### Why both `?t=Date.now()` AND `cache:'no-store'`?
- `?t=Date.now()` — forces a unique URL, bypassing CDN cache
- `cache:'no-store'` — instructs the browser to not cache the response at all
- GitHub Pages + browsers serve static files aggressively; `no-store` alone isn't
  always honored by all CDNs; `?t=` is the belt-and-suspenders approach

### setTimeout vs setInterval for polling
Use **recursive `setTimeout`** not `setInterval`:
```javascript
async function poll() {
  await doWork();
  setTimeout(poll, 30000);   // next poll starts AFTER current completes
}
poll();                       // boot
```
`setInterval` fires regardless of whether the previous request is done → requests
pile up under latency. `setTimeout` guarantees only one request is in flight.

### CDN cache vs browser cache
- `cache:'no-store'` covers the **browser** cache
- GitHub Pages CDN cache is partially outside your control
- For a data file changed frequently (`snapshot.json`), the `?t=` param is what
  forces the CDN to fetch fresh — it sees it as a new resource each time

---

## 3. LangGraph Multi-Agent Council Patterns

### Single-file "council prompt" structure
One markdown file, one `##` header per role, shared output contract:

```markdown
## Role 1 — Regime Detector
Classify the regime as risk_on | risk_off | volatile.

## Role 2 — AI Infrastructure Tilt Detector
Decide whether to overweight compute/cloud names.

## Role 3 — Allocator
Propose sleeve weights summing to 1.0.

## Role 4 — Risk Validator
Check weights against caps. Set risk.approved: false if any cap breached.

## Output Contract
Respond ONLY with YAML in this exact shape:
```yaml
regime: ...
allocation: ...
risk: ...
```
```

### Why single-file works
- Fewer API calls (single prompt vs. multi-turn)
- Easier to version-control (one file to diff)
- LLMs can hold context across all four roles in one pass
- For dynamic re-deliberation (risk validator loops back to allocator), graduate
  to a LangGraph supervisor graph

### YAML parsing — always safe_load + pydantic

```python
import re, yaml
from pydantic import BaseModel

def parse_council(text: str) -> CouncilOutput:
    m = re.search(r"```(?:ya?ml)?\s*(?P<y>.*?)```", text, re.DOTALL)
    raw = m.group("y") if m else text
    data = yaml.safe_load(raw)            # safe_load — never yaml.load
    return CouncilOutput.model_validate(data)  # raises on bad shape
```

`yaml.load` without Loader is a code execution vulnerability.
Wrap in pydantic for typed validation with helpful error messages.

### LangGraph supervisor routing

```python
from typing import Literal
from typing_extensions import TypedDict
from langgraph.types import Command

class Router(TypedDict):
    next: Literal["regime_detector", "allocator", "risk_validator", "FINISH"]
    reasoning: str    # explicit CoT improves routing accuracy

def supervisor(state):
    decision = llm.with_structured_output(Router).invoke(messages)
    goto = END if decision["next"] == "FINISH" else decision["next"]
    return Command(goto=goto, update={"next": goto})
```

Use `with_structured_output` so the supervisor cannot emit an invalid destination.
Add a `reasoning` field to improve routing decisions (forces the model to reason
before deciding).

### Anthropic prefill caveat (June 2026)
Response prefilling (forcing model to begin with ```` ```yaml ````) is reportedly
disabled on newer Anthropic models. Rely on explicit system-prompt format instructions
instead. Verify at docs.anthropic.com for your specific model.

---

## 4. Pydantic-Settings YAML Config Pattern

### The most common mistake
`yaml_file=...` in `SettingsConfigDict` alone is **not enough**. You must also
override `settings_customise_sources` to include `YamlConfigSettingsSource`.
Without the override, the YAML file is ignored silently.

### Correct implementation

```python
from pydantic_settings import (
    BaseSettings, SettingsConfigDict,
    PydanticBaseSettingsSource, YamlConfigSettingsSource,
)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(yaml_file="config.yaml")

    @classmethod
    def settings_customise_sources(
        cls, settings_cls,
        init_settings, env_settings, dotenv_settings, file_secret_settings,
    ):
        # YAML is included; env vars can override if placed before YAML
        return (init_settings, env_settings, YamlConfigSettingsSource(settings_cls),
                dotenv_settings, file_secret_settings)
```

Source order = precedence. `env_settings` before YAML → env overrides YAML.
YAML before `env_settings` → YAML is authoritative.

### Cross-field validation

```python
from pydantic import model_validator

class PortfolioConfig(BaseSettings):
    layers: dict[str, Sleeve]
    risk: RiskCaps

    @model_validator(mode="after")
    def _validate_weights_and_caps(self):
        total = sum(s.weight for s in self.layers.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")
        return self
```

Use `@model_validator(mode="after")` for cross-field rules.
Use `@field_validator` for single-field rules.
Use `Field(ge=0, le=1)` for range constraints.

### Concentration cap subtlety
The 25% cap applies to **individual holdings**, not to diversified sleeves.
A 60% "core" sleeve internally split 25%+20%+10%+5% has no name over 25%.
The validator must check sub-mix positions, not the sleeve aggregate.
(This bug was caught and fixed during the portfolio_config.py build.)

---

## 5. Design Decisions Q&A

### Q: Why a single static HTML file instead of React/Vite?
**A**: The core requirement was "local demo available for immediate download."
A single self-contained `docs/index.html` satisfies this — `open docs/index.html`
just works with no build step, no npm, no Node. GitHub Pages serves it directly
from `/docs` with no CI build. If the dashboard outgrows a single file, migrate
to Vite with `outDir: '../../docs'` and add a `setup-node + npm ci + npm run build`
step to `pages.yml`.

### Q: Why not peaceiris/actions-gh-pages for deployment?
**A**: The review document explicitly said "follow orama and perpetua defaults,
not the example shown here." GitHub-native actions (`configure-pages` +
`upload-pages-artifact` + `deploy-pages`) are the canonical pattern — no
third-party action, minimal permissions surface.

### Q: Why default GITHUB_TOKEN and not a PAT for the snapshot workflow?
**A**: GITHUB_TOKEN commits don't re-trigger workflows (GitHub platform guarantee).
PAT commits DO re-trigger, creating an infinite push→trigger→push loop.
Fine-grained PATs also can't be revoked via API (only at github.com/settings).

### Q: Why heuristic judge as default instead of requiring an LLM?
**A**: Every feature that requires an external service must have a fallback.
The heuristic vote (technical×1.0, ml×0.9, sentiment×0.7, onchain×0.5) makes
the bot fully functional with no keys, no local LLM, no Docker. "Zero config"
means the bot runs usefully with no environment variables set. Users upgrade
incrementally: LLM_PROVIDER=ollama → openai → anthropic.

### Q: Why module-level imports in data modules?
**A**: Mock patching with `patch("module.attribute")` replaces the name binding
in the module's namespace at import time. Lazy imports inside functions create
a fresh local binding that the patch doesn't reach. All data modules must import
at module level for proper test isolation.

### Q: How does the Polymarket fee model work in 2026?
**A**: In January 2026 Polymarket introduced taker fees on 5-minute crypto markets.
The fee curve is a symmetric bell: `fee_fraction = fee_rate × 4 × p × (1-p)`.
This peaks at p=0.5 (~1.56% at current rates). At p=0.1 or p=0.9 the fee is
~0.28%. The 8% edge floor must be measured **net of fees** — the old 0.04%
hardcoded rate was 39× too low at the most-traded probability.

### Q: What is the Clarence design system?
**A**: A B2B SaaS design philosophy (Clarence Gio Bolonia, UI/UX designer):
- **Color encodes state, not decoration**: green=bullish/success, red=bearish/critical,
  amber=warning, grey=neutral/inactive
- **KPI values 2-3× larger than labels** for instant scannability
- **Layout**: left-sidebar + top-header + card grid (never full-page tables)
- **Charts**: candlestick for price, line for trends — never pie charts
- **Pill status badges** (not text labels alone)
- **Skeleton loading** not spinners
- Explicitly designed dark surfaces (not just CSS `invert`)
- Blue/navy palette (#2335FF SynthNexus primary)

---

## 6. Operational Notes

### Scheduled workflow lag
GitHub scheduled workflows are not guaranteed to run at the exact cron time.
Delays of 10–30 minutes are common; over an hour has been observed during peak load.
Do not depend on exact timing for anything time-critical. The `workflow_dispatch`
trigger allows manual on-demand runs from the Actions tab.

### 60-day scheduled workflow keep-alive
Scheduled workflows in public repos are auto-disabled when no repository activity
has occurred in 60 days. Only commits, PRs, and issues reset the clock (not releases).
The snapshot workflow's own commit resets the clock, making it self-sustaining.

### Fine-grained PAT limitations
- Cannot create repositories via API (need `Administration: write` scope)
- Cannot be revoked via API (only at github.com/settings/personal-access-tokens)
- Commits with PAT re-trigger `on: push` workflows → infinite loop risk
- Use default GITHUB_TOKEN for any Action that commits back to the repo

### Polymarket CLOB 250ms taker delay (2026)
A 250ms delay was introduced for taker orders in the crypto prediction markets.
Factor this into 5-minute signal timing — a signal generated at T=0 fills at
~T+0.25s, which is meaningful on a 300-second (5-min) candle.
