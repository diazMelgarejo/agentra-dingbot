# License Selection: Why Apache 2.0

## Recommendation: Keep Apache 2.0 ✅

The current `LICENSE` file (Apache 2.0) is the right choice. Here is the full
rationale, including the FreqTrade GPL-3.0 situation.

---

## Dependency License Audit

| Dependency | License | Copyleft? |
|------------|---------|-----------|
| LangGraph / LangChain | MIT | No |
| CCXT | MIT | No |
| pandas / numpy / scipy | BSD-3 | No |
| pandas-ta | MIT | No |
| TA-Lib (C library) | BSD | No |
| ta-lib-python wrapper | MIT | No |
| scikit-learn | BSD-3 | No |
| LightGBM | MIT | No |
| joblib | BSD-3 | No |
| pydantic / FastAPI | MIT | No |
| uvicorn | BSD-3 | No |
| aiohttp / yfinance | Apache-2.0 | No |
| py-clob-client | MIT | No |
| websockets | BSD-3 | No |
| **FreqTrade + FreqAI** | **GPL-3.0** | **Yes — if you copy its source code** |

All direct dependencies except FreqTrade are MIT, BSD, or Apache — all permissive
licenses fully compatible with Apache 2.0.

---

## The FreqTrade GPL-3.0 Situation

FreqTrade is licensed under GPL-3.0. This is a "copyleft" license, which means:

**If you copy or embed FreqTrade's source code** into this project → this project
must also become GPL-3.0 (the "viral" effect).

**If you call FreqTrade as a separate service** (via its REST API, CLI, or Docker
container) → GPL does NOT apply to your code. This is the standard interpretation
of the GPL "system library" boundary.

### Integration plan for Step 6 (FreqTrade as a service)

```
SuperBot (Apache 2.0)          FreqTrade Docker container (GPL-3.0)
──────────────────────         ────────────────────────────────────
debate_engine                  Listens on :8080
   ↓                               ↑
risk_manager                   /api/v1/forcebuy  ←── HTTP REST call
   ↓                               |
executor/agent.py  ──────────→  /api/v1/status
                                /api/v1/performance
```

The HTTP boundary is the legal and technical firewall. This is identical to how
thousands of projects use MySQL (GPL) or Redis (BSD → SSPL) via network calls
without inheriting the license.

**Conclusion**: integrate FreqTrade via its REST API only, never `import freqtrade`.
Apache 2.0 is preserved.

---

## Why Apache 2.0 Beats MIT for This Project

Both are permissive, but Apache 2.0 has two advantages that matter for a trading
system:

### 1. Patent Grant

Apache 2.0 includes an **explicit patent license**:

> "each Contributor hereby grants to You a perpetual, worldwide, non-exclusive,
> no-charge, royalty-free, irrevocable patent license to make, use, sell..."

MIT and BSD have no patent clause. For trading algorithms, which are in a field
with active patent activity (algorithmic trading, ML signal generation, etc.),
the patent grant protects you and anyone using your code.

### 2. Notice Preservation

Apache 2.0 requires attribution in derived works but in a way that's easy to
comply with (a `NOTICE` file). MIT is easier but provides less traceability when
code travels across organizations.

---

## What to Change in the LICENSE File

Nothing. The current `LICENSE` file is Apache 2.0. Keep it.

If you ever decide to:
- **Copy FreqTrade source code**: add a `SPDX-License-Identifier: GPL-3.0` header
  to those specific files and note the dual-licensing in `README.md`.
- **Commercialize / SaaS this**: Apache 2.0 is fine as-is. No contributor license
  agreement (CLA) is needed for personal projects.
- **Want maximum permissiveness** (e.g., others can use in closed-source products
  without any obligation): Apache 2.0 already allows this.

---

## SPDX Header to Add to Every .py File (Optional but Best Practice)

```python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Agentic SuperBot Contributors
```

---

## Quick Reference

| License | Patent Grant | Copyleft | Compatibility with our deps |
|---------|-------------|----------|----------------------------|
| MIT | No | No | ✅ Full |
| BSD-3 | No | No | ✅ Full |
| **Apache 2.0 (current)** | **Yes** | **No** | **✅ Full** |
| GPL-3.0 | Yes | Yes (viral) | ⚠️ Only if we copy GPL code in |
| LGPL-3.0 | Yes | Weak | ⚠️ Avoid |
| AGPL-3.0 | Yes | Network copyleft | ⚠️ Avoid for SaaS |
