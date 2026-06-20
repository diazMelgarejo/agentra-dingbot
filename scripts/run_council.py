#!/usr/bin/env python3
"""
scripts/run_council.py  —  Filing-Driven Council Runner
========================================================
Feeds a market context blob through docs/COUNCIL_PROMPT.md to a council
of agents (Ollama, LM Studio, OpenAI, or Anthropic) and produces a
validated machine-readable YAML allocation plan.

The council does NOT copy individual trades. It extracts regime posture and
factor allocations from the filing. Human approval is required before any
implementation.

Usage
-----
    # Demo run (no LLM needed — shows the prompt + schema)
    python scripts/run_council.py --mode demo

    # Ollama (local, free)
    python scripts/run_council.py --backend ollama --model llama3.1:8b

    # LM Studio (local, free)
    python scripts/run_council.py --backend lmstudio --model "your-model"

    # Anthropic (cloud)
    python scripts/run_council.py --backend anthropic --model claude-sonnet-4-6

    # OpenAI-compatible
    python scripts/run_council.py --backend openai --model gpt-4o

    # With a custom market context file
    python scripts/run_council.py --context my_filing.md --backend ollama
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

COUNCIL_PROMPT_PATH = _ROOT / "docs" / "COUNCIL_PROMPT.md"
PORTFOLIO_YAML_PATH = _ROOT / "config" / "portfolio.yaml"

# ── Default demo market context (matches the 2026 filing sample) ─────────────
_DEMO_CONTEXT = """
Filing analysis — 2026 mid-year institutional allocation review:

  net_buying_vs_selling: net buying (positive ratio)
  sector_tilt: large-cap tech and AI infrastructure
  breadth: broad S&P 500 + Nasdaq index exposure
  management: delegated / professional (thousands of trades, factor process)
  macro_calendar:
    - "Nasdaq-100 quarterly rebalance: effective 2026-06-22"
    - "June CPI release: 2026-07-14 08:30 ET"
  regime_signals:
    vix: 19.5
    fear_greed: 28        # Fear — contrarian bullish in up-trend
    trend: up             # price above 200-day MA
    btc_etf_flows: positive last 5 sessions (crypto ETF inflows sustained)
  constraint: Do not mirror individual disclosed trade names or sizes.
"""


# ── Parse council output ───────────────────────────────────────────────────────

def parse_council_output(text: str) -> dict:
    """Extract and validate the fenced YAML block from the model's response."""
    m = re.search(r"```(?:ya?ml)?\s*(?P<y>.*?)```", text, re.DOTALL)
    raw = m.group("y") if m else text
    plan = yaml.safe_load(raw)

    # Validate required top-level keys
    required = {"regime", "tilt", "allocation", "swing", "risk"}
    missing = required - set(plan.keys())
    if missing:
        raise ValueError(f"Council output missing keys: {missing}")

    # Allocation must sum to 1.0
    alloc = plan["allocation"]
    total = sum(alloc.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Allocation weights sum to {total:.4f}, not 1.0")

    # Cross-check against portfolio_config.yaml if available
    # Portfolio swing gate: all 4 conditions required when authorized
    swing_authorized = PORTFOLIO_YAML_PATH.exists() and plan["swing"].get("authorized")
    if swing_authorized:
            conditions = plan["swing"].get("conditions_met", [])
            if len(conditions) < 4:
                print(f"⚠️  Swing authorized but only {len(conditions)}/4 conditions met: {conditions}")

    return plan


# ── LLM backends ──────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, model: str, host: str = "http://localhost:11434") -> str:
    import urllib.request
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(f"{host}/api/generate",
                                  data=payload, method="POST",
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r).get("response", "")


def _call_lmstudio(prompt: str, model: str, host: str = "http://localhost:1234") -> str:
    import urllib.request
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }).encode()
    req = urllib.request.Request(f"{host}/v1/chat/completions",
                                  data=payload, method="POST",
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)
        return data["choices"][0]["message"]["content"]


def _call_anthropic(prompt: str, model: str) -> str:
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic") from None
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model, max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text


def _call_openai(prompt: str, model: str, base_url: str | None = None) -> str:
    try:
        import openai
    except ImportError:
        raise ImportError("pip install openai") from None
    kwargs = {"base_url": base_url} if base_url else {}
    client = openai.OpenAI(**kwargs)
    resp = client.chat.completions.create(
        model=model, max_tokens=1500, temperature=0.1,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content


# ── Build the full prompt ──────────────────────────────────────────────────────

def build_prompt(context: str) -> str:
    """Inject the market context into the council prompt template."""
    template = COUNCIL_PROMPT_PATH.read_text()
    # Replace the placeholder block with the actual context
    template = re.sub(
        r"```\n<PASTE FILING SUMMARY OR MARKET BLOB HERE>.*?```",
        f"```\n{context.strip()}\n```",
        template, flags=re.DOTALL
    )
    return template


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Run the portfolio allocation council")
    p.add_argument("--backend", choices=["ollama", "lmstudio", "anthropic", "openai", "demo"],
                   default="demo")
    p.add_argument("--model", default="llama3.1:8b")
    p.add_argument("--host", default=None, help="Override API host URL")
    p.add_argument("--context", default=None, help="Path to market context file (.md or .txt)")
    p.add_argument("--out", default=None, help="Write plan JSON to this path")
    args = p.parse_args()

    # Load context
    if args.context:
        context = Path(args.context).read_text()
        print(f"📄 Context from: {args.context}")
    else:
        context = _DEMO_CONTEXT
        print("📄 Using built-in demo context (Jun 2026 filing sample)")

    prompt = build_prompt(context)

    if args.backend == "demo":
        print("\n" + "─"*60)
        print("DEMO MODE — prompt preview (first 800 chars):")
        print("─"*60)
        print(prompt[:800])
        print("─"*60)
        print("\nTo run a real council:")
        print("  Ollama:     python scripts/run_council.py --backend ollama --model llama3.1:8b")
        print("  LM Studio:  python scripts/run_council.py --backend lmstudio")
        print("  Anthropic:  python scripts/run_council.py --backend anthropic --model claude-sonnet-4-6")
        return 0

    print(f"\n🤖 Calling {args.backend} / {args.model}…")
    try:
        if args.backend == "ollama":
            raw = _call_ollama(prompt, args.model, args.host or "http://localhost:11434")
        elif args.backend == "lmstudio":
            raw = _call_lmstudio(prompt, args.model, args.host or "http://localhost:1234")
        elif args.backend == "anthropic":
            raw = _call_anthropic(prompt, args.model)
        elif args.backend == "openai":
            raw = _call_openai(prompt, args.model, args.host)
        else:
            raise ValueError(f"Unknown backend: {args.backend}")
    except Exception as exc:
        print(f"❌ LLM call failed: {exc}")
        return 1

    print("\n📋 Raw response:")
    print("─"*60)
    print(raw[:2000])
    print("─"*60)

    try:
        plan = parse_council_output(raw)
        print("\n✅ Plan validated:")
        print(json.dumps(plan, indent=2))
        if args.out:
            Path(args.out).write_text(json.dumps(plan, indent=2))
            print(f"\n💾 Saved to {args.out}")
        print("\n⚠️  HUMAN APPROVAL REQUIRED before any implementation.")
        print("   Audit log this run and review the allocation with a risk manager.")
        return 0
    except Exception as exc:
        print(f"\n❌ Validation failed: {exc}")
        print("Re-run or manually review the raw response above.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
