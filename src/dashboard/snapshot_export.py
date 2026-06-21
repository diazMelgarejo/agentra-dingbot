"""
src/dashboard/snapshot_export.py  —  Static Snapshot Exporter
==============================================================
Writes a single JSON snapshot the public GitHub Pages dashboard polls.

The Pages site is static — it cannot reach a running backend. This module
produces `docs/data/snapshot.json` so the public dashboard can show *real*
data, refreshed once per session/day (or whenever an admin runs it / a cron
GitHub Action commits it).

Modes
-----
  --mode cycle   Run one real LangGraph cycle, export the resulting state.
                 (Needs network for OHLCV/Polymarket/F&G/VIX; zero-key OK.)
  --mode demo    Write a realistic demo snapshot (no network). Default fallback.

Usage
-----
    python -m dashboard.snapshot_export --mode cycle --symbol BTC/USDT
    python -m dashboard.snapshot_export --mode demo
    make snapshot                       # convenience target

Output shape: see dashboard/state_view.py (to_dashboard_view) plus a
`meta` envelope with generated_at, mode, symbol, and git_sha when available.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

logger = structlog.get_logger("snapshot")

# Default output path — served by GitHub Pages from /docs
DEFAULT_OUT    = _ROOT / "docs" / "data" / "snapshot.json"
LATEST_OUT     = _ROOT / "docs" / "data" / "latest.json"     # polled by Pages dashboard


# ── Demo snapshot (no network) ────────────────────────────────────────────────

def _demo_view(symbol: str = "BTC/USDT") -> dict[str, Any]:
    """A realistic, self-consistent demo snapshot (matches dashboard shape)."""
    import random

    # deterministic-ish base with small jitter so refreshes look alive
    rng = random.Random(int(datetime.now(UTC).timestamp()) // 3600)
    base = 67000 + rng.uniform(-1500, 1500)
    candles = []
    now = int(datetime.now(UTC).timestamp())
    price = base
    for i in range(140, 0, -1):
        drift = (rng.random() - 0.46) * 180
        o = price
        c = max(50000, o + drift)
        candles.append({
            "time": now - i * 3600,
            "open": round(o, 2), "high": round(max(o, c) + rng.random() * 120, 2),
            "low": round(min(o, c) - rng.random() * 120, 2), "close": round(c, 2),
        })
        price = c

    fng = int(28 + rng.uniform(-6, 6))
    prob = min(0.9, max(0.3, 0.67 + (rng.random() - 0.5) * 0.06))
    vix = round(19.5 + (rng.random() - 0.5) * 1.5, 2)

    return {
        "symbol": symbol,
        "debate_consensus": "STRONG_BUY",
        "debate_confidence": round(min(0.95, max(0.5, 0.72 + (rng.random() - 0.5) * 0.06)), 3),
        "technical": {"signal": "BUY", "rsi_14": round(38.2 + (rng.random() - 0.5) * 4, 1),
                      "ema_cross": "BULL", "macd": round(0.0023 + (rng.random() - 0.5) * 0.001, 5)},
        "sentiment": {"signal": "BUY", "fear_greed_index": fng, "vix": vix},
        "onchain": {"signal": "NEUTRAL", "funding_rate": round(0.0001 + (rng.random() - 0.5) * 0.0001, 6)},
        "ml": {"signal": "BUY", "prob_up": round(prob, 4), "model_type": "sklearn_hgb"},
        "risk": {"approved": True, "position_size_pct": 10.8, "stop_loss_pct": 2.0,
                 "take_profit_pct": 5.0, "vix_risk_level": "NORMAL"},
        "polymarket": [
            {"question": "BTC Up in next 5 min?", "yes_price": 0.54, "our_prob": round(prob, 2)},
            {"question": "BTC above $67,500 by :00?", "yes_price": 0.48, "our_prob": 0.51},
            {"question": "BTC Up in next 15 min?", "yes_price": 0.57, "our_prob": 0.59},
        ],
        "ohlcv_4h": candles,
    }


# ── Live cycle snapshot ───────────────────────────────────────────────────────

async def _cycle_view(symbol: str) -> dict[str, Any]:
    """Run one real orchestrator cycle and map to the dashboard shape."""
    from core.orchestrator import run_one_cycle
    from dashboard.state_view import to_dashboard_view

    logger.info("snapshot_cycle_start", symbol=symbol)
    state = await run_one_cycle(symbol, dry_run=True)
    view = to_dashboard_view(state)
    logger.info("snapshot_cycle_done", symbol=symbol,
                consensus=view.get("debate_consensus"))
    return view


# ── Envelope + write ──────────────────────────────────────────────────────────

def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(_ROOT),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _wrap(view: dict[str, Any], mode: str, symbol: str) -> dict[str, Any]:
    """Add a meta envelope around the dashboard view."""
    return {
        "meta": {
            "generated_at": datetime.now(UTC).isoformat(),
            "mode": mode,
            "symbol": symbol,
            "git_sha": _git_sha(),
            "schema": "agentra.dashboard.v1",
        },
        "data": view,
    }


def write_snapshot(payload: dict[str, Any], out: Path = DEFAULT_OUT) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    logger.info("snapshot_written", path=str(out), bytes=out.stat().st_size)
    return out


async def _main_async(mode: str, symbol: str, out: Path) -> int:
    if mode == "cycle":
        try:
            view = await _cycle_view(symbol)
        except Exception as exc:
            logger.error("cycle_failed_falling_back_to_demo", error=str(exc))
            view = _demo_view(symbol)
            mode = "demo-fallback"
    else:
        view = _demo_view(symbol)

    payload = _wrap(view, mode, symbol)
    write_snapshot(payload, out)
    # Always keep latest.json in sync (dashboard polls this path)
    if out != LATEST_OUT:
        write_snapshot(payload, LATEST_OUT)
    print(f"✅ snapshot written: {out}  (mode={mode}, "
          f"consensus={view.get('debate_consensus')})")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Export dashboard snapshot JSON")
    p.add_argument("--mode", choices=["cycle", "demo"], default="demo",
                   help="cycle = run real pipeline; demo = no-network sample")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    return asyncio.run(_main_async(args.mode, args.symbol, Path(args.out)))


if __name__ == "__main__":
    raise SystemExit(main())
