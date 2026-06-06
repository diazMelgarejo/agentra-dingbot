"""
src/deploy/live.py  —  Live / Paper Trading Entry Point
========================================================
Wraps the LangGraph pipeline in a continuous async loop.
Uses the real orchestrator (Steps 1–7), not the old HybridTrader.

Usage
-----
    # Paper mode (safe default — no real orders)
    python src/deploy/live.py

    # Explicit paper
    python src/deploy/live.py --paper --symbol BTC/USDT

    # Live mode (requires LIVE_TRADING=true env var AND typed confirmation)
    LIVE_TRADING=true python src/deploy/live.py --live

Safety gates (all enforced here, not just in executor)
-----
  1. LIVE_TRADING=true required for any real order
  2. Typed "LIVE" confirmation at runtime
  3. Kill switch checked at startup
  4. API key permission check at startup (no withdraw)
  5. Ghost position reconciliation at startup
  6. Daily equity-based circuit breaker (resets at midnight UTC)
  7. All file paths are absolute (no relative paths — breaks under cron)
"""
from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path

import structlog

# ── Absolute project root (safe under cron / systemd / tmux) ─────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent   # project root
sys.path.insert(0, str(_ROOT / "src"))

logger = structlog.get_logger("live")

# ── Midnight equity-reset ──────────────────────────────────────────────────────

_start_of_day_equity: float = 0.0
_daily_loss_limit_pct: float = 5.0   # overridden from config at startup

async def _midnight_reset(cfg) -> None:
    """Reset daily equity reference point at UTC midnight."""
    global _start_of_day_equity
    while True:
        now = datetime.now(UTC)
        secs = 86400 - (now.hour * 3600 + now.minute * 60 + now.second)
        await asyncio.sleep(secs)
        _start_of_day_equity = 0.0   # reset; will be sampled on next cycle
        logger.info("daily_equity_reset")


def _equity_circuit_fired(current_equity: float) -> bool:
    global _start_of_day_equity
    if _start_of_day_equity == 0.0:
        _start_of_day_equity = current_equity
        return False
    drop = (_start_of_day_equity - current_equity) / _start_of_day_equity
    fired = drop >= (_daily_loss_limit_pct / 100.0)
    if fired:
        logger.critical("equity_circuit_breaker_fired",
                        start=_start_of_day_equity, current=current_equity,
                        drop_pct=f"{drop:.1%}", limit_pct=f"{_daily_loss_limit_pct:.1%}")
    return fired


# ── Startup safety gates ──────────────────────────────────────────────────────

async def _startup_checks(live_mode: bool) -> bool:
    """Run all safety checks. Returns False if any critical issue found."""
    from agents.executor.safety import KillSwitch, is_live_trading_enabled, startup_safety_checks
    from core.config import get_settings

    get_settings()

    # 1. Kill switch
    ks = KillSwitch(flag_dir=str(_ROOT / "data"))
    if ks.is_armed():
        logger.critical("startup_blocked_kill_switch_armed")
        return False

    # 2. Mode consistency check
    env_live = is_live_trading_enabled()
    if live_mode and not env_live:
        logger.critical("startup_blocked",
                        msg="--live flag set but LIVE_TRADING env var not 'true'")
        return False

    # 3. Composite safety checks (API perms etc.) — skip for paper
    if live_mode:
        issues = await startup_safety_checks(exchange=None)
        critical = [i for i in issues if i.startswith("CRITICAL")]
        for issue in issues:
            logger.warning("startup_check", issue=issue)
        if critical:
            return False

    logger.info("startup_checks_passed", live_mode=live_mode)
    return True


async def _confirm_live() -> bool:
    """Interactive confirmation before enabling live trading."""
    print("\n" + "=" * 60)
    print("⚠️  LIVE TRADING MODE — Real funds will be used")
    print("=" * 60)
    print("Checklist before proceeding:")
    print("  [ ] API key is trade-only (no withdraw permission)")
    print("  [ ] IP whitelist is set on the exchange")
    print("  [ ] You have tested on paper for ≥1 week")
    print("  [ ] Daily loss limit and kill switch are understood")
    print()
    try:
        confirm = input("Type LIVE to confirm, anything else to abort: ")
    except (EOFError, KeyboardInterrupt):
        return False
    return confirm.strip() == "LIVE"


# ── Main cycle loop ────────────────────────────────────────────────────────────

async def run_loop(symbol: str, dry_run: bool, interval_seconds: int = 60) -> None:
    """Continuous cycle loop. Runs until interrupted or kill switch fires."""
    from agents.executor.safety import KillSwitch
    from core.orchestrator import run_one_cycle

    ks = KillSwitch(flag_dir=str(_ROOT / "data"))
    cycle = 0

    logger.info("loop_start", symbol=symbol, dry_run=dry_run, interval=interval_seconds)

    while True:
        # Kill switch checked every cycle
        if ks.is_armed():
            logger.critical("kill_switch_armed_halting")
            break

        cycle += 1
        start = asyncio.get_event_loop().time()

        try:
            result = await run_one_cycle(symbol, dry_run=dry_run)

            # Equity-based daily circuit breaker
            equity = float(result.get("equity", 0.0) or 0.0)
            if equity > 0 and _equity_circuit_fired(equity):
                logger.critical("daily_equity_limit_reached_halting")
                ks.arm()
                break

            order = result.get("order")
            consensus = result.get("debate_consensus")
            logger.info("cycle_done", cycle=cycle, symbol=symbol,
                        consensus=getattr(consensus, "value", str(consensus)),
                        order_placed=order is not None,
                        routed_via=getattr(order, "routed_via", None) if order else None)

        except KeyboardInterrupt:
            logger.info("interrupted_by_user")
            break
        except Exception as exc:
            logger.error("cycle_failed", cycle=cycle, error=str(exc))

        # Sleep for the remainder of the interval
        elapsed = asyncio.get_event_loop().time() - start
        sleep_for = max(0, interval_seconds - elapsed)
        await asyncio.sleep(sleep_for)

    logger.info("loop_ended", cycles=cycle)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main(live_mode: bool, symbol: str, interval: int) -> None:
    global _daily_loss_limit_pct
    from core.config import get_settings
    cfg = get_settings()
    _daily_loss_limit_pct = cfg.polymarket.daily_drawdown_limit_pct
    dry_run = not live_mode

    # Safety gate
    if live_mode and not await _confirm_live():
        print("Aborted.")
        return
    if not await _startup_checks(live_mode):
        print("Startup safety checks failed — see logs.")
        return

    # Midnight reset task
    loop = asyncio.get_running_loop()
    reset_task = loop.create_task(_midnight_reset(cfg))

    # Graceful shutdown handler
    _shutdown = asyncio.Event()
    def _sig_handler():
        logger.info("shutdown_signal_received")
        _shutdown.set()
        reset_task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _sig_handler)

    await run_loop(symbol=symbol, dry_run=dry_run, interval_seconds=interval)
    reset_task.cancel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentra DingBot — live/paper runner")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--paper", action="store_true", default=True, help="Paper mode (default)")
    group.add_argument("--live",  action="store_true", default=False, help="Live trading")
    parser.add_argument("--symbol",   default="BTC/USDT", help="Trading pair")
    parser.add_argument("--interval", type=int, default=60, help="Cycle interval seconds")
    args = parser.parse_args()

    live_mode = args.live
    asyncio.run(main(live_mode=live_mode, symbol=args.symbol, interval=args.interval))
