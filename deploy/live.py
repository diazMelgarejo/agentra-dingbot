"""
Live / Paper trading entry point.

Usage:
    python live.py --paper               # paper mode (default)
    python live.py --paper --symbol BTC  # explicit
    python live.py --live                # REAL money — use with caution!

The loop:
  - Runs HybridTrader main loop continuously
  - Resets circuit breaker at midnight UTC via scheduler
"""
import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone

# Make sure project root is importable
sys.path.insert(0, ".")

from bot.main_trader import HybridTrader
from config.settings import settings

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("live")


async def midnight_reset(trader: HybridTrader) -> None:
    """Reset risk manager at midnight UTC every day."""
    while True:
        now = datetime.now(timezone.utc)
        secs_to_midnight = (86400 - (now.hour * 3600 + now.minute * 60 + now.second))
        await asyncio.sleep(secs_to_midnight)
        trader.risk_mgr.reset_daily()
        logger.info("Daily risk reset at midnight UTC")


async def main(paper: bool) -> None:
    trader = HybridTrader(paper_mode=paper)

    # Graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, trader.stop)

    asyncio.create_task(midnight_reset(trader))
    await trader.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polymarket Hybrid SuperBot")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--paper", action="store_true", default=True, help="Paper trading (safe)")
    group.add_argument("--live",  action="store_true", default=False, help="Live trading (real money)")
    parser.add_argument("--symbol", default="BTC", help="Base asset (default: BTC)")
    args = parser.parse_args()

    paper_mode = not args.live
    if not paper_mode:
        print("\n⚠️  WARNING: LIVE MODE — real USDC will be used!")
        confirm = input("Type YES to confirm: ")
        if confirm.strip() != "YES":
            print("Aborted.")
            sys.exit(0)

    asyncio.run(main(paper=paper_mode))
