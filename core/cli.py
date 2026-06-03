"""
core/cli.py
───────────
Entry-point for the `agentic-trader` command-line tool.

Commands:
  run        -- Run a single analysis cycle (dry-run by default)
  dashboard  -- Start the FastAPI dashboard server
  backtest   -- Historical backtesting (Phase 3)
"""
from __future__ import annotations

import argparse
import asyncio
import sys

import structlog

logger = structlog.get_logger(__name__)


def _configure_logging(level: str = "INFO") -> None:
    import logging
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agentic-trader",
        description="Multi-agent crypto trading platform",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging verbosity (default: INFO)",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ── run ──────────────────────────────────────────────────────────────────
    run_p = sub.add_parser("run", help="Execute one trading analysis cycle")
    run_p.add_argument("--symbol",  default="BTC/USDT",  help="Trading pair (default: BTC/USDT)")
    run_p.add_argument("--dry-run", action="store_true", default=True,
                       help="Simulate orders without real execution (default: True)")
    run_p.add_argument("--live",    action="store_true",
                       help="Enable live order placement (requires --live flag explicitly)")

    # ── dashboard ────────────────────────────────────────────────────────────
    dash_p = sub.add_parser("dashboard", help="Launch the FastAPI dashboard")
    dash_p.add_argument("--host", default="0.0.0.0")
    dash_p.add_argument("--port", type=int, default=8000)
    dash_p.add_argument("--reload", action="store_true", help="Enable hot-reload for development")

    # ── backtest ─────────────────────────────────────────────────────────────
    bt_p = sub.add_parser("backtest", help="Run historical backtest (Phase 3)")
    bt_p.add_argument("--symbol", default="BTC/USDT")
    bt_p.add_argument("--start",  required=True, help="Start date YYYY-MM-DD")
    bt_p.add_argument("--end",    required=True, help="End date YYYY-MM-DD")

    args = parser.parse_args()
    _configure_logging(args.log_level)

    if args.command == "run":
        dry_run = not args.live  # explicit --live disables dry_run
        asyncio.run(_cmd_run(args.symbol, dry_run))

    elif args.command == "dashboard":
        _cmd_dashboard(args.host, args.port, args.reload)

    elif args.command == "backtest":
        asyncio.run(_cmd_backtest(args.symbol, args.start, args.end))

    else:
        parser.print_help()
        sys.exit(0)


# ─── Command implementations ──────────────────────────────────────────────────

async def _cmd_run(symbol: str, dry_run: bool) -> None:
    from core.orchestrator import build_trading_graph
    from core.state import TradingState

    mode = "DRY-RUN" if dry_run else "⚠️  LIVE"
    logger.info("cycle_start", symbol=symbol, mode=mode)

    graph = build_trading_graph()
    if graph is None:
        logger.error("graph_build_failed")
        sys.exit(1)

    initial = TradingState(symbol=symbol, dry_run=dry_run).__dict__
    result = await graph.ainvoke(initial)

    signal   = result.get("final_signal")
    conf     = result.get("final_confidence", 0)
    order    = result.get("order")
    errors   = result.get("errors", [])

    logger.info("cycle_complete", signal=signal, confidence=f"{conf:.1%}", order=str(order))

    if errors:
        logger.warning("cycle_errors", errors=errors)


def _cmd_dashboard(host: str, port: int, reload: bool) -> None:
    import uvicorn
    from dashboard.app import create_app

    logger.info("dashboard_start", host=host, port=port)
    uvicorn.run(
        "dashboard.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
        log_level="info",
    )


async def _cmd_backtest(symbol: str, start: str, end: str) -> None:
    logger.warning("backtest_not_implemented", eta="Phase 3", symbol=symbol, start=start, end=end)
    print("Backtesting coming in Phase 3. Track progress at: https://github.com/yourusername/agentic-trader")


if __name__ == "__main__":
    main()
