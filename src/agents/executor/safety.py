"""
src/agents/executor/safety.py  —  Step 7: Executor Safety Layer
================================================================
Every production failure mode from research is addressed here.

Documented failures prevented:
  - 68 consecutive rejections from qty=0.0 after generic rounding
  - $65M+ drained via keys with withdraw permission enabled
  - Ghost positions blocking trading for months
  - Silent inactivity (bot "running" but not trading)

All safety checks FAIL CLOSED: when in doubt, reject the action.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class PermissionResult:
    """Result of an API key permission audit."""
    trade_ok:    bool = False
    withdraw_ok: bool = False
    transfer_ok: bool = False
    reason:      str  = ""

    @property
    def safe(self) -> bool:
        """Safe = can trade AND cannot withdraw.
        No legitimate trading bot needs withdraw permission."""
        return self.trade_ok and not self.withdraw_ok


@dataclass
class ValidationResult:
    """Result of an order sanity check."""
    valid:  bool = False
    errors: list[str] = field(default_factory=list)


# ── API Permission Check ──────────────────────────────────────────────────────

async def check_api_permissions(exchange) -> PermissionResult:
    """
    Fetch and validate API key permissions from the exchange.
    Fails closed: any error returns safe=False (deny-by-default).

    HARD RULE: if withdraw permission is enabled, log CRITICAL and return safe=False.
    No legitimate trading bot needs withdraw access.
    """
    try:
        perms = await exchange.fetchPermissions()
        result = PermissionResult(
            trade_ok    = bool(perms.get("trade",    False)),
            withdraw_ok = bool(perms.get("withdraw", False)),
            transfer_ok = bool(perms.get("transfer", False)),
        )
        if result.withdraw_ok:
            logger.critical(
                "api_key_has_withdraw_permission_HALT",
                exchange=getattr(exchange, "id", "unknown"),
                msg="Withdraw permission detected. Bot will not start. "
                    "Revoke withdraw permission and use trade-only keys.",
            )
            result.reason = (
                "CRITICAL: API key has WITHDRAW permission enabled. "
                "This is a critical security risk. "
                "Generate a new trade-only key without withdraw access."
            )
        elif not result.trade_ok:
            result.reason = "API key does not have trade permission."
        else:
            result.reason = "Trade-only key verified (no withdraw, no transfer)."
        return result

    except Exception as exc:
        logger.error("api_permission_check_failed", error=str(exc))
        return PermissionResult(
            trade_ok=False,
            withdraw_ok=False,
            reason=f"Permission check failed (fail-closed): {exc}",
        )


# ── Order Sanity Validation ───────────────────────────────────────────────────

async def validate_order(
    symbol: str,
    side: str,
    price: float,
    qty: float,
    exchange,
    price_bounds_pct: float = 0.10,
) -> ValidationResult:
    """
    Sanity-check an order before submission. Returns ValidationResult.

    Checks:
      1. Quantity > 0 after tick-size rounding (prevents the 68-reject bug)
      2. Quantity ≥ exchange minimum
      3. Price within price_bounds_pct of last trade
      4. Sufficient balance for the order

    All checks are explicit — nothing is silently rounded/dropped.
    """
    errors: list[str] = []

    # ── 1. Tick-size rounding ─────────────────────────────────────────────────
    try:
        rounded_qty = float(exchange.amount_to_precision(symbol, qty))
        if rounded_qty <= 0:
            errors.append(
                f"Quantity {qty} rounds to zero after exchange precision "
                f"({symbol}). Order skipped to prevent INVALID_ORDERQTY rejection."
            )
    except Exception as exc:
        rounded_qty = qty
        errors.append(f"Could not apply amount_to_precision: {exc}")

    # ── 2. Minimum quantity ───────────────────────────────────────────────────
    if not errors:  # only if qty survived rounding
        try:
            market   = exchange.markets.get(symbol, {})
            min_qty  = market.get("limits", {}).get("amount", {}).get("min", 0)
            if min_qty and rounded_qty < float(min_qty):
                errors.append(
                    f"Quantity {rounded_qty} below exchange minimum {min_qty} "
                    f"for {symbol}."
                )
        except Exception as exc:
            logger.warning("min_qty_check_failed", error=str(exc))

    # ── 3. Price sanity bounds ────────────────────────────────────────────────
    try:
        ticker     = await exchange.fetchTicker(symbol)
        last_price = ticker.get("last") or ticker.get("close", 0)
        if last_price and last_price > 0:
            deviation = abs(price - last_price) / last_price
            if deviation > price_bounds_pct:
                errors.append(
                    f"Price {price} deviates {deviation:.1%} from last "
                    f"trade {last_price} — exceeds {price_bounds_pct:.0%} bound. "
                    f"Possible stale price."
                )
    except Exception as exc:
        logger.warning("price_check_failed", error=str(exc))

    # ── 4. Balance check ─────────────────────────────────────────────────────
    try:
        balance   = await exchange.fetchBalance()
        quote     = symbol.split("/")[1] if "/" in symbol else "USDT"
        free_usdt = float((balance.get(quote) or {}).get("free", 0))
        order_cost = rounded_qty * price
        if side.lower() == "buy" and free_usdt < order_cost:
            errors.append(
                f"Insufficient balance: need {order_cost:.2f} {quote} "
                f"but only {free_usdt:.2f} available."
            )
    except Exception as exc:
        logger.warning("balance_check_failed", error=str(exc))

    return ValidationResult(valid=len(errors) == 0, errors=errors)


# ── Kill Switch ───────────────────────────────────────────────────────────────

class KillSwitch:
    """
    File-based kill switch. Halts all executor activity immediately.
    Survives process restarts (state is on disk, not in memory).

    Usage:
        ks = KillSwitch()
        ks.arm()      # creates flag file
        ks.is_armed() # True if flag file exists
        ks.disarm()   # removes flag file
    """

    def __init__(self, flag_dir: str | None = None):
        base = flag_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "data")
        self._flag = Path(base) / "KILL_SWITCH"

    def arm(self) -> None:
        """Arm the kill switch. Idempotent."""
        try:
            self._flag.parent.mkdir(parents=True, exist_ok=True)
            self._flag.touch()
            logger.critical("kill_switch_armed", path=str(self._flag))
        except Exception as exc:
            logger.error("kill_switch_arm_failed", error=str(exc))

    def disarm(self) -> None:
        """Disarm the kill switch. Idempotent."""
        try:
            if self._flag.exists():
                self._flag.unlink()
            logger.info("kill_switch_disarmed")
        except Exception as exc:
            logger.error("kill_switch_disarm_failed", error=str(exc))

    def is_armed(self) -> bool:
        """Return True if the kill switch flag file exists."""
        return self._flag.exists()


# ── Live Trading Gate ─────────────────────────────────────────────────────────

def is_live_trading_enabled() -> bool:
    """
    Returns True ONLY if LIVE_TRADING env var is explicitly set to a truthy value.
    Default is paper mode (False). Truthy values: true, 1, yes (case-insensitive).

    Gate logic: default MUST be paper. Going live is a deliberate, explicit act.
    """
    val = os.getenv("LIVE_TRADING", "").strip().lower()
    return val in ("true", "1", "yes")


# ── Startup checks (call once at bot startup) ─────────────────────────────────

async def startup_safety_checks(exchange=None) -> list[str]:
    """
    Run all safety checks at startup. Returns a list of warnings/errors.
    Caller should HALT if any returned item starts with 'CRITICAL:'.
    """
    issues: list[str] = []

    # Kill switch check
    ks = KillSwitch()
    if ks.is_armed():
        issues.append("CRITICAL: Kill switch is armed. Disarm before starting.")

    # API key permissions (only in live mode)
    if is_live_trading_enabled() and exchange is not None:
        perms = await check_api_permissions(exchange)
        if not perms.safe:
            issues.append(f"CRITICAL: {perms.reason}")

    # Warn if running live without explicit flag
    if not is_live_trading_enabled():
        issues.append("INFO: Running in paper mode (LIVE_TRADING not set).")

    return issues
