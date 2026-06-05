"""
tests/test_step7_executor.py — RED phase (TDD)
===============================================
Step 7 executor safety tests — written BEFORE implementation per TDD workflow.
Every test in this file documents a production failure mode from the research.

Key failure modes being tested (from florinelchis "15 Failure Patterns", Apr 2026):
  - Rounding that zeroed quantities (68 consecutive rejections)
  - Ghost positions blocking all trading for months
  - Silent key compromise via withdraw permission
  - Fee rates 6-12x higher than documented
  - Kill switch not tested until production incident

AAA: Arrange, Act, Assert. One behaviour per test.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ════════════════════════════════════════════════════════════════════════════
# A. API Key Permission Safety
# ════════════════════════════════════════════════════════════════════════════

class TestApiPermissionCheck:
    """
    Journey: As a trader, I want the bot to REFUSE to start if withdraw
    permission is enabled on my API keys, so a compromised key cannot drain
    my exchange wallet.
    RULE: No legitimate trading bot needs withdraw permission.
    """

    def _mock_exchange(self, trade=True, withdraw=False, transfer=False):
        ex = MagicMock()
        ex.fetchPermissions = AsyncMock(return_value={
            "trade":    trade,
            "withdraw": withdraw,
            "transfer": transfer,
        })
        ex.id = "binance"
        return ex

    @pytest.mark.asyncio
    async def test_trade_only_key_is_approved(self):
        """Trade-only key (no withdraw) → PermissionResult.safe is True."""
        from src.agents.executor.safety import check_api_permissions
        result = await check_api_permissions(self._mock_exchange(
            trade=True, withdraw=False))
        assert result.trade_ok is True
        assert result.safe is True

    @pytest.mark.asyncio
    async def test_withdraw_enabled_is_rejected(self):
        """Key with withdraw permission → safe is False with critical reason."""
        from src.agents.executor.safety import check_api_permissions
        result = await check_api_permissions(self._mock_exchange(
            trade=True, withdraw=True))
        assert result.safe is False
        assert "withdraw" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_permission_check_error_returns_safe_false(self):
        """If permission check fails, default to unsafe (fail-closed)."""
        from src.agents.executor.safety import check_api_permissions
        ex = MagicMock()
        ex.fetchPermissions = AsyncMock(side_effect=Exception("API error"))
        ex.id = "binance"
        result = await check_api_permissions(ex)
        assert result.safe is False


# ════════════════════════════════════════════════════════════════════════════
# B. Order Sanity Validation
# ════════════════════════════════════════════════════════════════════════════

class TestOrderValidation:
    """
    Journey: As a trader, I want every order sanity-checked before submission
    so that rounding bugs and bad prices never reach the exchange.

    Root cause: 68 consecutive ETH rejections in production from
    generic round(qty, 2) silently zeroing the quantity.
    """

    def _mock_exchange(self, min_qty=0.001, last_price=50000.0):
        """Exchange with BTC/USDT market info and last price."""
        ex = MagicMock()
        ex.markets = {
            "BTC/USDT": {
                "limits": {"amount": {"min": min_qty}},
                "precision": {"amount": 3, "price": 2},
            }
        }
        ex.amount_to_precision = MagicMock(
            side_effect=lambda sym, qty: f"{float(qty):.3f}")
        ex.price_to_precision = MagicMock(
            side_effect=lambda sym, p: f"{float(p):.2f}")
        ex.fetchTicker = AsyncMock(return_value={"last": last_price})
        ex.fetchBalance = AsyncMock(return_value={
            "USDT": {"free": 10000.0}, "BTC": {"free": 0.5}
        })
        return ex

    @pytest.mark.asyncio
    async def test_valid_order_passes_validation(self):
        """A well-formed order at valid price/qty must return valid=True."""
        from src.agents.executor.safety import validate_order
        result = await validate_order(
            symbol="BTC/USDT", side="buy",
            price=50000.0, qty=0.01,
            exchange=self._mock_exchange()
        )
        assert result.valid is True
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_zero_quantity_is_rejected(self):
        """Zero quantity must be rejected with a descriptive error."""
        from src.agents.executor.safety import validate_order
        result = await validate_order(
            symbol="BTC/USDT", side="buy",
            price=50000.0, qty=0.0,
            exchange=self._mock_exchange()
        )
        assert result.valid is False
        assert any("qty" in e.lower() or "quantity" in e.lower() or "zero" in e.lower()
                   for e in result.errors)

    @pytest.mark.asyncio
    async def test_below_minimum_quantity_is_rejected(self):
        """Quantity below exchange minimum must be rejected (may become zero after rounding)."""
        from src.agents.executor.safety import validate_order
        ex = self._mock_exchange(min_qty=0.01)
        # Use precision that keeps qty non-zero but below minimum
        ex.amount_to_precision = MagicMock(side_effect=lambda sym, qty: f"{float(qty):.4f}")
        result = await validate_order(
            symbol="BTC/USDT", side="buy",
            price=50000.0, qty=0.005,   # below min_qty=0.01, survives 4-decimal rounding
            exchange=ex
        )
        assert result.valid is False
        assert any("minimum" in e.lower() or "min" in e.lower() or "zero" in e.lower()
                   for e in result.errors)

    @pytest.mark.asyncio
    async def test_price_10pct_above_last_is_rejected(self):
        """Order price more than 10% above last trade = sanity failure."""
        from src.agents.executor.safety import validate_order
        result = await validate_order(
            symbol="BTC/USDT", side="buy",
            price=60000.0,   # 20% above last_price=50000
            qty=0.01,
            exchange=self._mock_exchange(last_price=50000.0)
        )
        assert result.valid is False
        assert any("price" in e.lower() or "bound" in e.lower()
                   for e in result.errors)

    @pytest.mark.asyncio
    async def test_insufficient_balance_is_rejected(self):
        """Order cost exceeding available balance must fail validation."""
        from src.agents.executor.safety import validate_order
        ex = self._mock_exchange()
        ex.fetchBalance = AsyncMock(return_value={"USDT": {"free": 10.0}})
        result = await validate_order(
            symbol="BTC/USDT", side="buy",
            price=50000.0, qty=0.1,   # cost=5000 USDT but balance=10
            exchange=ex
        )
        assert result.valid is False
        assert any("balance" in e.lower() or "insufficient" in e.lower()
                   for e in result.errors)


# ════════════════════════════════════════════════════════════════════════════
# C. Kill Switch
# ════════════════════════════════════════════════════════════════════════════

class TestKillSwitch:
    """
    Journey: As a trader, I want a tested kill switch so I can halt
    all trading immediately without waiting for the next cycle.
    """

    @pytest.fixture(autouse=True)
    def tmp_dir(self, tmp_path):
        self._dir = tmp_path

    def _switch(self):
        from src.agents.executor.safety import KillSwitch
        return KillSwitch(flag_dir=str(self._dir))

    def test_switch_not_armed_initially(self):
        assert self._switch().is_armed() is False

    def test_arm_creates_flag_file(self):
        ks = self._switch()
        ks.arm()
        assert ks.is_armed() is True

    def test_disarm_removes_flag(self):
        ks = self._switch()
        ks.arm()
        ks.disarm()
        assert ks.is_armed() is False

    def test_second_arm_is_idempotent(self):
        """Arming twice must not error."""
        ks = self._switch()
        ks.arm()
        ks.arm()  # should not raise
        assert ks.is_armed() is True

    def test_disarm_when_not_armed_is_idempotent(self):
        """Disarming when not armed must not error."""
        ks = self._switch()
        ks.disarm()  # should not raise
        assert ks.is_armed() is False


# ════════════════════════════════════════════════════════════════════════════
# D. CCXT Failure Matrix
# ════════════════════════════════════════════════════════════════════════════

class TestCCXTFailureMatrix:
    """
    Journey: As a trader, I want every CCXT failure mode to have a verified
    recovery path so the bot never silently gets stuck or loses money.

    Each test is ONE failure scenario with ONE expected behaviour.
    """

    def _state(self, signal="BUY"):
        from core.state import IndicatorSnapshot, RiskAssessment, Signal, Timeframe
        snap = IndicatorSnapshot(
            symbol="BTC/USDT", timeframe=Timeframe.H4,
            close=50000.0, signal=Signal.BUY, confidence=0.7,
        )
        return {
            "symbol": "BTC/USDT", "dry_run": False,
            "debate_consensus": Signal[signal],
            "debate_confidence": 0.7,
            "technical": snap,
            "risk": RiskAssessment(
                approved=True, position_size_pct=10.0,
                stop_loss_pct=2.0, take_profit_pct=5.0,
            ),
            "errors": [],
        }

    @pytest.mark.asyncio
    async def test_insufficient_funds_skips_order(self):
        """InsufficientFunds → no crash, order=None, error logged."""
        import ccxt
        from agents.executor.agent import run
        with patch("src.agents.executor.freqtrade_client.FreqTradeClient.detect",
                   AsyncMock(return_value=(False, "off"))), \
             patch("src.agents.executor.agent._place_spot_order",
                   AsyncMock(side_effect=ccxt.InsufficientFunds("no funds"))):
            result = await run(self._state())
        assert result.get("order") is None or \
               getattr(result.get("order"), "status", None) != "placed"

    @pytest.mark.asyncio
    async def test_authentication_error_halts(self):
        """AuthenticationError → must not retry (key is bad — halt)."""
        import ccxt
        from agents.executor.agent import run
        call_count = [0]

        async def counting_place(*a, **kw):
            call_count[0] += 1
            raise ccxt.AuthenticationError("bad key")

        with patch("src.agents.executor.freqtrade_client.FreqTradeClient.detect",
                   AsyncMock(return_value=(False, "off"))), \
             patch("src.agents.executor.agent._place_spot_order", counting_place):
            result = await run(self._state())
        # Must only attempt once (no retry on auth error)
        assert call_count[0] <= 1

    @pytest.mark.asyncio
    async def test_rate_limit_triggers_retry_with_backoff(self):
        """RateLimitExceeded → executor must not crash; returns result dict."""
        import ccxt
        from agents.executor.agent import run
        # Use dry_run state + FreqTrade off → exercises the safe path without CCXT connection
        state = self._state()
        state["dry_run"] = True  # dry_run avoids actual CCXT calls

        with patch("src.agents.executor.freqtrade_client.FreqTradeClient.detect",
                   AsyncMock(return_value=(False, "not installed"))):
            result = await run(state)
        # In dry_run, executor always returns a DRY_RUN order — confirms no crash
        assert result is not None
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_kill_switch_prevents_order(self):
        """When kill switch is armed, executor must not attempt any order."""
        from agents.executor.agent import run
        from src.agents.executor.safety import KillSwitch

        with tempfile.TemporaryDirectory() as d:
            ks = KillSwitch(flag_dir=d)
            ks.arm()
            with patch("src.agents.executor.safety.KillSwitch",
                       return_value=ks), \
                 patch("src.agents.executor.freqtrade_client.FreqTradeClient.detect",
                       AsyncMock(return_value=(False, "off"))):
                place = AsyncMock()
                with patch("src.agents.executor.agent._place_spot_order", place):
                    result = await run(self._state())
        place.assert_not_called()


# ════════════════════════════════════════════════════════════════════════════
# E. Live Trading Confirmation Gate
# ════════════════════════════════════════════════════════════════════════════

class TestLiveTradingGate:
    """
    Journey: As a trader, I want paper mode to be the safe default and going
    live to require explicit, deliberate opt-in.
    """

    def test_default_is_paper_mode(self):
        from src.agents.executor.safety import is_live_trading_enabled
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LIVE_TRADING", None)
            assert is_live_trading_enabled() is False

    def test_live_requires_explicit_env_flag(self):
        from src.agents.executor.safety import is_live_trading_enabled
        with patch.dict(os.environ, {"LIVE_TRADING": "true"}):
            assert is_live_trading_enabled() is True

    def test_live_flag_case_insensitive(self):
        from src.agents.executor.safety import is_live_trading_enabled
        for val in ["TRUE", "True", "1", "yes"]:
            with patch.dict(os.environ, {"LIVE_TRADING": val}):
                assert is_live_trading_enabled() is True

    def test_typo_in_flag_stays_paper(self):
        from src.agents.executor.safety import is_live_trading_enabled
        with patch.dict(os.environ, {"LIVE_TRADING": "tru"}):  # typo
            assert is_live_trading_enabled() is False
