"""
tests/test_freqtrade_optional.py  —  Optional sidecar + zero-key operation
============================================================================
Two concerns:

  A. FreqTrade is fully optional
     - detect() returns False (homegrown) when no binary and no API
     - detect() honours mode=off / on / auto correctly
     - executor routes "homegrown" when FreqTrade absent, "freqtrade" when present
     - a FreqTrade error never breaks execution (falls back to homegrown)

  B. Zero-key heuristic judge
     - debate_engine produces a real consensus with provider="none" (no LLM)
     - weighted vote direction is correct (bullish analysts → BUY-side, etc.)
     - LLM failure with a configured provider falls back to the heuristic judge
     - the whole pipeline runs end-to-end with zero API keys / no Ollama
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from core.state import (
    IndicatorSnapshot,
    MLSnapshot,
    OnChainSnapshot,
    RiskAssessment,
    SentimentSnapshot,
    Signal,
    Timeframe,
    TradingState,
)

# ── A. FreqTrade detection ──────────────────────────────────────────────────────

class TestFreqTradeDetection:

    @pytest.mark.asyncio
    async def test_mode_off_never_uses(self):
        from agents.executor.freqtrade_client import FreqTradeClient
        use, reason = await FreqTradeClient.detect(
            "http://localhost:8080", "u", "p", mode="off")
        assert use is False
        assert "off" in reason.lower()

    @pytest.mark.asyncio
    async def test_auto_no_binary_no_api_is_homegrown(self):
        from agents.executor.freqtrade_client import FreqTradeClient
        with patch.object(FreqTradeClient, "find_binary", return_value=None), \
             patch.object(FreqTradeClient, "ping", AsyncMock(return_value=False)):
            use, reason = await FreqTradeClient.detect(
                "http://localhost:8080", "u", "p", mode="auto")
        assert use is False
        assert "not installed" in reason.lower() or "not running" in reason.lower()

    @pytest.mark.asyncio
    async def test_auto_binary_present_api_up_uses_it(self):
        from agents.executor.freqtrade_client import FreqTradeClient
        with patch.object(FreqTradeClient, "find_binary",
                          return_value="/opt/homebrew/bin/freqtrade"), \
             patch.object(FreqTradeClient, "ping", AsyncMock(return_value=True)):
            use, reason = await FreqTradeClient.detect(
                "http://localhost:8080", "u", "p", mode="auto")
        assert use is True
        assert "detected" in reason.lower()

    @pytest.mark.asyncio
    async def test_auto_binary_present_api_down_is_homegrown(self):
        """Binary installed but service not started → don't use, tell user why."""
        from agents.executor.freqtrade_client import FreqTradeClient
        with patch.object(FreqTradeClient, "find_binary",
                          return_value="/usr/local/bin/freqtrade"), \
             patch.object(FreqTradeClient, "ping", AsyncMock(return_value=False)):
            use, reason = await FreqTradeClient.detect(
                "http://localhost:8080", "u", "p", mode="auto")
        assert use is False
        assert "start freqtrade" in reason.lower()

    @pytest.mark.asyncio
    async def test_mode_on_requires_api(self):
        from agents.executor.freqtrade_client import FreqTradeClient
        with patch.object(FreqTradeClient, "ping", AsyncMock(return_value=False)):
            use, reason = await FreqTradeClient.detect(
                "http://localhost:8080", "u", "p", mode="on")
        assert use is False
        assert "unreachable" in reason.lower()

    @pytest.mark.asyncio
    async def test_mode_on_api_up_uses_it(self):
        from agents.executor.freqtrade_client import FreqTradeClient
        with patch.object(FreqTradeClient, "ping", AsyncMock(return_value=True)):
            use, reason = await FreqTradeClient.detect(
                "http://localhost:8080", "u", "p", mode="on")
        assert use is True

    def test_find_binary_checks_path_first(self):
        from agents.executor.freqtrade_client import FreqTradeClient
        with patch("agents.executor.freqtrade_client.shutil.which",
                   return_value="/custom/path/freqtrade"):
            assert FreqTradeClient.find_binary() == "/custom/path/freqtrade"

    def test_find_binary_returns_none_when_absent(self):
        from agents.executor.freqtrade_client import FreqTradeClient
        with patch("agents.executor.freqtrade_client.shutil.which", return_value=None), \
             patch("agents.executor.freqtrade_client.Path.exists", return_value=False):
            assert FreqTradeClient.find_binary() is None


# ── A2. Executor routing ─────────────────────────────────────────────────────────

class TestExecutorRouting:

    def _approved_state(self):
        snap = IndicatorSnapshot(symbol="BTC/USDT", timeframe=Timeframe.H4,
                                 close=50000.0, signal=Signal.BUY, confidence=0.7)
        return {
            "symbol": "BTC/USDT",
            "dry_run": True,
            "debate_consensus": Signal.BUY,
            "debate_confidence": 0.7,
            "technical": snap,
            "risk": RiskAssessment(approved=True, position_size_pct=10.0,
                                   stop_loss_pct=2.0, take_profit_pct=5.0),
            "errors": [],
        }

    @pytest.mark.asyncio
    async def test_routes_homegrown_when_freqtrade_absent(self):
        from agents.executor.agent import run
        from agents.executor.freqtrade_client import FreqTradeClient
        with patch.object(FreqTradeClient, "detect",
                          AsyncMock(return_value=(False, "not installed"))):
            result = await run(self._approved_state())
        assert result["order"] is not None
        assert result["order"].routed_via == "homegrown"

    @pytest.mark.asyncio
    async def test_routes_freqtrade_when_present(self):
        from agents.executor.agent import run
        from agents.executor.freqtrade_client import FreqTradeClient
        with patch.object(FreqTradeClient, "detect",
                          AsyncMock(return_value=(True, "detected binary + API up"))):
            result = await run(self._approved_state())
        assert result["order"].routed_via == "freqtrade"

    @pytest.mark.asyncio
    async def test_freqtrade_error_falls_back_to_homegrown(self):
        """If detect() raises, executor must still produce a homegrown order."""
        from agents.executor.agent import run
        from agents.executor.freqtrade_client import FreqTradeClient
        with patch.object(FreqTradeClient, "detect",
                          AsyncMock(side_effect=RuntimeError("boom"))):
            result = await run(self._approved_state())
        assert result["order"] is not None
        assert result["order"].routed_via == "homegrown"

    @pytest.mark.asyncio
    async def test_dry_run_with_freqtrade_does_not_send_entry(self):
        """In dry_run, even when detected, no real force_entry is sent."""
        from agents.executor.agent import run
        from agents.executor.freqtrade_client import FreqTradeClient
        fe = AsyncMock()
        with patch.object(FreqTradeClient, "detect",
                          AsyncMock(return_value=(True, "detected"))), \
             patch.object(FreqTradeClient, "force_entry", fe):
            result = await run(self._approved_state())   # dry_run=True
        assert result["order"].routed_via == "freqtrade"
        fe.assert_not_called()


# ── B. Zero-key heuristic judge ──────────────────────────────────────────────────

def _snap(sig: Signal, conf: float, kind: str = "tech"):
    if kind == "tech":
        return IndicatorSnapshot(symbol="BTC/USDT", timeframe=Timeframe.H4,
                                 close=50000.0, signal=sig, confidence=conf)
    if kind == "sent":
        return SentimentSnapshot(symbol="BTC/USDT", signal=sig, confidence=conf)
    if kind == "onchain":
        return OnChainSnapshot(symbol="BTC/USDT", signal=sig, confidence=conf)
    if kind == "ml":
        return MLSnapshot(symbol="BTC/USDT", signal=sig, confidence=conf)
    raise ValueError(kind)


class TestHeuristicJudge:

    def test_all_bullish_gives_buy_side(self):
        from agents.debate_engine.agent import _heuristic_judge
        state = {
            "technical": _snap(Signal.STRONG_BUY, 0.9, "tech"),
            "sentiment": _snap(Signal.BUY, 0.7, "sent"),
            "ml":        _snap(Signal.BUY, 0.8, "ml"),
            "onchain":   _snap(Signal.NEUTRAL, 0.1, "onchain"),
        }
        sig, conf, reason = _heuristic_judge(state)
        assert sig in (Signal.BUY, Signal.STRONG_BUY)
        assert conf > 0.4
        assert "avg=" in reason

    def test_all_bearish_gives_sell_side(self):
        from agents.debate_engine.agent import _heuristic_judge
        state = {
            "technical": _snap(Signal.STRONG_SELL, 0.9, "tech"),
            "sentiment": _snap(Signal.SELL, 0.7, "sent"),
            "ml":        _snap(Signal.STRONG_SELL, 0.8, "ml"),
        }
        sig, conf, _ = _heuristic_judge(state)
        assert sig in (Signal.SELL, Signal.STRONG_SELL)
        assert conf > 0.4

    def test_mixed_signals_dampen_to_neutral(self):
        from agents.debate_engine.agent import _heuristic_judge
        state = {
            "technical": _snap(Signal.BUY, 0.6, "tech"),
            "ml":        _snap(Signal.SELL, 0.6, "ml"),
        }
        sig, conf, _ = _heuristic_judge(state)
        assert sig == Signal.NEUTRAL

    def test_no_signals_returns_neutral_zero_conf(self):
        from agents.debate_engine.agent import _heuristic_judge
        sig, conf, reason = _heuristic_judge({})
        assert sig == Signal.NEUTRAL
        assert conf == 0.0
        assert "no analyst" in reason.lower()

    def test_confidence_bounded(self):
        from agents.debate_engine.agent import _heuristic_judge
        state = {
            "technical": _snap(Signal.STRONG_BUY, 1.0, "tech"),
            "sentiment": _snap(Signal.STRONG_BUY, 1.0, "sent"),
            "ml":        _snap(Signal.STRONG_BUY, 1.0, "ml"),
            "onchain":   _snap(Signal.STRONG_BUY, 1.0, "onchain"),
        }
        _, conf, _ = _heuristic_judge(state)
        assert 0.0 <= conf <= 1.0

    def test_technical_outweighs_onchain(self):
        """Technical (weight 1.0) should dominate on-chain (weight 0.5) on conflict."""
        from agents.debate_engine.agent import _heuristic_judge
        state = {
            "technical": _snap(Signal.BUY, 0.8, "tech"),
            "onchain":   _snap(Signal.SELL, 0.8, "onchain"),
        }
        sig, _, _ = _heuristic_judge(state)
        # Technical bullish at higher weight → net should not be SELL
        assert sig in (Signal.BUY, Signal.NEUTRAL)


class TestDebateEngineZeroKey:

    @pytest.mark.asyncio
    async def test_provider_none_uses_heuristic_no_llm_call(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "none")
        from core.config import get_settings
        get_settings.cache_clear()

        from agents.debate_engine import agent as de
        # If any LLM path were hit, this would raise
        with patch.object(de, "_call_agent",
                          AsyncMock(side_effect=AssertionError("LLM must not be called"))):
            state = {
                "symbol": "BTC/USDT",
                "technical": _snap(Signal.BUY, 0.7, "tech"),
                "sentiment": _snap(Signal.BUY, 0.6, "sent"),
            }
            result = await de.run(state)
        assert result["debate_consensus"] in (Signal.BUY, Signal.STRONG_BUY)
        assert 0.0 <= result["debate_confidence"] <= 1.0
        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_heuristic(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")  # configured but unreachable
        from core.config import get_settings
        get_settings.cache_clear()

        from agents.debate_engine import agent as de
        with patch.object(de, "_call_agent",
                          AsyncMock(side_effect=ConnectionError("ollama down"))):
            state = {
                "symbol": "BTC/USDT",
                "technical": _snap(Signal.SELL, 0.7, "tech"),
                "ml":        _snap(Signal.SELL, 0.6, "ml"),
            }
            result = await de.run(state)
        # Heuristic fallback produced a real consensus despite LLM failure
        assert result["debate_consensus"] in (Signal.SELL, Signal.STRONG_SELL, Signal.NEUTRAL)
        get_settings.cache_clear()


# ── B2. Full pipeline, zero keys ────────────────────────────────────────────────

class TestZeroKeyPipeline:

    @pytest.mark.asyncio
    async def test_full_pipeline_no_keys_no_llm(self, monkeypatch):
        """
        End-to-end pipeline with provider='none' and FreqTrade absent.
        No API keys, no Ollama, no Docker — must complete and produce a consensus.
        """
        monkeypatch.setenv("LLM_PROVIDER", "none")
        monkeypatch.setenv("FREQTRADE_MODE", "off")
        from core.config import get_settings
        get_settings.cache_clear()

        from core.orchestrator import build_trading_graph
        from tests.conftest import make_full_snapshot

        snap = make_full_snapshot()
        graph = build_trading_graph()
        with patch("data.snapshot.fetch_full_snapshot", AsyncMock(return_value=snap)), \
             patch("agents.onchain_analyst.agent._fetch_funding_rate",
                   AsyncMock(return_value=0.001)):
            result = await graph.ainvoke(TradingState(symbol="BTC/USDT", dry_run=True))

        # Pipeline completed with a real (heuristic) consensus and no LLM
        assert result.get("debate_consensus") in Signal
        assert "technical" in result and result["technical"] is not None
        assert isinstance(result.get("errors", []), list)
        get_settings.cache_clear()
