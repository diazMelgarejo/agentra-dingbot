"""
tests/test_risk_manager.py
Tests for risk manager signal gating and position sizing.
"""
import pytest
from unittest.mock import MagicMock, patch
from core.state import Signal, RiskAssessment, IndicatorSnapshot, Timeframe


def _tech(close=50_000.0, atr=1_200.0):
    """Create a minimal IndicatorSnapshot."""
    snap = MagicMock(spec=IndicatorSnapshot)
    snap.close  = close
    snap.atr_14 = atr
    return snap


def _state(signal=Signal.BUY, confidence=0.7, tech=None):
    return {
        "debate_consensus":  signal,
        "debate_confidence": confidence,
        "technical":         tech or _tech(),
        "symbol":            "BTC/USDT",
        "dry_run":           True,
    }


@pytest.mark.asyncio
async def test_neutral_signal_rejected():
    from agents.risk_manager.agent import run
    result = await run(_state(signal=Signal.NEUTRAL))
    assert result["risk"].approved is False
    assert "NEUTRAL" in result["risk"].reasoning


@pytest.mark.asyncio
async def test_low_confidence_rejected():
    from agents.risk_manager.agent import run
    result = await run(_state(confidence=0.1))
    assert result["risk"].approved is False


@pytest.mark.asyncio
async def test_buy_signal_approved():
    from agents.risk_manager.agent import run
    result = await run(_state(signal=Signal.BUY, confidence=0.8))
    r = result["risk"]
    assert r.approved is True
    assert r.position_size_pct > 0
    assert r.stop_loss_pct > 0
    assert r.take_profit_pct > r.stop_loss_pct  # TP always bigger than SL


@pytest.mark.asyncio
async def test_strong_buy_larger_than_buy():
    from agents.risk_manager.agent import run
    r_strong = (await run(_state(signal=Signal.STRONG_BUY, confidence=0.9)))["risk"]
    r_buy    = (await run(_state(signal=Signal.BUY,        confidence=0.9)))["risk"]
    assert r_strong.position_size_pct >= r_buy.position_size_pct


@pytest.mark.asyncio
async def test_max_loss_cap_enforced():
    from agents.risk_manager.agent import run, _MAX_LOSS_PCT
    # Force high ATR to trigger cap
    result = await run(_state(signal=Signal.STRONG_BUY, confidence=1.0, tech=_tech(atr=50_000)))
    r = result["risk"]
    assert r.approved is True
    assert r.max_loss_pct <= _MAX_LOSS_PCT + 0.01   # small float tolerance


@pytest.mark.asyncio
async def test_fallback_sl_when_no_atr():
    from agents.risk_manager.agent import run, _FALLBACK_SL_PCT
    tech = _tech(atr=0)  # atr=0 triggers fallback
    tech.atr_14 = None
    result = await run(_state(signal=Signal.BUY, confidence=0.7, tech=tech))
    r = result["risk"]
    assert r.approved is True
    assert abs(r.stop_loss_pct - _FALLBACK_SL_PCT) < 0.01


@pytest.mark.asyncio
async def test_risk_reward_ratio_correct():
    from agents.risk_manager.agent import run
    result = await run(_state(signal=Signal.BUY, confidence=0.8))
    r = result["risk"]
    # BUY rule: tp_rr = 2.5
    assert abs(r.risk_reward_ratio - 2.5) < 0.01
    assert abs(r.take_profit_pct / r.stop_loss_pct - 2.5) < 0.1
