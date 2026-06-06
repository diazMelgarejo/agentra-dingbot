"""
tests/conftest.py  —  Shared fixtures for entire test suite.
All factories produce deterministic data via fixed seeds.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

# ── OHLCV factories ───────────────────────────────────────────────────────────

def make_ohlcv(n: int = 200, *, trend: str = "flat",
               base: float = 50_000.0, freq: str = "4h",
               seed: int = 42) -> pd.DataFrame:
    """Deterministic OHLCV DataFrame."""
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    if trend == "up":
        close = np.linspace(base, base * 1.40, n) + rng.standard_normal(n) * base * 0.003
    elif trend == "down":
        close = np.linspace(base, base * 0.60, n) + rng.standard_normal(n) * base * 0.003
    else:
        close = base + rng.standard_normal(n).cumsum() * base * 0.005
    close = np.abs(close)
    sp    = close * 0.001
    return pd.DataFrame({
        "open":   close - sp,
        "high":   close + sp * 2,
        "low":    close - sp * 2,
        "close":  close,
        "volume": rng.uniform(5, 200, n),
    }, index=dates)


def make_multi_tf(seed: int = 42) -> dict[str, pd.DataFrame]:
    return {
        "5m": make_ohlcv(200, freq="5min", seed=seed),
        "1h": make_ohlcv(200, freq="1h",   seed=seed+1),
        "4h": make_ohlcv(200, freq="4h",   seed=seed+2),
        "1d": make_ohlcv(365, freq="1D",   seed=seed+3),
    }


def make_sentiment_raw(fg: int = 45, vix: float = 20.0) -> dict[str, Any]:
    risk = "NORMAL" if vix < 30 else ("ELEVATED" if vix < 40 else "EXTREME")
    mult = 1.0 if vix < 30 else (0.5 if vix < 40 else 0.0)
    return {
        "fear_greed":      {"value": fg, "classification": "Neutral", "timestamp": "0"},
        "vix":             vix,
        "vix_risk_level":  risk,
        "size_multiplier": mult,
    }


def make_polymarket_snap(n_markets: int = 0) -> dict[str, Any]:
    return {
        "markets":          [],
        "enriched_markets": [],
        "farmable_markets": [],
        "total_discovered": n_markets,
    }


def make_full_snapshot(**kwargs) -> dict[str, Any]:
    return {
        "ohlcv":      make_multi_tf(),
        "sentiment":  make_sentiment_raw(),
        "polymarket": make_polymarket_snap(),
        "errors":     [],
        **kwargs,
    }


# ── Pytest fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def ohlcv():
    return make_multi_tf()


@pytest.fixture
def sentiment_raw():
    return make_sentiment_raw()


@pytest.fixture
def full_snapshot():
    return make_full_snapshot()


@pytest.fixture
def mock_snapshot(full_snapshot):
    """Patch fetch_full_snapshot to return deterministic data."""
    with patch("data.snapshot.fetch_full_snapshot",
               AsyncMock(return_value=full_snapshot)) as m:
        yield m


@pytest.fixture
def mock_funding_rate():
    with patch("agents.onchain_analyst.agent._fetch_funding_rate",
               AsyncMock(return_value=0.001)) as m:
        yield m


@pytest.fixture
def _force_llm_path(monkeypatch):
    """
    Force the debate engine onto the LLM code path so that patching _call_agent
    and _judge takes effect. (Default provider is now "none" → heuristic judge,
    which would otherwise bypass these mocks.)
    """
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    from core.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def mock_debate_neutral(_force_llm_path):
    """Stub out LLM debate to return NEUTRAL — deterministic for pipeline tests."""
    from core.state import Signal
    with patch("agents.debate_engine.agent._call_agent",
               AsyncMock(return_value="Market is uncertain.")), \
         patch("agents.debate_engine.agent._judge",
               AsyncMock(return_value=(Signal.NEUTRAL, 0.2, "Mocked judge"))) as m:
        yield m


@pytest.fixture
def mock_debate_buy(_force_llm_path):
    from core.state import Signal
    with patch("agents.debate_engine.agent._call_agent",
               AsyncMock(return_value="Strong bullish case.")), \
         patch("agents.debate_engine.agent._judge",
               AsyncMock(return_value=(Signal.BUY, 0.75, "Bullish consensus"))) as m:
        yield m


@pytest.fixture
def mock_debate_sell(_force_llm_path):
    from core.state import Signal
    with patch("agents.debate_engine.agent._call_agent",
               AsyncMock(return_value="Strong bearish case.")), \
         patch("agents.debate_engine.agent._judge",
               AsyncMock(return_value=(Signal.SELL, 0.70, "Bearish consensus"))) as m:
        yield m
