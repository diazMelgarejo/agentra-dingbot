"""
tests/test_portfolio_config.py — TDD for the multi-sleeve portfolio loader.
Written BEFORE the implementation (RED phase).
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError


class TestPortfolioConfig:
    """
    Journey: As a quant, I want the portfolio config validated at load time
    so a mis-weighted sleeve or a breached risk cap fails fast and loudly,
    never producing a silently wrong allocation.
    """

    def test_loads_default_yaml(self):
        from strategies.portfolio_config import load_portfolio
        cfg = load_portfolio()
        assert cfg.meta.version
        assert len(cfg.layers) == 4

    def test_layer_weights_sum_to_one(self):
        from strategies.portfolio_config import load_portfolio
        cfg = load_portfolio()
        total = sum(s.weight for s in cfg.layers.values())
        assert abs(total - 1.0) < 1e-6

    def test_risk_caps_present(self):
        from strategies.portfolio_config import load_portfolio
        cfg = load_portfolio()
        assert cfg.risk.max_position_concentration_pct == 25.0
        assert cfg.risk.max_trade_risk_pct == 2.0
        assert cfg.risk.max_portfolio_heat_pct == 6.0
        assert cfg.risk.max_correlated_positions == 3

    def test_weights_not_summing_to_one_rejected(self):
        from strategies.portfolio_config import PortfolioConfig
        bad = {
            "meta": {"title": "x", "version": "1", "strategy_type": "x", "risk_profile": "x"},
            "layers": {
                "a": {"weight": 0.5, "role": "x"},
                "b": {"weight": 0.2, "role": "x"},   # sums to 0.7, not 1.0
            },
            "risk": {"max_position_concentration_pct": 25, "max_trade_risk_pct": 2,
                     "max_portfolio_heat_pct": 6, "max_correlated_positions": 3,
                     "min_reward_to_risk": 3},
        }
        with pytest.raises(ValidationError):
            PortfolioConfig.model_validate(bad)

    def test_sleeve_weight_exceeding_concentration_is_flagged(self):
        """A single sleeve over the concentration cap must raise."""
        from strategies.portfolio_config import PortfolioConfig
        bad = {
            "meta": {"title": "x", "version": "1", "strategy_type": "x", "risk_profile": "x"},
            "layers": {
                "huge": {"weight": 0.80, "role": "x"},   # 80% > 25% cap
                "rest": {"weight": 0.20, "role": "x"},
            },
            "risk": {"max_position_concentration_pct": 25, "max_trade_risk_pct": 2,
                     "max_portfolio_heat_pct": 6, "max_correlated_positions": 3,
                     "min_reward_to_risk": 3},
        }
        with pytest.raises(ValidationError):
            PortfolioConfig.model_validate(bad)

    def test_to_summary_returns_readable_lines(self):
        from strategies.portfolio_config import load_portfolio
        cfg = load_portfolio()
        s = cfg.summary()
        assert "core_beta" in s
        assert "60" in s or "0.6" in s
