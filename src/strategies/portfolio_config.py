"""
src/strategies/portfolio_config.py  —  Multi-Sleeve Portfolio Loader
=====================================================================
Loads and validates config/portfolio.yaml into a typed object.

Validation (fails fast at load time):
  - layer weights must sum to 1.0
  - no single sleeve may exceed the concentration cap
  - risk caps are range-checked

This is a regime/factor-allocation framework derived from the sample
trade-plan spec — NOT a copy-trading system. The agents read these weights
and caps; they never mirror individual disclosed trades.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_YAML = _ROOT / "config" / "portfolio.yaml"


# ── Sub-models ────────────────────────────────────────────────────────────────

class PortfolioMeta(BaseModel):
    title: str
    version: str
    strategy_type: str
    risk_profile: str
    note: str | None = None


class Sleeve(BaseModel):
    weight: float = Field(ge=0.0, le=1.0)
    role: str
    mix: dict[str, float] | None = None
    basket: list[str] | None = None
    rationale: list[str] | None = None
    notes: list[str] | None = None
    deployment_conditions: list[str] | None = None


class RiskCaps(BaseModel):
    max_position_concentration_pct: float = Field(gt=0, le=100)
    max_trade_risk_pct: float = Field(gt=0, le=100)
    max_portfolio_heat_pct: float = Field(gt=0, le=100)
    max_correlated_positions: int = Field(ge=1)
    min_reward_to_risk: float = Field(gt=0)
    no_blind_replication: bool = True
    delayed_filings_not_for_copy_trading: bool = True


class Catalyst(BaseModel):
    name: str
    date: str
    action: str


# ── Top-level config ──────────────────────────────────────────────────────────

class PortfolioConfig(BaseModel):
    meta: PortfolioMeta
    layers: dict[str, Sleeve]
    risk: RiskCaps
    catalysts: list[Catalyst] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_weights_and_caps(self) -> PortfolioConfig:
        # 1. Layer weights must sum to 1.0
        total = sum(s.weight for s in self.layers.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Layer weights must sum to 1.0, got {total:.4f}. "
                f"Check config/portfolio.yaml layer weights."
            )
        # 2. No single NAME may exceed the concentration cap.
        #    The cap applies to individual holdings, not to a diversified sleeve.
        #    - sleeves with a sub-`mix`: check each sub-position
        #    - sleeves with a `basket` of N names: assume even split (weight/N)
        #    - sleeves that are a single position: check the sleeve weight
        cap = self.risk.max_position_concentration_pct / 100.0
        for name, sleeve in self.layers.items():
            if sleeve.mix:
                for pos, w in sleeve.mix.items():
                    if w > cap + 1e-9:
                        raise ValueError(
                            f"Holding '{pos}' in sleeve '{name}' is {w:.0%}, "
                            f"exceeds concentration cap "
                            f"{self.risk.max_position_concentration_pct:.0f}%."
                        )
            elif sleeve.basket:
                per_name = sleeve.weight / max(len(sleeve.basket), 1)
                if per_name > cap + 1e-9:
                    raise ValueError(
                        f"Sleeve '{name}' even-split is {per_name:.0%}/name, "
                        f"exceeds concentration cap "
                        f"{self.risk.max_position_concentration_pct:.0f}%."
                    )
            else:
                # single-position sleeve
                if sleeve.weight > cap + 1e-9:
                    raise ValueError(
                        f"Single-position sleeve '{name}' weight "
                        f"{sleeve.weight:.0%} exceeds concentration cap "
                        f"{self.risk.max_position_concentration_pct:.0f}%."
                    )
        # 3. Sub-mix (if present) should sum to the sleeve weight
        for name, sleeve in self.layers.items():
            if sleeve.mix:
                mix_total = sum(sleeve.mix.values())
                if abs(mix_total - sleeve.weight) > 1e-6:
                    raise ValueError(
                        f"Sleeve '{name}' mix sums to {mix_total:.4f} "
                        f"but sleeve weight is {sleeve.weight:.4f}."
                    )
        return self

    def summary(self) -> str:
        """Human-readable allocation summary."""
        lines = [f"Portfolio: {self.meta.title} (v{self.meta.version})", ""]
        for name, sleeve in self.layers.items():
            lines.append(f"  {name:20s} {sleeve.weight*100:5.1f}%  — {sleeve.role}")
        lines.append("")
        lines.append(f"  Risk: max {self.risk.max_position_concentration_pct:.0f}% "
                     f"per name · {self.risk.max_trade_risk_pct:.0f}% per trade · "
                     f"{self.risk.max_portfolio_heat_pct:.0f}% heat · "
                     f"≥{self.risk.min_reward_to_risk:.0f}:1 R:R")
        return "\n".join(lines)


# ── Loader ────────────────────────────────────────────────────────────────────

def load_portfolio(path: str | Path = _DEFAULT_YAML) -> PortfolioConfig:
    """Load and validate the portfolio YAML. Raises on any validation failure."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Portfolio config not found: {p}")
    raw: dict[str, Any] = yaml.safe_load(p.read_text())
    return PortfolioConfig.model_validate(raw)


if __name__ == "__main__":
    cfg = load_portfolio()
    print(cfg.summary())
