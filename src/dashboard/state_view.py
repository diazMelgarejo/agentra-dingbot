"""
src/dashboard/state_view.py  —  Dashboard State Mapper
========================================================
Single source of truth for the JSON shape the dashboard consumes.

Both the live WebSocket push (`/ws/signals`) and the static snapshot exporter
(`snapshot_export.py`) call `to_dashboard_view()` so the live and Pages
dashboards render identical structures. The dashboard's render() maps these
exact keys, so changing the shape here is the only place to touch.

Shape (matches docs/index.html render()):
{
  "symbol": "BTC/USDT",
  "debate_consensus": "STRONG_BUY",
  "debate_confidence": 0.72,
  "technical":  {"signal","rsi_14","ema_cross","macd"},
  "sentiment":  {"signal","fear_greed_index","vix"},
  "onchain":    {"signal","funding_rate"},
  "ml":         {"signal","prob_up","model_type"},
  "risk":       {"approved","position_size_pct","stop_loss_pct","take_profit_pct","vix_risk_level"},
  "polymarket": [{"question","yes_price","our_prob"}],
  "ohlcv_4h":   [{"time","open","high","low","close"}]
}
"""
from __future__ import annotations

import dataclasses
from datetime import datetime
from enum import Enum
from typing import Any


def _enum(v: Any) -> Any:
    """Unwrap Enum → its value; pass through everything else."""
    return v.value if isinstance(v, Enum) else v


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Attribute OR dict-key accessor (state may be object or dict)."""
    if obj is None:
        return default
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return getattr(obj, name, default)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _round(v: Any, n: int = 4) -> Any:
    return round(v, n) if isinstance(v, (int, float)) else v


def to_dashboard_view(state: Any) -> dict[str, Any]:
    """
    Convert a TradingState (object or dict) into the dashboard JSON shape.
    Tolerant of missing fields — any absent agent renders as null client-side.
    """
    technical = _get(state, "technical")
    sentiment = _get(state, "sentiment")
    onchain   = _get(state, "onchain")
    ml        = _get(state, "ml")
    risk      = _get(state, "risk")

    view: dict[str, Any] = {
        "symbol":            _get(state, "symbol", "BTC/USDT"),
        "debate_consensus":  _enum(_get(state, "debate_consensus", "NEUTRAL")),
        "debate_confidence": _round(_get(state, "debate_confidence", 0.0)),
    }

    # ── Technical ────────────────────────────────────────────────────────────
    if technical is not None:
        view["technical"] = {
            "signal":    _enum(_get(technical, "signal", "NEUTRAL")),
            "rsi_14":    _round(_get(technical, "rsi_14", _get(technical, "rsi"))),
            "ema_cross": _get(technical, "ema_cross"),
            "macd":      _round(_get(technical, "macd"), 5),
        }

    # ── Sentiment ────────────────────────────────────────────────────────────
    if sentiment is not None:
        view["sentiment"] = {
            "signal":           _enum(_get(sentiment, "signal", "NEUTRAL")),
            "fear_greed_index": _get(sentiment, "fear_greed_index"),
            "vix":              _round(_get(sentiment, "vix"), 2),
        }

    # ── On-chain ─────────────────────────────────────────────────────────────
    if onchain is not None:
        view["onchain"] = {
            "signal":       _enum(_get(onchain, "signal", "NEUTRAL")),
            "funding_rate": _round(_get(onchain, "funding_rate"), 6),
        }

    # ── ML ───────────────────────────────────────────────────────────────────
    if ml is not None:
        view["ml"] = {
            "signal":     _enum(_get(ml, "signal", "NEUTRAL")),
            "prob_up":    _round(_get(ml, "prob_up", 0.5), 4),
            "model_type": _get(ml, "model_type", "heuristic"),
        }

    # ── Risk assessment ──────────────────────────────────────────────────────
    if risk is not None:
        # vix_risk_level lives on SentimentSnapshot, not RiskAssessment
        vix_level = "NORMAL"
        if sentiment is not None:
            vix_level = _get(sentiment, "vix_risk_level", "NORMAL")
        view["risk"] = {
            "approved":          bool(_get(risk, "approved", False)),
            "position_size_pct": _round(_get(risk, "position_size_pct"), 2),
            "stop_loss_pct":     _round(_get(risk, "stop_loss_pct"), 2),
            "take_profit_pct":   _round(_get(risk, "take_profit_pct"), 2),
            "risk_reward_ratio": _round(_get(risk, "risk_reward_ratio"), 2),
            "max_loss_pct":      _round(_get(risk, "max_loss_pct"), 2),
            "vix_risk_level":    vix_level,
            "reasoning":         str(_get(risk, "reasoning", "") or "")[:120],
        }

    # ── Polymarket markets ───────────────────────────────────────────────────
    markets  = _get(state, "polymarket_markets", []) or []
    decision = _get(state, "polymarket_decision")
    decision_mid = _get(decision, "market_id", "") if decision else ""

    pm: list[dict[str, Any]] = []
    for m in markets[:6]:
        mid = _get(m, "market_id", "")
        # our_prob comes from PolymarketDecision.posterior_prob, not from the market
        our_prob = None
        if decision and mid and mid == decision_mid:
            our_prob = _round(_get(decision, "posterior_prob"), 3)
        pm.append({
            "question":   (_get(m, "question", "—") or "—")[:80],
            "yes_price":  _round(_get(m, "yes_price"), 3),
            "our_prob":   our_prob,
            "volume_24h": _round(_get(m, "volume_24h"), 0),
            "is_active":  bool(_get(m, "is_active", True)),
        })
    if pm:
        view["polymarket"] = pm

    # Polymarket decision summary (shown in sidebar badge when trade authorised)
    if decision is not None and _get(decision, "should_trade", False):
        view["polymarket_decision"] = {
            "should_trade":   True,
            "direction":      _enum(_get(decision, "direction", "NEUTRAL")),
            "question":       str(_get(decision, "question", ""))[:80],
            "yes_price":      _round(_get(decision, "yes_price"), 3),
            "posterior_prob": _round(_get(decision, "posterior_prob"), 3),
            "edge_pct":       _round(_get(decision, "edge_pct"), 2),
            "position_usdc":  _round(_get(decision, "position_usdc"), 2),
        }

    # ── OHLCV for the candlestick chart ──────────────────────────────────────
    ohlcv = _get(state, "ohlcv", {}) or {}
    candles = ohlcv.get("4h") if isinstance(ohlcv, dict) else None
    if candles is not None:
        view["ohlcv_4h"] = _coerce_candles(candles)

    return view


def _coerce_candles(candles: Any) -> list[dict[str, Any]]:
    """
    Normalise OHLCV into [{time,open,high,low,close}] with epoch-second time.
    Accepts a pandas DataFrame or a list of dicts/rows. Caps at 200 bars.
    """
    out: list[dict[str, Any]] = []
    try:
        # pandas DataFrame path
        if hasattr(candles, "iterrows"):
            tail = candles.tail(200)
            for ts, row in tail.iterrows():
                t = ts.timestamp() if hasattr(ts, "timestamp") else ts
                out.append({
                    "time":  int(t),
                    "open":  round(float(row["open"]), 2),
                    "high":  round(float(row["high"]), 2),
                    "low":   round(float(row["low"]), 2),
                    "close": round(float(row["close"]), 2),
                })
            return out
        # list path
        for row in list(candles)[-200:]:
            if isinstance(row, dict):
                t = row.get("time") or row.get("timestamp") or row.get("t")
                if isinstance(t, datetime):
                    t = t.timestamp()
                out.append({
                    "time":  int(t),
                    "open":  round(float(row["open"]), 2),
                    "high":  round(float(row["high"]), 2),
                    "low":   round(float(row["low"]), 2),
                    "close": round(float(row["close"]), 2),
                })
    except Exception:
        return out
    return out
