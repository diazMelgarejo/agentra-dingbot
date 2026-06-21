"""
tests/test_dashboard_snapshot.py — dashboard view mapper + snapshot exporter.
"""
from __future__ import annotations

import json


class TestStateView:
    """to_dashboard_view must produce the exact shape the dashboard renders."""

    def test_handles_empty_state(self):
        from dashboard.state_view import to_dashboard_view
        view = to_dashboard_view({})
        assert view["symbol"] == "BTC/USDT"
        assert view["debate_consensus"] == "NEUTRAL"

    def test_maps_dict_state(self):
        from dashboard.state_view import to_dashboard_view
        state = {
            "symbol": "ETH/USDT",
            "debate_consensus": "BUY",
            "debate_confidence": 0.71,
            "technical": {"signal": "BUY", "rsi_14": 38.2, "ema_cross": "BULL"},
            "sentiment": {"signal": "BUY", "fear_greed_index": 28, "vix": 19.5},
            "onchain": {"signal": "NEUTRAL", "funding_rate": 0.0001},
            "ml": {"signal": "BUY", "prob_up": 0.67, "model_type": "sklearn_hgb"},
        }
        view = to_dashboard_view(state)
        assert view["symbol"] == "ETH/USDT"
        assert view["debate_consensus"] == "BUY"
        assert view["technical"]["rsi_14"] == 38.2
        assert view["sentiment"]["fear_greed_index"] == 28
        assert view["ml"]["prob_up"] == 0.67

    def test_unwraps_enum_signal(self):
        from core.state import Signal
        from dashboard.state_view import to_dashboard_view
        view = to_dashboard_view({"debate_consensus": Signal.STRONG_BUY})
        assert view["debate_consensus"] == Signal.STRONG_BUY.value

    def test_maps_risk_assessment(self):
        from core.state import RiskAssessment
        from dashboard.state_view import to_dashboard_view
        risk = RiskAssessment(approved=True, position_size_pct=10.8,
                              stop_loss_pct=2.0, take_profit_pct=5.0)
        view = to_dashboard_view({"risk": risk})
        assert view["risk"]["approved"] is True
        assert view["risk"]["position_size_pct"] == 10.8

    def test_coerces_list_candles(self):
        from dashboard.state_view import to_dashboard_view
        candles = [{"time": 1700000000, "open": 1, "high": 2, "low": 0.5, "close": 1.5}]
        view = to_dashboard_view({"ohlcv": {"4h": candles}})
        assert view["ohlcv_4h"][0]["close"] == 1.5
        assert view["ohlcv_4h"][0]["time"] == 1700000000


class TestSnapshotExport:
    """The exporter must write valid enveloped JSON the dashboard can poll."""

    def test_demo_view_has_required_keys(self):
        from dashboard.snapshot_export import _demo_view
        v = _demo_view("BTC/USDT")
        for k in ["symbol", "debate_consensus", "technical", "sentiment",
                  "onchain", "ml", "risk", "polymarket", "ohlcv_4h"]:
            assert k in v, f"demo view missing {k}"

    def test_wrap_adds_meta_envelope(self):
        from dashboard.snapshot_export import _demo_view, _wrap
        payload = _wrap(_demo_view(), "demo", "BTC/USDT")
        assert payload["meta"]["schema"] == "agentra.dashboard.v1"
        assert payload["meta"]["mode"] == "demo"
        assert "generated_at" in payload["meta"]
        assert "data" in payload

    def test_write_snapshot_roundtrip(self, tmp_path):
        from dashboard.snapshot_export import _demo_view, _wrap, write_snapshot
        out = tmp_path / "snapshot.json"
        write_snapshot(_wrap(_demo_view(), "demo", "BTC/USDT"), out)
        loaded = json.loads(out.read_text())
        assert loaded["data"]["debate_consensus"] == "STRONG_BUY"
        assert loaded["meta"]["schema"] == "agentra.dashboard.v1"
