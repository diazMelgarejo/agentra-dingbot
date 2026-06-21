"""
test_dashboard.py — Step 8 dashboard tests (TDD: RED → GREEN → REFACTOR)

Tests:
  - GET /api/health returns 200
  - WebSocket /ws/signals connects and receives a heartbeat when no cycle has run
  - WebSocket /ws/signals rejects connections without a valid token when WS_TOKEN is set
  - _run_cycle writes docs/data/latest.json after a successful cycle
"""
from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def app():
    """Fresh app instance with no token required."""
    env = {"WS_READ_TOKEN": ""}
    with patch.dict(os.environ, env, clear=False):
        from src.dashboard.app import create_app
        return create_app()


@pytest.fixture()
def app_with_token():
    """App instance that requires WS_READ_TOKEN=secret123."""
    env = {"WS_READ_TOKEN": "secret123"}
    with patch.dict(os.environ, env, clear=False):
        from src.dashboard.app import create_app
        return create_app()


# ── Health endpoint ───────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, app):
        with TestClient(app) as client:
            resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_health_returns_ok_status(self, app):
        with TestClient(app) as client:
            resp = client.get("/api/health")
        assert resp.json()["status"] == "ok"

    def test_health_includes_version(self, app):
        with TestClient(app) as client:
            resp = client.get("/api/health")
        assert "version" in resp.json()


# ── WebSocket: no-token mode ──────────────────────────────────────────────────

class TestWebSocketNoToken:
    def test_ws_connects_and_receives_heartbeat(self, app):
        with TestClient(app) as client, client.websocket_connect("/ws/signals") as ws:
            msg = ws.receive_json()
        assert msg["type"] in ("heartbeat", "cycle_update")
        assert "timestamp" in msg

    def test_ws_message_has_data_key(self, app):
        with TestClient(app) as client, client.websocket_connect("/ws/signals") as ws:
            msg = ws.receive_json()
        assert "data" in msg


# ── WebSocket: token-gated mode ───────────────────────────────────────────────

class TestWebSocketTokenGate:
    def test_ws_rejects_missing_token(self, app_with_token):
        """Connection without ?token= must be closed with 4401."""
        with TestClient(app_with_token) as client, client.websocket_connect("/ws/signals") as ws:
            msg = ws.receive_json()
        assert msg.get("type") == "error" or msg.get("code") == 4401

    def test_ws_rejects_wrong_token(self, app_with_token):
        """Connection with wrong token must be rejected."""
        with TestClient(app_with_token) as client, \
             client.websocket_connect("/ws/signals?token=wrongtoken") as ws:
            msg = ws.receive_json()
        assert msg.get("type") == "error" or msg.get("code") == 4401

    def test_ws_accepts_correct_token(self, app_with_token):
        """Connection with correct token must succeed."""
        with TestClient(app_with_token) as client, \
             client.websocket_connect("/ws/signals?token=secret123") as ws:
            msg = ws.receive_json()
        assert msg["type"] in ("heartbeat", "cycle_update")


# ── Snapshot export ───────────────────────────────────────────────────────────

class TestSnapshotExport:
    @pytest.mark.asyncio
    async def test_run_cycle_writes_latest_json(self, tmp_path):
        """_run_cycle must write the snapshot file after a successful cycle."""
        import src.dashboard.app as dash_mod

        snap = tmp_path / "latest.json"
        fake_state = {"debate_consensus": "BUY", "debate_confidence": 0.8}

        with patch.dict(os.environ, {"_SNAPSHOT_PATH_OVERRIDE": str(snap)}), \
             patch("src.dashboard.app._run_cycle_inner",
                   new=AsyncMock(return_value=fake_state)):
            await dash_mod._run_cycle("BTC/USDT")

        out = json.loads(snap.read_text())
        assert out.get("debate_consensus") == "BUY"

    @pytest.mark.asyncio
    async def test_run_cycle_writes_valid_json_on_partial_state(self, tmp_path):
        """Even a minimal state must produce parseable JSON."""
        import src.dashboard.app as dash_mod

        snap = tmp_path / "latest.json"

        with patch.dict(os.environ, {"_SNAPSHOT_PATH_OVERRIDE": str(snap)}), \
             patch("src.dashboard.app._run_cycle_inner",
                   new=AsyncMock(return_value={})):
            await dash_mod._run_cycle("BTC/USDT")

        data = json.loads(snap.read_text())
        assert isinstance(data, dict)
