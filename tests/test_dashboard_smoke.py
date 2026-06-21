"""
tests/test_dashboard_smoke.py  —  Playwright UI Smoke Tests
============================================================
Tests the static docs/index.html dashboard against real Chromium.
Covers: page load, KPIs render, theme toggle, snapshot fallback, chart mount,
Polymarket panel, risk table.

Run:
    pytest tests/test_dashboard_smoke.py -v -s
    # or via make:
    make smoke-test

Design choices:
  - Serves docs/ from a real Python HTTP server (no file:// quirks)
  - No live backend needed — tests demo/snapshot mode only
  - Tests browser behaviour (DOM, CSS), not unit logic
  - Each test is independent (fresh page load)
"""
from __future__ import annotations

import json
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_DOCS = _ROOT / "docs"
_PORT = 8997   # dedicated port for smoke tests to avoid collisions

# ── HTTP server fixture ───────────────────────────────────────────────────────

class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *a): pass   # silence request logs during tests

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(_DOCS), **kwargs)


@pytest.fixture(scope="module")
def server():
    """Start an HTTP server serving docs/ for the duration of the test module."""
    srv = HTTPServer(("127.0.0.1", _PORT), _QuietHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    time.sleep(0.3)
    yield f"http://127.0.0.1:{_PORT}"
    srv.shutdown()


@pytest.fixture
def page(server):
    """Fresh Playwright page per test."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--no-sandbox"])
        ctx = browser.new_context(
            # Disable service workers to avoid cache issues
            service_workers="block",
        )
        pg = ctx.new_page()
        pg.goto(f"{server}/index.html", wait_until="domcontentloaded", timeout=10_000)
        yield pg
        pg.close()
        browser.close()


# ── Smoke tests ───────────────────────────────────────────────────────────────

class TestDashboardSmoke:
    """
    Smoke tests for the Agentra DingBot static dashboard.
    All tests run in demo mode (no backend, no WebSocket).
    """

    def test_page_title_loads(self, page):
        """Page must have the correct title."""
        assert "DingBot" in page.title() or "Agentra" in page.title()

    def test_brand_name_visible(self, page):
        """Sidebar brand name must be visible."""
        brand = page.locator(".brand-name").first
        assert brand.is_visible()

    def test_kpi_consensus_renders(self, page):
        """Consensus KPI must render with text (demo or live)."""
        # Wait up to 4s for demo mode to kick in
        page.wait_for_selector("#kpi-consensus:not(:empty)", timeout=4000)
        text = page.locator("#kpi-consensus").inner_text()
        assert text.strip() != ""

    def test_kpi_confidence_renders(self, page):
        """Confidence KPI must show a percentage value."""
        page.wait_for_selector("#kpi-conf:not(:empty)", timeout=4000)
        text = page.locator("#kpi-conf").inner_text()
        assert "%" in text

    def test_candlestick_chart_mounts(self, page):
        """Lightweight Charts canvas must be present after render."""
        # Chart creates a canvas element inside #price-chart
        page.wait_for_selector("#price-chart canvas, #price-chart .empty",
                                timeout=8000)
        canvas = page.query_selector("#price-chart canvas")
        empty  = page.query_selector("#price-chart .empty")
        # Either canvas (chart loaded) or empty (offline) is acceptable
        assert canvas is not None or empty is not None

    def test_theme_toggle_switches_attribute(self, page):
        """Theme toggle must flip data-theme between dark and light."""
        # Get initial theme
        initial = page.evaluate("document.documentElement.getAttribute('data-theme')")
        # Click the toggle
        page.locator("#theme-btn").click()
        time.sleep(0.2)
        after = page.evaluate("document.documentElement.getAttribute('data-theme')")
        assert initial != after, f"Theme did not change (was {initial}, still {after})"

    def test_theme_persists_to_localStorage(self, page):
        """Theme choice must be saved to localStorage."""
        page.locator("#theme-btn").click()
        time.sleep(0.2)
        stored = page.evaluate("localStorage.getItem('agentra-theme')")
        assert stored in ("dark", "light"), f"localStorage key not set: {stored!r}"

    def test_connection_badge_renders(self, page):
        """Connection status badge must be present (live or demo or snapshot)."""
        badge = page.locator("#conn")
        assert badge.is_visible()
        text = page.locator("#conn-text").inner_text()
        assert text in ("Live", "Demo Data", "Snapshot")

    def test_demo_fallback_fires_without_backend(self, page):
        """Without a live WebSocket, demo mode must activate within 5 seconds."""
        # Wait for the badge to settle (timeout is 2.5s in the JS)
        page.wait_for_function(
            "() => ['Demo Data','Snapshot'].includes(document.getElementById('conn-text')?.textContent)",
            timeout=5000
        )
        text = page.locator("#conn-text").inner_text()
        assert text in ("Demo Data", "Snapshot")

    def test_four_agent_cards_present(self, page):
        """All four agent signal pills must be present in the DOM."""
        for agent_id in ["ta-sig", "se-sig", "oc-sig", "ml-sig"]:
            el = page.query_selector(f"#{agent_id}")
            assert el is not None, f"Missing agent pill: #{agent_id}"

    def test_polymarket_panel_has_rows(self, page):
        """Polymarket panel must have at least one market row after demo loads."""
        page.wait_for_selector(".pm-row", timeout=5000)
        rows = page.query_selector_all(".pm-row")
        assert len(rows) >= 1, "No Polymarket rows rendered"

    def test_risk_table_has_five_rows(self, page):
        """Risk table must have all 5 rows (position, SL, TP, R:R, VIX)."""
        # Wait for risk render to finish
        page.wait_for_selector("#risk-pos", timeout=5000)
        for row_id in ["risk-pos", "risk-sl", "risk-tp", "risk-rr", "risk-vix"]:
            el = page.query_selector(f"#{row_id}")
            assert el is not None, f"Missing risk row: #{row_id}"

    def test_snapshot_json_serveable(self, server):
        """docs/data/latest.json must be served with HTTP 200 from the server."""
        import urllib.request
        url = f"{server}/data/latest.json?t=123"
        with urllib.request.urlopen(url, timeout=3) as r:
            assert r.status == 200
            payload = json.loads(r.read())
            assert "data" in payload
            assert "meta" in payload

    def test_demo_data_matches_dashboard_shape(self, server):
        """latest.json must contain all required top-level keys."""
        import urllib.request
        with urllib.request.urlopen(f"{server}/data/latest.json", timeout=3) as r:
            payload = json.loads(r.read())
        data = payload["data"]
        for key in ["symbol", "debate_consensus", "technical", "sentiment",
                    "onchain", "ml", "risk", "polymarket", "ohlcv_4h"]:
            assert key in data, f"latest.json missing key: {key}"

    def test_sidebar_nav_items_present(self, page):
        """All 6 sidebar nav items must be visible."""
        items = page.query_selector_all(".nav-item")
        assert len(items) >= 6, f"Expected ≥6 nav items, got {len(items)}"

    def test_footer_data_sources_present(self, page):
        """Footer must list all 4 data sources."""
        footer = page.locator(".foot").inner_text()
        for src in ["Binance", "Polymarket", "F&G", "VIX"]:
            assert src in footer, f"Footer missing source: {src}"
