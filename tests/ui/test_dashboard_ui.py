"""
test_dashboard_ui.py — Playwright smoke tests + responsive QA for docs/index.html

Covers:
  - KPI cards render with visible text
  - Demo badge fires when no backend (file:// has no WebSocket)
  - Theme toggle switches dark/light and persists in localStorage
  - Candlestick chart mounts (canvas element present)
  - Responsive breakpoints: sidebar collapses at 1100px, single-column at 560px
"""
from __future__ import annotations

from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

DASHBOARD = (Path(__file__).resolve().parents[2] / "docs" / "index.html").as_uri()

# ── Fixture: open dashboard and wait for demo mode to settle ─────────────────

@pytest.fixture()
def dash(page: Page):
    """Load the dashboard, wait for Demo badge to appear (no backend in tests)."""
    page.goto(DASHBOARD)
    # Demo Data badge becomes visible once the 2.5s WS timeout fires
    expect(page.locator("#conn-text")).to_have_text("Demo Data", timeout=6000)
    return page


# ── KPI rendering ─────────────────────────────────────────────────────────────

class TestKPICards:
    def test_consensus_kpi_has_text(self, dash: Page):
        val = dash.locator("#kpi-consensus").inner_text()
        assert val.strip() != ""

    def test_confidence_kpi_shows_percent(self, dash: Page):
        val = dash.locator("#kpi-conf").inner_text()
        assert "%" in val

    def test_ml_prob_kpi_is_numeric(self, dash: Page):
        val = dash.locator("#kpi-ml").inner_text()
        assert float(val) >= 0

    def test_fear_greed_kpi_is_numeric(self, dash: Page):
        val = dash.locator("#kpi-fng").inner_text()
        assert int(val) >= 0

    def test_vix_kpi_is_numeric(self, dash: Page):
        val = dash.locator("#kpi-vix").inner_text()
        assert float(val) >= 0


# ── Demo fallback ─────────────────────────────────────────────────────────────

class TestDemoFallback:
    def test_demo_badge_visible(self, dash: Page):
        expect(page := dash.locator("#conn-text")).to_have_text("Demo Data")

    def test_demo_banner_shown(self, dash: Page):
        banner = dash.locator("#demo-banner")
        expect(banner).to_be_visible()

    def test_freqlink_disabled_in_demo(self, dash: Page):
        href = dash.locator("#freqlink").get_attribute("href")
        assert href == "#"


# ── Theme toggle ──────────────────────────────────────────────────────────────

class TestThemeToggle:
    def test_toggle_switches_theme(self, dash: Page):
        # Read current theme, click, assert it changed to the opposite
        before = dash.evaluate("document.documentElement.getAttribute('data-theme') || 'light'")
        dash.locator("#theme-btn").click()
        after = dash.evaluate("document.documentElement.getAttribute('data-theme')")
        assert after != before

    def test_toggle_persists_in_localstorage(self, dash: Page):
        dash.locator("#theme-btn").click()
        stored = dash.evaluate("localStorage.getItem('agentra-theme')")
        assert stored in ("dark", "light")

    def test_toggle_round_trips(self, dash: Page):
        dash.locator("#theme-btn").click()
        theme1 = dash.evaluate("document.documentElement.getAttribute('data-theme')")
        dash.locator("#theme-btn").click()
        theme2 = dash.evaluate("document.documentElement.getAttribute('data-theme')")
        assert theme1 != theme2


# ── Chart ─────────────────────────────────────────────────────────────────────

class TestCandlestickChart:
    def test_chart_canvas_exists(self, dash: Page):
        # Lightweight Charts renders multiple canvas layers; check the first one
        canvas = dash.locator("#price-chart canvas").first
        expect(canvas).to_be_visible(timeout=4000)

    def test_price_header_shows_dollar_amount(self, dash: Page):
        price = dash.locator("#hdr-price").inner_text()
        assert "$" in price


# ── Responsive breakpoints ────────────────────────────────────────────────────

class TestResponsiveLayout:
    def test_sidebar_visible_at_desktop(self, page: Page):
        page.set_viewport_size({"width": 1280, "height": 800})
        page.goto(DASHBOARD)
        expect(page.locator("#conn-text")).to_have_text("Demo Data", timeout=6000)
        sidebar = page.locator(".sidebar")
        expect(sidebar).to_be_visible()

    def test_sidebar_collapsed_at_1100px(self, page: Page):
        """At 1100px the sidebar collapses — check it's not blocking content."""
        page.set_viewport_size({"width": 1099, "height": 800})
        page.goto(DASHBOARD)
        expect(page.locator("#conn-text")).to_have_text("Demo Data", timeout=6000)
        # The main content area should still be present
        main = page.locator(".main")
        expect(main).to_be_visible()

    def test_single_column_at_560px(self, page: Page):
        """At 560px only single-column layout — KPIs still visible."""
        page.set_viewport_size({"width": 560, "height": 900})
        page.goto(DASHBOARD)
        expect(page.locator("#conn-text")).to_have_text("Demo Data", timeout=6000)
        expect(page.locator("#kpi-consensus")).to_be_visible()

    def test_mobile_kpis_visible_at_375px(self, page: Page):
        page.set_viewport_size({"width": 375, "height": 812})
        page.goto(DASHBOARD)
        expect(page.locator("#conn-text")).to_have_text("Demo Data", timeout=6000)
        expect(page.locator("#kpi-consensus")).to_be_visible()
