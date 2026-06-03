"""
tests/test_data_ingestion.py  —  Step 2: Live Data Ingestion
All tests are fully mocked — no API keys or network required.
Covers: CCXT fetcher, Fear&Greed, VIX, Polymarket REST, unified snapshot.
"""
from __future__ import annotations
import pytest
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import asynccontextmanager


# ── Factories ─────────────────────────────────────────────────────────────────

def ohlcv_df(n=200) -> pd.DataFrame:
    rng   = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = 45_000 + np.cumsum(rng.standard_normal(n) * 100)
    sp    = close * 0.001
    return pd.DataFrame(
        {"open": close-sp, "high": close+sp*2, "low": close-sp*2,
         "close": close, "volume": rng.uniform(1, 50, n)},
        index=dates
    )


def raw_ohlcv(n=200):
    import time
    base = int(time.time() * 1000) - n * 300_000
    p    = 45_000.0
    rows = []
    rng  = np.random.default_rng(0)
    for i in range(n):
        p += rng.standard_normal() * 100
        rows.append([base + i*300_000, p-50, p+100, p-100, p, 10.0])
    return rows


@asynccontextmanager
async def fake_exchange_ctx(exc_mock):
    yield exc_mock


# ── CCXT Fetcher ──────────────────────────────────────────────────────────────

class TestCCXTFetcher:

    def test_raw_to_df_columns_and_dtypes(self):
        from data.fetcher import _raw_to_df
        df = _raw_to_df(raw_ohlcv(50))
        assert list(df.columns) == ["open","high","low","close","volume"]
        assert all(df[c].dtype == float for c in df.columns)
        assert df.index.name == "timestamp"

    def test_df_index_utc_aware(self):
        from data.fetcher import _raw_to_df
        df = _raw_to_df(raw_ohlcv(10))
        assert str(df.index.tz) == "UTC"

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_returns_empty_on_error(self):
        from data.fetcher import fetch_ohlcv
        mock = AsyncMock()
        mock.fetch_ohlcv.side_effect = Exception("network error")
        with patch("data.fetcher._exchange_ctx", return_value=fake_exchange_ctx(mock)):
            assert await fetch_ohlcv("BTC/USDT", "1h", 10) == []

    @pytest.mark.asyncio
    async def test_multi_tf_dict_keyed_by_timeframe(self):
        from data.fetcher import fetch_ohlcv_multi_timeframe
        mock = AsyncMock()
        mock.fetch_ohlcv = AsyncMock(return_value=raw_ohlcv(200))
        with patch("data.fetcher._exchange_ctx", return_value=fake_exchange_ctx(mock)), \
             patch("data.fetcher.asyncio.sleep", AsyncMock()):
            result = await fetch_ohlcv_multi_timeframe("BTC/USDT", ["5m","1h"])
        assert set(result) == {"5m","1h"}
        assert len(result["5m"]) == 200

    @pytest.mark.asyncio
    async def test_multi_tf_skips_empty_bars(self):
        from data.fetcher import fetch_ohlcv_multi_timeframe
        mock = AsyncMock()
        mock.fetch_ohlcv = AsyncMock(return_value=[])
        with patch("data.fetcher._exchange_ctx", return_value=fake_exchange_ctx(mock)), \
             patch("data.fetcher.asyncio.sleep", AsyncMock()):
            assert await fetch_ohlcv_multi_timeframe("BTC/USDT", ["5m"]) == {}

    @pytest.mark.asyncio
    async def test_fetch_ticker_returns_none_on_error(self):
        from data.fetcher import fetch_ticker
        mock = AsyncMock()
        mock.fetch_ticker.side_effect = Exception("boom")
        with patch("data.fetcher._exchange_ctx", return_value=fake_exchange_ctx(mock)):
            assert await fetch_ticker("BTC/USDT") is None


# ── Fear & Greed / VIX ───────────────────────────────────────────────────────

class TestFearGreed:

    @pytest.mark.asyncio
    async def test_returns_value_int_in_range(self):
        from data.fear_greed import fetch_fear_greed
        payload = {"data": [{"value":"35","value_classification":"Fear","timestamp":"0"}]}
        with patch("data.fear_greed.aiohttp.ClientSession") as cls:
            inst = AsyncMock()
            resp = AsyncMock()
            resp.json   = AsyncMock(return_value=payload)
            resp.raise_for_status = MagicMock()
            cm  = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=resp)
            cm.__aexit__  = AsyncMock(return_value=False)
            sess = MagicMock()
            sess.get = MagicMock(return_value=cm)
            sess.__aenter__ = AsyncMock(return_value=sess)
            sess.__aexit__  = AsyncMock(return_value=False)
            cls.return_value = sess
            result = await fetch_fear_greed()
        assert isinstance(result["value"], int)
        assert 0 <= result["value"] <= 100
        assert result["classification"] == "Fear"

    @pytest.mark.asyncio
    async def test_defaults_neutral_on_error(self):
        from data.fear_greed import fetch_fear_greed
        with patch("data.fear_greed.aiohttp.ClientSession") as cls:
            sess = MagicMock()
            sess.__aenter__ = AsyncMock(side_effect=Exception("timeout"))
            sess.__aexit__  = AsyncMock(return_value=False)
            cls.return_value = sess
            result = await fetch_fear_greed()
        assert result["value"] == 50

    def test_vix_returns_float(self):
        from data.fear_greed import fetch_vix
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame(
            {"Close": [22.5, 23.1, 21.8]},
            index=pd.date_range("2024-01-01", periods=3)
        )
        with patch("data.fear_greed.yf") as mock_yf:
            mock_yf.Ticker.return_value = mock_ticker
            result = fetch_vix()
        assert isinstance(result, float)
        assert result == pytest.approx(21.8)

    def test_vix_returns_none_on_error(self):
        from data.fear_greed import fetch_vix
        with patch("data.fear_greed.yf") as mock_yf:
            mock_yf.Ticker.side_effect = Exception("down")
            assert fetch_vix() is None

    @pytest.mark.parametrize("vix,expected_risk,expected_mult", [
        (15.0, "NORMAL",   1.0),
        (31.0, "ELEVATED", 0.5),
        (41.0, "EXTREME",  0.0),
    ])
    @pytest.mark.asyncio
    async def test_vix_risk_levels(self, vix, expected_risk, expected_mult):
        from data.fear_greed import fetch_sentiment_snapshot
        with patch("data.fear_greed.fetch_fear_greed", AsyncMock(return_value={
            "value": 50, "classification": "Neutral", "timestamp": ""
        })), patch("data.fear_greed.fetch_vix_async", AsyncMock(return_value=vix)):
            r = await fetch_sentiment_snapshot()
        assert r["vix_risk_level"]  == expected_risk
        assert r["size_multiplier"] == expected_mult

    @pytest.mark.asyncio
    async def test_sentiment_snapshot_keys(self):
        from data.fear_greed import fetch_sentiment_snapshot
        with patch("data.fear_greed.fetch_fear_greed", AsyncMock(return_value={
            "value": 30, "classification": "Fear", "timestamp": ""
        })), patch("data.fear_greed.fetch_vix_async", AsyncMock(return_value=18.5)):
            r = await fetch_sentiment_snapshot()
        assert {"fear_greed","vix","vix_risk_level","size_multiplier"} <= r.keys()


# ── Polymarket REST ───────────────────────────────────────────────────────────

class TestPolymarketFetcher:

    def test_compute_spread_normal(self):
        from data.polymarket import compute_spread
        snap = {
            "bids": [{"price":"0.48","size":"100"},{"price":"0.46","size":"50"}],
            "asks": [{"price":"0.52","size":"80"}, {"price":"0.54","size":"30"}],
        }
        assert abs(compute_spread(snap) - 0.04) < 0.001

    def test_compute_spread_empty(self):
        from data.polymarket import compute_spread
        assert compute_spread({}) is None
        assert compute_spread({"bids":[], "asks":[]}) is None

    @pytest.mark.asyncio
    async def test_yes_price_near_resolved_returns_none(self):
        from data.polymarket import fetch_yes_price
        for p in ["0.005", "0.995", "0.01", "0.99"]:
            sess = MagicMock()
            resp = AsyncMock()
            resp.json = AsyncMock(return_value={"price": p})
            resp.raise_for_status = MagicMock()
            cm = MagicMock(); cm.__aenter__ = AsyncMock(return_value=resp); cm.__aexit__ = AsyncMock(return_value=False)
            sess.get = MagicMock(return_value=cm)
            assert await fetch_yes_price(sess, "tok") is None, f"price {p} should be None"

    @pytest.mark.asyncio
    async def test_yes_price_error_returns_none(self):
        from data.polymarket import fetch_yes_price
        sess = MagicMock()
        sess.get.side_effect = Exception("network")
        assert await fetch_yes_price(sess, "tok") is None

    @pytest.mark.asyncio
    async def test_btc_eth_market_filter(self):
        from data.polymarket import fetch_btc_eth_markets
        raw = [
            {"question":"Will BTC go up or down?","id":"1","conditionId":"c1","clobTokenIds":["t1"],"endDate":"","volume24hr":1000},
            {"question":"Will ETH be up in 5min?","id":"2","conditionId":"c2","clobTokenIds":["t2"],"endDate":"","volume24hr":500},
            {"question":"US election 2026?","id":"3","conditionId":"c3","clobTokenIds":["t3"],"endDate":"","volume24hr":2000},
        ]
        resp = AsyncMock(); resp.json = AsyncMock(return_value=raw); resp.raise_for_status = MagicMock()
        cm = MagicMock(); cm.__aenter__ = AsyncMock(return_value=resp); cm.__aexit__ = AsyncMock(return_value=False)
        sess = MagicMock(); sess.get = MagicMock(return_value=cm)
        markets = await fetch_btc_eth_markets(sess, tags=["crypto"])
        assert len(markets) == 2
        assert all("btc" in m.question.lower() or "eth" in m.question.lower() for m in markets)

    @pytest.mark.asyncio
    async def test_farmable_markets_filters_by_spread(self):
        """Only markets with spread < 0.06 should be returned as farmable."""
        from data.polymarket import find_farmable_markets
        from core.state import PolymarketMarket
        m1 = PolymarketMarket(token_id="tok1", question="BTC up?", yes_price=0.50, volume_24h=1000)
        m2 = PolymarketMarket(token_id="tok2", question="ETH up?", yes_price=0.48, volume_24h=500)
        wide_book  = {"bids":[{"price":"0.45","size":"50"}],"asks":[{"price":"0.55","size":"50"}]}  # spread=0.10
        tight_book = {"bids":[{"price":"0.49","size":"50"}],"asks":[{"price":"0.52","size":"50"}]}  # spread=0.03
        resp_wide  = AsyncMock(); resp_wide.json  = AsyncMock(return_value=wide_book);  resp_wide.raise_for_status  = MagicMock()
        resp_tight = AsyncMock(); resp_tight.json = AsyncMock(return_value=tight_book); resp_tight.raise_for_status = MagicMock()
        cm_wide  = MagicMock(); cm_wide.__aenter__  = AsyncMock(return_value=resp_wide);  cm_wide.__aexit__  = AsyncMock(return_value=False)
        cm_tight = MagicMock(); cm_tight.__aenter__ = AsyncMock(return_value=resp_tight); cm_tight.__aexit__ = AsyncMock(return_value=False)
        sess = MagicMock()
        sess.get = MagicMock(side_effect=[cm_wide, cm_tight])
        with patch("data.polymarket.asyncio.sleep", AsyncMock()):
            farmable = await find_farmable_markets(sess, [m1, m2])
        assert len(farmable) == 1
        assert farmable[0]["market"].token_id == "tok2"


# ── Unified Snapshot ──────────────────────────────────────────────────────────

class TestUnifiedSnapshot:

    @pytest.mark.asyncio
    async def test_all_keys_present(self):
        from data.snapshot import fetch_full_snapshot
        with patch("data.snapshot.fetch_ohlcv_multi_timeframe", AsyncMock(return_value={"5m":ohlcv_df()})), \
             patch("data.snapshot.fetch_sentiment_snapshot",     AsyncMock(return_value={"fear_greed":{"value":45},"vix":20.0,"vix_risk_level":"NORMAL"})), \
             patch("data.snapshot.fetch_polymarket_snapshot",    AsyncMock(return_value={"markets":[],"enriched_markets":[],"farmable_markets":[]})):
            r = await fetch_full_snapshot("BTC/USDT")
        assert {"ohlcv","sentiment","polymarket","errors"} <= r.keys()
        assert isinstance(r["errors"], list)

    @pytest.mark.asyncio
    async def test_polymarket_failure_non_fatal(self):
        from data.snapshot import fetch_full_snapshot
        with patch("data.snapshot.fetch_ohlcv_multi_timeframe", AsyncMock(return_value={"5m":ohlcv_df()})), \
             patch("data.snapshot.fetch_sentiment_snapshot",     AsyncMock(return_value={})), \
             patch("data.snapshot.fetch_polymarket_snapshot",    AsyncMock(side_effect=Exception("API down"))):
            r = await fetch_full_snapshot("BTC/USDT", include_polymarket=True)
        assert "ohlcv" in r
        assert len(r["errors"]) > 0

    @pytest.mark.asyncio
    async def test_no_polymarket_call_when_disabled(self):
        from data.snapshot import fetch_full_snapshot
        with patch("data.snapshot.fetch_ohlcv_multi_timeframe", AsyncMock(return_value={})), \
             patch("data.snapshot.fetch_sentiment_snapshot",     AsyncMock(return_value={})), \
             patch("data.snapshot.fetch_polymarket_snapshot",    AsyncMock()) as mock_pm:
            await fetch_full_snapshot("BTC/USDT", include_polymarket=False)
        mock_pm.assert_not_called()


# ── Integration: data → indicators pipeline ──────────────────────────────────

class TestDataPipelineIntegration:

    def test_ohlcv_df_feeds_indicators_correctly(self):
        from agents.technical_analyst.indicators import compute_all_indicators
        ind = compute_all_indicators(ohlcv_df(200))
        assert ind is not None
        assert "rsi_14" in ind
        assert 0.0 <= ind["rsi_14"] <= 100.0, f"RSI out of range: {ind['rsi_14']}"
        assert ind["bb_lower"] < ind["bb_middle"] < ind["bb_upper"]
        assert ind["atr_14"] > 0

    def test_short_df_returns_none(self):
        from agents.technical_analyst.indicators import compute_all_indicators
        assert compute_all_indicators(ohlcv_df(10)) is None

    @pytest.mark.asyncio
    async def test_full_snapshot_data_flows_to_state(self):
        from data.snapshot import fetch_full_snapshot
        tfs  = {"5m":ohlcv_df(200),"1h":ohlcv_df(100),"4h":ohlcv_df(200),"1d":ohlcv_df(365)}
        sent = {"fear_greed":{"value":28,"classification":"Fear"},"vix":24.5,"vix_risk_level":"NORMAL","size_multiplier":1.0}
        pm   = {"markets":[],"enriched_markets":[],"farmable_markets":[],"total_discovered":0}
        with patch("data.snapshot.fetch_ohlcv_multi_timeframe", AsyncMock(return_value=tfs)), \
             patch("data.snapshot.fetch_sentiment_snapshot",     AsyncMock(return_value=sent)), \
             patch("data.snapshot.fetch_polymarket_snapshot",    AsyncMock(return_value=pm)):
            snap = await fetch_full_snapshot("BTC/USDT")
        assert "4h" in snap["ohlcv"] and "5m" in snap["ohlcv"]
        assert snap["sentiment"]["fear_greed"]["value"] == 28
        assert snap["sentiment"]["vix"] == 24.5
        assert snap["errors"] == []
        # Verify 4h OHLCV can drive indicators
        from agents.technical_analyst.indicators import compute_all_indicators
        ind = compute_all_indicators(snap["ohlcv"]["4h"])
        assert ind is not None
        assert ind["close"] > 0
