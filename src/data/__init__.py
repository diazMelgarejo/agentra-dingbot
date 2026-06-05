"""
data/  —  Step 2: Live Data Ingestion Layer
All data sources for the SuperBot:
  fetcher.py           — CCXT async wrapper: BTC/ETH OHLCV from Binance
  polymarket.py        — Gamma API + CLOB REST: market discovery, prices, spreads
  fear_greed.py        — CNN Fear & Greed Index (alternative.me) + VIX (yfinance)
  websocket_stream.py  — Polymarket L2 orderbook via WebSocket (live, auto-reconnect)
  snapshot.py          — Unified snapshot: all sources in one async call
"""
from data.fetcher           import fetch_ohlcv, fetch_ohlcv_multi_timeframe, fetch_ticker
from data.fear_greed        import fetch_fear_greed, fetch_vix, fetch_sentiment_snapshot
from data.polymarket        import (
    fetch_btc_eth_markets, fetch_yes_price, fetch_orderbook_snapshot,
    enrich_markets_with_prices, find_farmable_markets, fetch_polymarket_snapshot,
)
from data.snapshot          import fetch_full_snapshot
from data.websocket_stream  import LocalOrderbook, orderbook_stream

__all__ = [
    "fetch_ohlcv", "fetch_ohlcv_multi_timeframe", "fetch_ticker",
    "fetch_fear_greed", "fetch_vix", "fetch_sentiment_snapshot",
    "fetch_btc_eth_markets", "fetch_yes_price", "fetch_orderbook_snapshot",
    "enrich_markets_with_prices", "find_farmable_markets", "fetch_polymarket_snapshot",
    "fetch_full_snapshot",
    "LocalOrderbook", "orderbook_stream",
]
