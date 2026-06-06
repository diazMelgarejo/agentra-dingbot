"""
core/config.py  —  Agentic SuperBot v0.3.0
Unified config: merges agentic-trader v0.2.0 + Polymarket Hybrid SuperBot.
All values are env-var driven
defaults are safe (paper/sandbox mode).
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExchangeConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EXCHANGE_", extra="ignore")
    name: str = "binance"
    api_key: str = ""
    api_secret: str = ""
    sandbox: bool = True


class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")
    provider: str = Field("none", alias="LLM_PROVIDER")   # none|ollama|openai — "none" uses the zero-key heuristic judge
    model: str = Field("llama3.1:8b", alias="LLM_MODEL")
    temperature: float = 0.1
    max_tokens: int = 4096
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    ollama_base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")

    @model_validator(mode="after")
    def _check_openai_key(self) -> LLMConfig:
        if self.provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set when LLM_PROVIDER=openai")
        return self


class TradingConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")
    # BTC/ETH spot pairs
    pairs: list[str] = ["BTC/USDT", "ETH/USDT"]
    timeframes: list[str] = ["5m", "1h", "4h", "1d"]
    max_position_size_pct: float = Field(25.0, alias="MAX_POSITION_SIZE_PCT")
    max_drawdown_pct: float = Field(10.0, alias="MAX_DRAWDOWN_PCT")
    risk_per_trade_pct: float = Field(1.0, alias="RISK_PER_TRADE_PCT")
    min_confidence_threshold: float = Field(0.3, alias="MIN_CONFIDENCE_THRESHOLD")


class PolymarketConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")
    private_key: str = Field("", alias="POLYMARKET_PRIVATE_KEY")
    proxy_address: str = Field("", alias="POLYMARKET_PROXY_ADDRESS")
    chain_id: int = Field(137, alias="POLYMARKET_CHAIN_ID")
    bankroll_usdc: float = Field(100.0, alias="BANKROLL_USDC")
    kelly_fraction: float = Field(0.25, alias="KELLY_FRACTION")
    min_edge_pct: float = Field(8.0, alias="MIN_EDGE_PCT")
    max_concurrent_trades: int = Field(3, alias="MAX_CONCURRENT_TRADES")
    daily_drawdown_limit_pct: float = Field(5.0, alias="DAILY_DRAWDOWN_LIMIT_PCT")
    # Polymarket API endpoints (stable, no env override needed)
    gamma_api: str = "https://gamma-api.polymarket.com"
    clob_api: str = "https://clob.polymarket.com"
    clob_ws_market: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    paper_mode: bool = Field(True, alias="PAPER_MODE")


class AlertConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")
    telegram_bot_token: str = Field("", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field("", alias="TELEGRAM_CHAT_ID")
    alpha_vantage_api_key: str = Field("", alias="ALPHA_VANTAGE_API_KEY")



class MLConfig(BaseSettings):
    """FreqAI-style ML signal bridge configuration."""
    model_config = SettingsConfigDict(extra="ignore")
    enabled: bool = Field(True, alias="ML_ENABLED")
    model_type: str = Field("lightgbm", alias="ML_MODEL_TYPE")   # lightgbm | sklearn | heuristic (auto-falls back)
    timeframe: str = Field("1h", alias="ML_TIMEFRAME")           # which OHLCV timeframe to train on
    lookback_bars: int = Field(500, alias="ML_LOOKBACK_BARS")
    label_horizon: int = Field(3, alias="ML_LABEL_HORIZON")      # predict direction this many bars ahead
    label_deadzone: float = Field(0.0, alias="ML_LABEL_DEADZONE")
    min_train_samples: int = Field(100, alias="ML_MIN_TRAIN_SAMPLES")
    retrain_interval_cycles: int = Field(50, alias="ML_RETRAIN_INTERVAL")
    prob_strong_buy: float = Field(0.65, alias="ML_PROB_STRONG_BUY")
    prob_buy: float = Field(0.55, alias="ML_PROB_BUY")
    prob_sell: float = Field(0.45, alias="ML_PROB_SELL")
    prob_strong_sell: float = Field(0.35, alias="ML_PROB_STRONG_SELL")
    model_dir: str = Field("data/models", alias="ML_MODEL_DIR")
    random_state: int = Field(42, alias="ML_RANDOM_STATE")



class FreqTradeConfig(BaseSettings):
    """Optional FreqTrade execution sidecar — auto-detected, never required."""
    model_config = SettingsConfigDict(extra="ignore")
    # "auto" = use it only if detected & reachable; "on" = require it; "off" = never use
    mode: str = Field("auto", alias="FREQTRADE_MODE")
    base_url: str = Field("http://localhost:8080", alias="FREQTRADE_URL")
    username: str = Field("superbot", alias="FREQTRADE_USERNAME")
    password: str = Field("superbot_password", alias="FREQTRADE_PASSWORD")
    mcp_enabled: bool = Field(False, alias="FREQTRADE_MCP_ENABLED")
    mcp_url: str = Field("http://localhost:8765", alias="FREQTRADE_MCP_URL")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    polymarket: PolymarketConfig = Field(default_factory=PolymarketConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    freqtrade: FreqTradeConfig = Field(default_factory=FreqTradeConfig)
    db_path: str = Field("data/trades.db", alias="DB_PATH")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    debug: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton — loaded once, shared across all modules."""
    return Settings()
