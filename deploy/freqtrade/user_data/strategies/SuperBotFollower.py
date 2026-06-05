"""
SuperBotFollower.py  —  FreqTrade Strategy (GPL-3.0)
=====================================================
A thin "do-nothing" FreqTrade strategy. All entry/exit decisions come from
the Agentra DingBot via force_entry / force_exit REST calls.

FreqTrade handles:
  - CCXT order execution + retry logic
  - Position management (stop-loss, take-profit)
  - Trade history + SQLite persistence
  - FreqUI live monitoring

The SuperBot handles:
  - Signal generation (LangGraph agents)
  - Debate + risk assessment
  - Calling force_entry/force_exit via FreqTradeClient

NOTE: This file is licensed under GPL-3.0 (required by FreqTrade).
      It does NOT contain any SuperBot intelligence.
      The GPL boundary is the HTTP REST API. See docs/LICENSE_RATIONALE.md.
"""
# SPDX-License-Identifier: GPL-3.0
from freqtrade.strategy import IStrategy
import pandas as pd


class SuperBotFollower(IStrategy):
    """
    Passive strategy shell — defers all decisions to the SuperBot via REST.
    FreqTrade handles execution, position tracking, and stop-loss enforcement.
    """
    INTERFACE_VERSION = 3
    can_short = False
    timeframe = "5m"

    # Safety backstop — SuperBot normally manages exits via force_exit.
    # These are last-resort values if the SuperBot stops sending signals.
    minimal_roi = {"0": 0.10}   # 10% ROI as emergency exit
    stoploss    = -0.05          # Hard 5% stop-loss — SuperBot uses ATR stops normally
    trailing_stop = False

    def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """SuperBot does all TA — return dataframe unchanged."""
        return df

    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """No automatic entries — SuperBot uses force_entry REST calls."""
        df["enter_long"]  = 0
        df["enter_short"] = 0
        return df

    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """No automatic exits — SuperBot uses force_exit REST calls."""
        df["exit_long"]  = 0
        df["exit_short"] = 0
        return df
