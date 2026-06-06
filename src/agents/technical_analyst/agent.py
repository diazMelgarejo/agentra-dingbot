"""
agents/technical_analyst/agent.py  —  Step 3: TA Agent
========================================================
Produces two IndicatorSnapshots per cycle:

  technical    — 4h standard TA (EMA/BB/MACD/ATR)  → spot pipeline
  technical_5m — 5m fast TA (MACD(3,15,3)/VWAP/CVD) → Polymarket pipeline

Signal evaluation uses multi-timeframe confirmation:
  4h  signal forms the primary bias (higher weight)
  1h  signal acts as a trend filter   (alignment bonus)
  5m  signal confirms entry timing    (used by polymarket_agent)

The scoring system is rule-based and fully deterministic — no LLM required here.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import structlog

from core.state import IndicatorSnapshot, Signal, Timeframe

logger = structlog.get_logger(__name__)

_PRIMARY_TF     = "4h"     # main bias timeframe
_CONFIRM_TF     = "1h"     # trend confirmation timeframe
_FAST_TF        = "5m"     # Polymarket entry timing


# ── LangGraph node entry point ────────────────────────────────────────────────

async def run(state: dict[str, Any]) -> dict[str, Any]:
    from agents.technical_analyst.indicators import (
        compute_all_indicators,
        compute_fast_indicators,
    )

    symbol = state.get("symbol", "BTC/USDT")
    ohlcv  = state.get("ohlcv", {})
    new_errors = []

    # ── 4h standard analysis (primary bias) ───────────────────────────────────
    snap_4h: IndicatorSnapshot | None = None

    if _PRIMARY_TF not in ohlcv:
        msg = f"technical_analyst: no {_PRIMARY_TF!r} data in ohlcv"
        logger.warning("missing_primary_tf", tf=_PRIMARY_TF)
        new_errors.append(msg)
        return {"technical": None, "errors": new_errors}

    ind_4h = compute_all_indicators(ohlcv[_PRIMARY_TF])
    if ind_4h is None:
        new_errors.append("technical_analyst: 4h indicator compute failed")
        return {"technical": None, "errors": new_errors}

    # Optional 1h confirmation for multi-timeframe scoring bonus
    ind_1h = None
    if _CONFIRM_TF in ohlcv:
        ind_1h = compute_all_indicators(ohlcv[_CONFIRM_TF])

    sig_4h, conf_4h, reason_4h = _evaluate_standard(ind_4h, ind_1h)

    snap_4h = IndicatorSnapshot(
        symbol=symbol, timeframe=Timeframe.H4,
        timestamp=datetime.now(UTC),
        # Price
        close=ind_4h.get("close"),
        volume=ind_4h.get("volume"),
        # RSI
        rsi_14=ind_4h.get("rsi_14"),
        # Bollinger
        bb_upper=ind_4h.get("bb_upper"),
        bb_middle=ind_4h.get("bb_middle"),
        bb_lower=ind_4h.get("bb_lower"),
        bb_width=ind_4h.get("bb_width"),
        # EMAs
        ema_9=ind_4h.get("ema_9"),
        ema_21=ind_4h.get("ema_21"),
        ema_50=ind_4h.get("ema_50"),
        ema_200=ind_4h.get("ema_200"),
        # MACD
        macd_line=ind_4h.get("macd_line"),
        macd_signal=ind_4h.get("macd_signal"),
        macd_hist=ind_4h.get("macd_hist"),
        # Volatility
        atr_14=ind_4h.get("atr_14"),
        # Signal
        signal=sig_4h,
        confidence=conf_4h,
        reasoning=reason_4h,
    )

    logger.info("tech_4h_done",
                symbol=symbol,
                signal=sig_4h.value,
                confidence=f"{conf_4h:.1%}",
                ema_cross=ind_4h.get("ema_cross"),
                rsi=f"{ind_4h.get('rsi_14', 0):.1f}",
                bb_pct=f"{ind_4h.get('bb_pct', 0.5):.2f}")

    # ── 5m fast analysis (Polymarket entry timing) ─────────────────────────────
    snap_5m: IndicatorSnapshot | None = None

    if _FAST_TF in ohlcv:
        ind_5m = compute_fast_indicators(ohlcv[_FAST_TF])
        if ind_5m:
            sig_5m, conf_5m, reason_5m = _evaluate_fast(ind_5m)
            snap_5m = IndicatorSnapshot(
                symbol=symbol, timeframe=Timeframe.M5,
                timestamp=datetime.now(UTC),
                close=ind_5m.get("close"),
                volume=ind_5m.get("volume"),
                rsi_14=ind_5m.get("rsi_14"),
                macd_fast_hist=ind_5m.get("macd_fast_hist"),
                vwap=ind_5m.get("vwap"),
                cvd=ind_5m.get("cvd"),
                signal=sig_5m,
                confidence=conf_5m,
                reasoning=reason_5m,
            )
            logger.info("tech_5m_done",
                        symbol=symbol,
                        signal=sig_5m.value,
                        macd_fast=f"{ind_5m.get('macd_fast_hist', 0):.3f}",
                        rsi=f"{ind_5m.get('rsi_14', 50):.1f}")

    return {
        "technical":    snap_4h,   # spot trading pipeline
        "technical_5m": snap_5m,   # Polymarket pipeline
        "errors":       new_errors,
    }


# ── Standard 4h scoring ───────────────────────────────────────────────────────

def _evaluate_standard(
    ind: dict[str, Any],
    ind_confirm: dict[str, Any] | None = None,
) -> tuple[Signal, float, str]:
    """
    Rule-based scoring on the 4h timeframe.
    ind_confirm (1h) adds a small alignment bonus when both timeframes agree.

    Score range: roughly [-11, +11].
    Normalised confidence: |score| / 11.0, clamped to [0, 1].
    """
    score: float = 0.0
    reasons: list[str] = []
    close = ind.get("close") or 0.0

    # ── RSI ────────────────────────────────────────────────────────────────────
    rsi = ind.get("rsi_14")
    if rsi is not None:
        if rsi <= 20:
            score += 3.0
            reasons.append(f"RSI extreme oversold ({rsi:.1f})")
        elif rsi <= 25:
            score += 2.5
            reasons.append(f"RSI deeply oversold ({rsi:.1f})")
        elif rsi <= 30:
            score += 2.0
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi <= 40:
            score += 1.0
            reasons.append(f"RSI near oversold ({rsi:.1f})")
        elif rsi >= 80:
            score -= 3.0
            reasons.append(f"RSI extreme overbought ({rsi:.1f})")
        elif rsi >= 75:
            score -= 2.5
            reasons.append(f"RSI deeply overbought ({rsi:.1f})")
        elif rsi >= 70:
            score -= 2.0
            reasons.append(f"RSI overbought ({rsi:.1f})")
        elif rsi >= 60:
            score -= 0.5
            reasons.append(f"RSI elevated ({rsi:.1f})")

    # ── EMA alignment ──────────────────────────────────────────────────────────
    ema_cross = ind.get("ema_cross", "FLAT")
    if ema_cross == "BULL":
        score += 2.0
        reasons.append("EMA 9>21>50 (bull stack)")
    elif ema_cross == "BEAR":
        score -= 2.0
        reasons.append("EMA 9<21<50 (bear stack)")

    # EMA-200 trend filter (smaller weight — macro context only)
    e200 = ind.get("ema_200")
    if e200 and close:
        if close > e200 * 1.01:
            score += 0.5
            reasons.append("Price well above EMA-200")
        elif close > e200:
            score += 0.2
            reasons.append("Price above EMA-200")
        elif close < e200 * 0.99:
            score -= 0.5
            reasons.append("Price well below EMA-200")
        else:
            score -= 0.2
            reasons.append("Price below EMA-200")

    # ── Bollinger Bands ────────────────────────────────────────────────────────
    bb_pct = ind.get("bb_pct")   # 0 = at lower band, 1 = at upper band
    bb_width = ind.get("bb_width")
    if bb_pct is not None:
        if bb_pct <= 0.05:
            score += 2.0
            reasons.append(f"Price at/below lower BB ({bb_pct:.2f})")
        elif bb_pct <= 0.20:
            score += 1.0
            reasons.append(f"Price near lower BB ({bb_pct:.2f})")
        elif bb_pct >= 0.95:
            score -= 2.0
            reasons.append(f"Price at/above upper BB ({bb_pct:.2f})")
        elif bb_pct >= 0.80:
            score -= 1.0
            reasons.append(f"Price near upper BB ({bb_pct:.2f})")
    # BB squeeze: low volatility precedes breakout — informational only
    if bb_width is not None and bb_width < 0.02:
        reasons.append(f"BB squeeze (width={bb_width:.3f}) — breakout likely")

    # ── MACD (12,26,9) ─────────────────────────────────────────────────────────
    mhist = ind.get("macd_hist")
    mline = ind.get("macd_line")
    msig  = ind.get("macd_signal")
    if mhist is not None:
        score += 1.0 if mhist > 0 else -1.0
        reasons.append(f"MACD hist {mhist:+.2f} ({'bull' if mhist > 0 else 'bear'})")
    if mline is not None and msig is not None:
        if mline > msig:
            score += 0.5
            reasons.append("MACD line > signal (bull cross)")
        elif mline < msig:
            score -= 0.5
            reasons.append("MACD line < signal (bear cross)")

    # ── 1h confirmation bonus (multi-timeframe) ────────────────────────────────
    if ind_confirm:
        confirm_cross = ind_confirm.get("ema_cross", "FLAT")
        confirm_rsi   = ind_confirm.get("rsi_14")
        # Only apply bonus when 1h agrees with 4h direction
        if ema_cross == "BULL" and confirm_cross == "BULL":
            score += 0.5
            reasons.append("1h EMA also bullish (MTF confirm)")
        elif ema_cross == "BEAR" and confirm_cross == "BEAR":
            score -= 0.5
            reasons.append("1h EMA also bearish (MTF confirm)")
        # RSI divergence warning: 4h oversold but 1h already recovering
        if rsi and confirm_rsi:
            if rsi < 30 and confirm_rsi > 50:
                score += 0.3
                reasons.append("1h RSI recovering — divergence bullish")
            elif rsi > 70 and confirm_rsi < 50:
                score -= 0.3
                reasons.append("1h RSI weakening — divergence bearish")

    # ── Signal mapping ─────────────────────────────────────────────────────────
    if score >=  5.0:
        sig = Signal.STRONG_BUY
    elif score >=  1.5:
        sig = Signal.BUY
    elif score <= -5.0:
        sig = Signal.STRONG_SELL
    elif score <= -1.5:
        sig = Signal.SELL
    else :
                       sig = Signal.NEUTRAL

    confidence = min(abs(score) / 11.0, 1.0)
    reasoning  = " | ".join(reasons) if reasons else "Insufficient data"

    return sig, confidence, reasoning


# ── Fast 5m scoring ───────────────────────────────────────────────────────────

def _evaluate_fast(ind: dict[str, Any]) -> tuple[Signal, float, str]:
    """
    Score fast 5-min indicators for Polymarket entry timing.
    Weights: MACD(3,15,3) momentum + RSI/VWAP reversal + CVD order flow.
    Max score: 4.5.
    """
    score: float = 0.0
    reasons: list[str] = []

    mh    = ind.get("macd_fast_hist", 0.0) or 0.0
    rsi   = ind.get("rsi_14", 50.0) or 50.0
    close = ind.get("close", 0.0) or 0.0
    vwap  = ind.get("vwap", 0.0) or 0.0
    cvd   = ind.get("cvd", 0.0) or 0.0

    # ── MACD(3,15,3) momentum ─────────────────────────────────────────────────
    if mh > 0:
        score += min(1.5, 1.5 * (1 + abs(mh) / 50))  # scale with magnitude
        reasons.append(f"MACD(3,15,3) hist +{mh:.3f} bull")
    elif mh < 0:
        score -= min(1.5, 1.5 * (1 + abs(mh) / 50))
        reasons.append(f"MACD(3,15,3) hist {mh:.3f} bear")

    # ── RSI + VWAP reversal ───────────────────────────────────────────────────
    if vwap > 0 and close > 0:
        above_vwap = close > vwap
        if rsi < 30 and above_vwap:
            score += 1.5
            reasons.append(f"RSI oversold ({rsi:.1f}) + above VWAP")
        elif rsi < 40 and above_vwap:
            score += 0.8
            reasons.append(f"RSI near oversold ({rsi:.1f}) + above VWAP")
        elif rsi > 70 and not above_vwap:
            score -= 1.5
            reasons.append(f"RSI overbought ({rsi:.1f}) + below VWAP")
        elif rsi > 60 and not above_vwap:
            score -= 0.8
            reasons.append(f"RSI elevated ({rsi:.1f}) + below VWAP")

    # ── CVD order flow confirmation ────────────────────────────────────────────
    if cvd != 0:
        # Rising CVD = buyers accumulating = bullish; falling = selling pressure
        if cvd > 0:
            score += 0.5
            reasons.append(f"CVD positive ({cvd:+.0f}) buy pressure")
        else:
            score -= 0.5
            reasons.append(f"CVD negative ({cvd:+.0f}) sell pressure")

    # ── Signal mapping ─────────────────────────────────────────────────────────
    if score >=  2.5:
        sig = Signal.STRONG_BUY
    elif score >=  1.0:
        sig = Signal.BUY
    elif score <= -2.5:
        sig = Signal.STRONG_SELL
    elif score <= -1.0:
        sig = Signal.SELL
    else :
                       sig = Signal.NEUTRAL

    confidence = min(abs(score) / 4.5, 1.0)
    return sig, confidence, " | ".join(reasons) or "No fast signal"


# ── Backward-compatibility alias (tests import _evaluate) ─────────────────────
_evaluate = _evaluate_standard
