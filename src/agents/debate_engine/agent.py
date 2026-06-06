"""
agents/debate_engine/agent.py
──────────────────────────────
Orchestrates a structured Bull vs Bear debate using an LLM, then has a
"head trader" judge adjudicate the final signal.

If the LLM call fails, the agent falls back to the technical analyst signal
so the pipeline never stalls.

Supported LLM providers: ollama (local), openai (cloud).
"""
from __future__ import annotations

import asyncio
from typing import Any

import structlog

from core.state import Signal

logger = structlog.get_logger(__name__)

# ─── Prompt templates ──────────────────────────────────────────────────────────

_BULL_SYSTEM = (
    "You are a BULLISH crypto analyst. Given the market evidence below, make the "
    "strongest possible case for buying {symbol}. Be specific: cite indicator values, "
    "momentum, and risk/reward. Keep your response under 200 words."
)

_BEAR_SYSTEM = (
    "You are a BEARISH crypto analyst. Given the market evidence below, make the "
    "strongest possible case AGAINST buying {symbol} (or for selling). Be specific: "
    "cite indicator values, risks, and potential downside. Keep under 200 words."
)

_JUDGE_SYSTEM = (
    "You are HEAD TRADER with final authority over {symbol}. You have heard both the "
    "bull and bear cases. Weigh them objectively and output EXACTLY this format on a "
    "single line with no extra text:\n"
    "SIGNAL|CONFIDENCE|REASONING\n"
    "Where SIGNAL ∈ [STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL], "
    "CONFIDENCE is a float 0.0–1.0, and REASONING is one concise sentence."
)


async def run(state: dict[str, Any]) -> dict[str, Any]:
    from core.config import get_settings
    symbol   = state.get("symbol", "BTC/USDT")
    provider = get_settings().llm.provider
    evidence = _compile_evidence(state)

    # ── Zero-key path: no LLM configured → deterministic heuristic judge ───────
    if provider == "none":
        sig, conf, reason = _heuristic_judge(state)
        logger.info("debate_done", symbol=symbol, mode="heuristic",
                    consensus=sig.value, confidence=f"{conf:.1%}")
        return {
            "bull_case":         f"[heuristic] weighted analyst vote → {sig.value}",
            "bear_case":         f"[heuristic] {reason}",
            "debate_consensus":  sig,
            "debate_confidence": conf,
        }

    # ── LLM path: ollama (local) or openai (cloud) ─────────────────────────────
    try:
        bull_task = asyncio.create_task(_call_agent("bull", symbol, evidence))
        bear_task = asyncio.create_task(_call_agent("bear", symbol, evidence))
        bull_case, bear_case = await asyncio.gather(bull_task, bear_task)
        sig, conf, reason = await _judge(symbol, bull_case, bear_case)
    except Exception as exc:
        logger.warning("debate_llm_failed_using_heuristic", error=str(exc))
        # Graceful fallback: deterministic heuristic judge (never stalls, no keys)
        sig, conf, reason = _heuristic_judge(state)
        bull_case = bear_case = ""
        reason = f"LLM unavailable ({exc}); heuristic judge → {reason}"

    logger.info("debate_done", symbol=symbol, consensus=sig.value, confidence=f"{conf:.1%}")

    return {
        "bull_case":         bull_case,
        "bear_case":         bear_case,
        "debate_consensus":  sig,
        "debate_confidence": conf,
    }


# ─── Zero-key heuristic judge ─────────────────────────────────────────────────

# Numeric mapping for weighted voting across analysts.
_SIGNAL_SCORE = {
    Signal.STRONG_BUY:  2.0,
    Signal.BUY:         1.0,
    Signal.NEUTRAL:     0.0,
    Signal.SELL:       -1.0,
    Signal.STRONG_SELL:-2.0,
}

# Per-source weights — technical leads, ML and sentiment confirm, on-chain nudges.
_SOURCE_WEIGHTS = {
    "technical": 1.0,
    "ml":        0.9,
    "sentiment": 0.7,
    "onchain":   0.5,
}


def _heuristic_judge(state: dict[str, Any]) -> tuple[Signal, float, str]:
    """
    Deterministic consensus from the analyst snapshots — no LLM, no API key.

    Each analyst contributes (signal_score × its_confidence × source_weight).
    The weighted average maps back to a Signal
    |avg| scales confidence.
    This is the default judge and also the universal fallback when an LLM
    provider is configured but unreachable.
    """
    weighted_sum = 0.0
    weight_total = 0.0
    contributions = []

    for key, weight in _SOURCE_WEIGHTS.items():
        snap = state.get(key)
        if snap is None:
            continue
        sig = getattr(snap, "signal", None)
        conf = float(getattr(snap, "confidence", 0.0) or 0.0)
        if sig is None:
            continue
        score = _SIGNAL_SCORE.get(sig, 0.0)
        contribution = score * conf * weight
        weighted_sum += contribution
        weight_total += weight * conf
        if conf > 0:
            contributions.append(f"{key}={sig.value}@{conf:.2f}")

    if weight_total == 0:
        return Signal.NEUTRAL, 0.0, "no analyst signals available"

    avg = weighted_sum / weight_total   # ∈ [-2, +2]

    if   avg >=  1.3:
        consensus = Signal.STRONG_BUY
    elif avg >=  0.4:
        consensus = Signal.BUY
    elif avg <= -1.3:
        consensus = Signal.STRONG_SELL
    elif avg <= -0.4:
        consensus = Signal.SELL
    else:
        consensus = Signal.NEUTRAL

    confidence = min(abs(avg) / 2.0, 1.0)
    reason = f"weighted vote ({', '.join(contributions) or 'all neutral'}) avg={avg:+.2f}"
    return consensus, round(confidence, 4), reason


# ─── Evidence compilation ─────────────────────────────────────────────────────

def _compile_evidence(state: dict[str, Any]) -> str:
    """Format agent snapshots into a compact evidence block for the LLM."""
    parts = []
    for key, label in [("technical", "TECHNICAL"), ("sentiment", "SENTIMENT"), ("onchain", "ON-CHAIN")]:
        item = state.get(key)
        if item:
            parts.append(
                f"[{label}] Signal: {item.signal.value} (confidence: {item.confidence:.2f})\n"
                f"  → {item.reasoning}"
            )
    # Optional ML evidence (Step 5) — only included when an MLSnapshot is present,
    # so the always-on technical/sentiment/on-chain block is preserved unchanged.
    ml = state.get("ml")
    if ml:
        prob = "n/a" if ml.prob_up is None else f"{ml.prob_up:.3f}"
        parts.append(
            f"[ML] Signal: {ml.signal.value} (confidence: {ml.confidence:.2f}, P(up)={prob})\n"
            f"  → {ml.reasoning}"
        )
    return "\n\n".join(parts) if parts else "No evidence available."


# ─── LLM calls ────────────────────────────────────────────────────────────────

async def _call_agent(role: str, symbol: str, evidence: str) -> str:
    from core.config import get_settings
    s = get_settings()
    system = (_BULL_SYSTEM if role == "bull" else _BEAR_SYSTEM).format(symbol=symbol)
    return await _llm(system, evidence, s)


async def _judge(symbol: str, bull: str, bear: str) -> tuple[Signal, float, str]:
    from core.config import get_settings
    s      = get_settings()
    system = _JUDGE_SYSTEM.format(symbol=symbol)
    user   = f"BULL CASE:\n{bull}\n\nBEAR CASE:\n{bear}"

    try:
        raw = await _llm(system, user, s)
        parts = [p.strip() for p in raw.strip().split("|")]
        if len(parts) < 2:
            raise ValueError(f"Unexpected judge response: {raw!r}")
        sig_str  = parts[0].upper()
        conf_str = parts[1]
        reason   = parts[2] if len(parts) > 2 else "No reasoning provided"
        return Signal(sig_str), max(0.0, min(1.0, float(conf_str))), reason
    except Exception as exc:
        logger.warning("judge_parse_failed", error=str(exc))
        return Signal.NEUTRAL, 0.2, f"Judge parse failed: {exc}"


async def _llm(system: str, user: str, settings) -> str:
    """Route to the configured LLM provider."""
    if settings.llm.provider == "openai":
        return await _openai(system, user, settings)
    return await _ollama(system, user, settings)


async def _openai(system: str, user: str, s) -> str:
    import httpx

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {s.llm.openai_api_key}"},
            json={
                "model":       s.llm.model,
                "temperature": s.llm.temperature,
                "max_tokens":  s.llm.max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


async def _ollama(system: str, user: str, s) -> str:
    import httpx

    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            f"{s.llm.ollama_base_url}/api/chat",
            json={
                "model":  s.llm.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
