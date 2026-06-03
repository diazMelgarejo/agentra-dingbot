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
from typing import Any, Dict, Tuple

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


async def run(state: Dict[str, Any]) -> Dict[str, Any]:
    symbol   = state.get("symbol", "BTC/USDT")
    evidence = _compile_evidence(state)

    try:
        # Run bull and bear in parallel for speed
        bull_task = asyncio.create_task(_call_agent("bull", symbol, evidence))
        bear_task = asyncio.create_task(_call_agent("bear", symbol, evidence))
        bull_case, bear_case = await asyncio.gather(bull_task, bear_task)

        sig, conf, reason = await _judge(symbol, bull_case, bear_case)

    except Exception as exc:
        logger.error("debate_failed", error=str(exc))
        # Graceful fallback: use technical signal
        tech = state.get("technical")
        sig  = tech.signal     if tech else Signal.NEUTRAL
        conf = tech.confidence if tech else 0.0
        bull_case = bear_case = ""
        reason = f"Debate failed ({exc}); using technical fallback"

    logger.info("debate_done", symbol=symbol, consensus=sig.value, confidence=f"{conf:.1%}")

    return {
        "bull_case":         bull_case,
        "bear_case":         bear_case,
        "debate_consensus":  sig,
        "debate_confidence": conf,
    }


# ─── Evidence compilation ─────────────────────────────────────────────────────

def _compile_evidence(state: Dict[str, Any]) -> str:
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


async def _judge(symbol: str, bull: str, bear: str) -> Tuple[Signal, float, str]:
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
