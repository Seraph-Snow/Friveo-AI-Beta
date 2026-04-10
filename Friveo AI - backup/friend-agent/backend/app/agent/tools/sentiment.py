# =============================================================================
# app/agent/tools/sentiment.py — Sentiment Analysis Tool
# =============================================================================
# WHY A STANDALONE MODULE NOT INLINE IN graph.py?
#   1. Testable independently — you can call analyze_sentiment() directly
#      without running the whole graph
#   2. Reusable — the journal agent and summariser will also need sentiment
#   3. Swappable — if you want to try a different approach (fine-tuned model,
#      external API), you change one file not the whole graph
#
# DESIGN CHOICE: LLM not a Python library
#   Libraries like VADER score "I'm fine" as positive.
#   They have no concept of conversation context or sarcasm.
#   "I'm fine." said after "everything's falling apart" is not positive.
#   The LLM understands all of this naturally.
#   Cost: ~50-100 tokens per call. Worth it for quality.
# =============================================================================

import json
import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from app.core.config import settings
from app.agent.prompts import SENTIMENT_PROMPT

logger = logging.getLogger(__name__)

# Small fast model for classification — consistent, cheap, structured
_llm = ChatOllama(
    base_url=settings.ollama_base_url,
    model=settings.ollama_model,
    temperature=0.1,
    format="json",
)

# Default result returned on any failure
_DEFAULT_SENTIMENT = {
    "valence": 0.0,
    "energy": "medium",
    "label": "neutral",
    "sarcasm_detected": False,
    "tone": "neutral",
    "intensity": "mild",
    "notes": "",
}


def _compute_signals(message: str) -> dict:
    """
    Pure Python signal extraction — no LLM needed.
    These give the LLM behavioral hints it can't infer from text alone.

    caps_ratio:           How much is uppercase — high means yelling or excitement
    punctuation_intensity: Count of !, ?, ... — emotional intensity marker
    message_length_signal: Categorized word count
    """
    if not message:
        return {"caps_ratio": 0.0, "punctuation_intensity": 0,
                "message_length_signal": "very_short"}

    alpha = [c for c in message if c.isalpha()]
    caps_ratio = (
        sum(1 for c in alpha if c.isupper()) / len(alpha)
        if alpha else 0.0
    )

    punctuation_intensity = (
        message.count('!') +
        message.count('?') +
        message.count('...') * 2    # ellipsis = trailing off / suppression
    )

    word_count = len(message.split())
    if word_count <= 3:
        length = "very_short"    # terse — could be dismissive or just brief
    elif word_count <= 10:
        length = "short"
    elif word_count <= 30:
        length = "medium"
    elif word_count <= 80:
        length = "long"
    else:
        length = "very_long"    # user needs to vent — don't rush them

    return {
        "caps_ratio": round(caps_ratio, 2),
        "punctuation_intensity": punctuation_intensity,
        "message_length_signal": length,
    }


def _format_messages_for_analysis(
    current_message: str,
    recent_turns: list,
    max_turns: int = 5,
) -> str:
    """
    Format recent conversation turns + current message as a block of text
    for the LLM to analyze together.

    WHY INCLUDE RECENT TURNS?
      Sentiment is deeply contextual. "I'm fine" means something very different
      if the last 3 messages were about a breakup vs a good day at work.
      The LLM needs context to score accurately.

    NOTE ON KEY NAMES:
      MongoDB turns are stored as { user_message, agent_reply }.
      We map these to user/assistant roles for the sentiment prompt.
    """
    lines = []
    turns_to_include = recent_turns[-max_turns:] if recent_turns else []

    for turn in turns_to_include:
        user_msg = turn.get("user_message", "").strip()
        agent_msg = turn.get("agent_reply", "").strip()
        if user_msg:
            lines.append(f"user: {user_msg}")
        if agent_msg:
            lines.append(f"assistant: {agent_msg}")

    lines.append(f"user: {current_message}")
    return "\n".join(lines)


async def analyze_sentiment(
    message: str,
    recent_turns: list = None,
) -> dict:
    """
    Analyze the emotional tone of a message in context.

    Args:
        message:      The current user message
        recent_turns: Last N conversation turns for context

    Returns:
        dict with keys:
            valence (-1.0 to 1.0) — negative to positive
            energy (low/medium/high)
            label (3-5 word description)
            sarcasm_detected (bool)
            tone (warm/neutral/cold/distressed/excited/frustrated/sad/playful)
            intensity (mild/moderate/strong)
            notes (anything unusual in the emotional pattern)
            caps_ratio, punctuation_intensity, message_length_signal (signals)
    """
    if recent_turns is None:
        recent_turns = []

    signals = _compute_signals(message)
    messages_text = _format_messages_for_analysis(message, recent_turns)

    prompt = SENTIMENT_PROMPT.format(
        messages=messages_text,
        caps_ratio=signals["caps_ratio"],
        punctuation_intensity=signals["punctuation_intensity"],
        message_length_signal=signals["message_length_signal"],
    )

    try:
        response = await _llm.ainvoke([HumanMessage(content=prompt)])

        # Safe JSON parse
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw.strip())

        # Merge signals into result so the graph has everything in one dict
        # Merge signals into result
        result.update(signals)

        # Guarantee all keys exist — model sometimes omits optional fields
        # This prevents KeyErrors anywhere downstream that reads sentiment
        for key, default in _DEFAULT_SENTIMENT.items():
            if key not in result:
                logger.debug(f"[Sentiment] Missing key '{key}', using default: {default}")
                result[key] = default

        # Clamp valence to valid range in case model hallucinates out-of-range values
        result["valence"] = max(-1.0, min(1.0, float(result.get("valence", 0.0))))

        logger.info(
            f"[Sentiment] label={result.get('label')} | "
            f"valence={result.get('valence')} | "
            f"sarcasm={result.get('sarcasm_detected')}"
        )
        return result

    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"[Sentiment] Failed: {e}")
        return {**_DEFAULT_SENTIMENT, **signals}


async def run_sentiment_tool(state: dict) -> dict:
    """
    LangGraph-compatible wrapper.
    Takes AgentState, returns dict of state updates.
    """
    result = await analyze_sentiment(
        message=state["message"],
        recent_turns=state.get("recent_turns", []),
    )
    return {"sentiment": result}