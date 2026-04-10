# =============================================================================
# app/agent/tools/event_extractor.py — Event Extraction Tool
# =============================================================================
# STRATEGY: Regex first, LLM fallback
#
# WHY NOT LLM ONLY?
#   "I have a meeting tomorrow at 3pm" — regex catches this perfectly.
#   Calling an LLM for something deterministic wastes tokens and adds latency.
#
# WHY NOT REGEX ONLY?
#   "after the long weekend", "sometime before my sister visits" — regex
#   can't resolve these to actual dates. The LLM can, with timezone context.
#
# The pipeline:
#   1. Quick regex scan — does this message even mention time?
#   2. If yes → LLM call to extract structured event with resolved date
#   3. If no clear time reference → short-circuit, return has_event: false
#
# WHY EVENTS MATTER FOR THE AGENT:
#   Detected events get saved to MongoDB and scheduled as Celery reminders.
#   Tomorrow morning the agent proactively messages: "Good luck on your exam!"
#   This is what transforms the agent from reactive to genuinely present.
# =============================================================================

import re
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from app.core.config import settings
from app.agent.prompts import EVENT_EXTRACTOR_PROMPT

logger = logging.getLogger(__name__)

_llm = ChatOllama(
    base_url=settings.ollama_base_url,
    model=settings.ollama_model,
    temperature=0.1,
    format="json",
)

_DEFAULT_EVENT = {
    "has_event": False,
    "title": None,
    "date": None,
    "time": None,
    "event_type": None,
    "confidence": 0.0,
    "reminder_lead_hours": 12,
}

# =============================================================================
# NOTE: No regex pre-filtering here.
# =============================================================================
# We previously had a TIME_REGEX gate that skipped the LLM if no
# time reference was found. This caused "puja this sunday", "annual day",
# "farewell" etc. to be missed entirely.
#
# The event extractor is a dedicated focused LLM call — let it always run.
# It returns has_event:false cleanly when nothing is there. That is the
# correct behavior. Regex pre-filtering belongs in the router, not here.
# =============================================================================

# Event type keywords — used only for logging hints, not for gating
_EVENT_KEYWORDS = {
    "exam":        ["exam", "test", "quiz", "finals", "midterm", "assessment"],
    "meeting":     ["meeting", "standup", "call", "sync", "catch up", "interview"],
    "appointment": ["appointment", "doctor", "dentist", "hospital", "clinic"],
    "deadline":    ["deadline", "due", "submit", "hand in", "hand-in"],
    "social":      ["birthday", "party", "wedding", "dinner", "lunch", "date"],
    "trip":        ["flight", "travel", "trip", "vacation", "holiday", "leave"],
}


def _guess_event_type(message: str) -> Optional[str]:
    """
    Guess the event type from keywords — faster than asking the LLM.
    Used as a hint in the extraction prompt.
    """
    message_lower = message.lower()
    for event_type, keywords in _EVENT_KEYWORDS.items():
        if any(kw in message_lower for kw in keywords):
            return event_type
    return "other"


async def extract_event(
    message: str,
    user_timezone: str = "UTC",
) -> dict:
    """
    Extract a scheduled event from a message.

    Args:
        message:        The user's message text
        user_timezone:  User's timezone for resolving relative dates

    Returns:
        dict with keys: has_event, title, date, time, event_type,
                        confidence, reminder_lead_hours
    """

    # Get current datetime for relative date resolution
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Step 3: LLM extraction
    prompt = EVENT_EXTRACTOR_PROMPT.format(
        message=message,
        timezone=user_timezone,
        current_datetime=now,
    )

    try:
        response = await _llm.ainvoke([HumanMessage(content=prompt)])

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        event = json.loads(raw.strip())

        # Validate — if confidence is too low, treat as no event
        # Threshold is 0.2 (not 0.4) because cultural/informal events
        # like "puja", "annual day", "farewell" may score lower but are
        # still valid events worth tracking
        if event.get("confidence", 0) < 0.2:
            logger.info(f"[Events] Confidence {event.get('confidence')} too low, discarding")
            return _DEFAULT_EVENT.copy()

        # Guarantee all keys exist — prevents downstream KeyErrors
        for key, default in _DEFAULT_EVENT.items():
            if key not in event:
                event[key] = default

        if event.get("has_event"):
            logger.info(
                f"[Events] Extracted: '{event.get('title')}' "
                f"on {event.get('date')} "
                f"(type: {event.get('event_type')}, "
                f"confidence: {event.get('confidence')})"
            )
        else:
            logger.info(
                f"[Events] No event found "
                f"(confidence: {event.get('confidence', 0)}, "
                f"notes: {event.get('notes', 'none')})"
            )

        return event

    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"[Events] Extraction failed: {e}")
        return _DEFAULT_EVENT.copy()


async def run_event_extractor_tool(state: dict) -> dict:
    """
    LangGraph-compatible wrapper.
    Takes AgentState, returns dict of state updates.
    """
    timezone = state.get("user_profile", {}).get("timezone", "UTC")

    result = await extract_event(
        message=state["message"],
        user_timezone=timezone,
    )
    return {"extracted_event": result}