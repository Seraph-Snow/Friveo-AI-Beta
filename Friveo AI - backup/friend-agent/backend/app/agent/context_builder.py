import logging
# =============================================================================
# app/agent/context_builder.py — Pre-Graph Context Assembly
# =============================================================================
# This runs BEFORE the LangGraph graph starts.
# It's not a node — it's a setup function that populates the initial state.
#
# WHY OUTSIDE THE GRAPH?
#   These are pure data fetches — no decisions, no LLM calls.
#   Putting them inside a graph node would work but adds unnecessary
#   graph complexity. Keep the graph for things that involve reasoning.
#
# WHAT IT DOES:
#   1. Loads user profile from Postgres (or Redis cache)
#   2. Loads recent turns from MongoDB
#   3. Loads agent emotional state from MongoDB
#   4. Loads recent mood history from MongoDB
#   5. Computes raw message signals (caps, punctuation etc.)
#   6. Assembles the initial AgentState object
# =============================================================================

import re
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.sql_models import User, PersonalityType
from app.db.mongo import get_sessions_collection, get_mood_logs_collection
from app.db.redis_client import cache_get, cache_set
logger = logging.getLogger(__name__)

from app.agent.prompts import (
    build_system_prompt, agent_state_to_prompt, PERSONA_PROMPTS
)
from app.agent.tools.summariser import get_last_session_summary
from app.agent.personality_engine import get_role_system_prompt, AGENT_ROLES


# =============================================================================
# DEFAULT AGENT EMOTIONAL STATE
# Used for new users who have no history yet
# =============================================================================
DEFAULT_AGENT_STATE = {
    "agent_mood":        "neutral",
    "agent_trust":       0.3,      # start cautious, build over time
    "agent_openness":    0.4,      # start measured, open up with trust
    "agent_energy":      "medium",
    "agent_last_impact": "This is a new relationship — starting fresh with warmth.",
}


# =============================================================================
# MESSAGE SIGNAL ANALYZER
# Pure Python — no LLM needed. Fast and deterministic.
# =============================================================================
def analyze_message_signals(message: str) -> dict:
    """
    Extract behavioral signals from message text that help the LLM
    understand HOW something was said, not just what was said.

    caps_ratio: ratio of uppercase letters — high = potential yelling/excitement
    punctuation_intensity: count of emotional punctuation (!?...) 
    message_length_signal: categorized length
    """
    if not message:
        return {
            "caps_ratio": 0.0,
            "punctuation_intensity": 0,
            "message_length_signal": "very_short"
        }

    # Caps ratio — only count alphabetic characters
    alpha_chars = [c for c in message if c.isalpha()]
    caps_ratio = (
        sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if alpha_chars else 0.0
    )

    # Punctuation intensity — emotional punctuation only
    exclamations = message.count('!')
    questions = message.count('?')
    ellipsis = message.count('...')
    punctuation_intensity = exclamations + questions + (ellipsis * 2)

    # Message length signal
    word_count = len(message.split())
    if word_count <= 3:
        length_signal = "very_short"
    elif word_count <= 10:
        length_signal = "short"
    elif word_count <= 30:
        length_signal = "medium"
    elif word_count <= 80:
        length_signal = "long"
    else:
        length_signal = "very_long"

    return {
        "caps_ratio": round(caps_ratio, 2),
        "punctuation_intensity": punctuation_intensity,
        "message_length_signal": length_signal,
    }


# =============================================================================
# MAIN CONTEXT BUILDER
# =============================================================================
async def build_context(
    user: User,
    message: str,
    session_id: str,
    db: AsyncSession,
) -> dict:
    """
    Assembles the complete initial state for the agent graph.

    Returns a dict matching AgentState fields (minus router/tool outputs
    which will be set by the graph nodes).
    """

    user_id = str(user.id)

    # -------------------------------------------------------------------------
    # 1. USER PROFILE — try Redis cache first, then Postgres
    # -------------------------------------------------------------------------
    cache_key = f"user:{user_id}:profile"
    cached_profile = await cache_get(cache_key)

    if cached_profile:
        user_profile = cached_profile
        agent_type_code = cached_profile.get("agent_type_code", "DEFAULT")
        persona_prompt_key = agent_type_code
    else:
        # Fetch personality types if available
        agent_type_code = "DEFAULT"
        personality_name = None

        if user.agent_persona_type_id:
            result = await db.execute(
                select(PersonalityType).where(
                    PersonalityType.id == user.agent_persona_type_id
                )
            )
            agent_type = result.scalar_one_or_none()
            if agent_type:
                agent_type_code = agent_type.code

        if user.personality_type_id:
            result = await db.execute(
                select(PersonalityType).where(
                    PersonalityType.id == user.personality_type_id
                )
            )
            user_type = result.scalar_one_or_none()
            if user_type:
                personality_name = user_type.name

        agent_role = getattr(user, "agent_role", "friend") or "friend"

        user_profile = {
            "display_name": user.display_name or "friend",
            "email": user.email,
            "timezone": user.timezone or "UTC",
            "is_onboarded": user.is_onboarded,
            "personality_name": personality_name,
            "agent_type_code": agent_type_code,
            "agent_role": agent_role,
        }

        # Cache for 10 minutes — profile rarely changes mid-session
        await cache_set(cache_key, user_profile, ttl_seconds=600)

    # -------------------------------------------------------------------------
    # 2. RECENT TURNS + CROSS-SESSION SUMMARY
    # -------------------------------------------------------------------------
    sessions_col = get_sessions_collection()
    session_doc = await sessions_col.find_one({"session_id": session_id})

    recent_turns = []
    session_summary = ""
    previous_session_summary = ""
    is_new_session = True

    if session_doc:
        turns = session_doc.get("turns", [])
        recent_turns = turns[-10:] if len(turns) > 10 else turns
        session_summary = session_doc.get("summary", "")
        is_new_session = len(turns) == 0
    else:
        is_new_session = True

    # On new session start — load previous session summary + proactive Qdrant retrieval
    proactive_memories = []
    if is_new_session:
        previous_session_summary = await get_last_session_summary(
            user_id=user_id,
            current_session_id=session_id,
        )
        if previous_session_summary:
            logger.info(
                f"[Context] Loaded previous session summary "
                f"({len(previous_session_summary)} chars)"
            )

        # Proactive retrieval — search Qdrant on session start even without keywords
        # This surfaces relevant past context before the user even asks for it
        try:
            from app.agent.tools.memory_retriever import retrieve_memories
            from app.db.qdrant_client import get_qdrant_client
            if get_qdrant_client() is not None:
                proactive_memories = await retrieve_memories(
                    message=message,
                    user_id=user_id,
                    session_id=session_id,
                    top_k=2,
                    score_threshold=0.70,  # higher threshold for proactive
                )
                if proactive_memories:
                    logger.info(
                        f"[Context] Proactive retrieval: {len(proactive_memories)} "
                        f"memories (scores: {[m['score'] for m in proactive_memories]})"
                    )
        except Exception as e:
            logger.debug(f"[Context] Proactive retrieval skipped: {e}")

    # -------------------------------------------------------------------------
    # 3. AGENT EMOTIONAL STATE — from MongoDB
    # -------------------------------------------------------------------------
    mood_col = get_mood_logs_collection()

    # Get agent state document
    agent_state_doc = await mood_col.find_one(
        {"user_id": user_id, "doc_type": "agent_state"}
    )

    if agent_state_doc:
        agent_state = {
            "agent_mood":        agent_state_doc.get("mood", "neutral"),
            "agent_trust":       agent_state_doc.get("trust_level", 0.3),
            "agent_openness":    agent_state_doc.get("openness", 0.4),
            "agent_energy":      agent_state_doc.get("energy", "medium"),
            "agent_last_impact": agent_state_doc.get("last_impact",
                                 "This relationship is still developing."),
        }
    else:
        agent_state = DEFAULT_AGENT_STATE.copy()

    # Build the natural language version for system prompt injection
    agent_state_prompt = agent_state_to_prompt(
        mood=agent_state["agent_mood"],
        trust=agent_state["agent_trust"],
        openness=agent_state["agent_openness"],
        energy=agent_state["agent_energy"],
        last_impact=agent_state["agent_last_impact"],
    )

    # -------------------------------------------------------------------------
    # 4. MOOD HISTORY — last 3 sentiment snapshots
    # -------------------------------------------------------------------------
    mood_cursor = mood_col.find(
        {"user_id": user_id, "doc_type": "sentiment_snapshot"},
        sort=[("timestamp", -1)],
        limit=3
    )
    mood_history = await mood_cursor.to_list(length=3)

    # Build a readable mood context string
    # WHY CAREFUL PHRASING HERE?
    #   The mood history informs the agent's awareness — it should NOT override
    #   the agent's response to the current message. If the user was stressed
    #   yesterday but is talking about a puja today, the agent should respond
    #   to the puja, not to yesterday's stress.
    #   We frame mood history as background awareness, not a directive.
    if mood_history:
        mood_labels = [m.get("label", "neutral") for m in mood_history]
        latest_valence = mood_history[0].get("valence", 0)

        # Describe the mood trend as background context, not a command
        mood_context = f"Background mood awareness: {' → '.join(reversed(mood_labels))}"

        # Only flag persistent negative mood (3 sessions all negative)
        # Single negative session is NOT enough to color all future responses
        all_negative = all(
            m.get("valence", 0) < -0.4
            for m in mood_history
        )
        if all_negative and len(mood_history) >= 2:
            mood_context += (
                "\nThis person has been going through a difficult stretch. "
                "Be warm but let them lead — don't assume they're still in that place."
            )
    else:
        mood_context = "No mood history yet — respond naturally to what they say."

    # -------------------------------------------------------------------------
    # 4b. ANALYTICS CONTEXT — weekly insight summary for agent awareness
    # -------------------------------------------------------------------------
    analytics_context = ""
    try:
        from app.agent.analytics_engine import build_analytics_context_for_prompt
        analytics_context = await build_analytics_context_for_prompt(user_id)
    except Exception:
        pass

    # Append analytics context to mood context
    if analytics_context:
        mood_context = mood_context + "\n" + analytics_context

    # -------------------------------------------------------------------------
    # 5. MESSAGE SIGNALS — pure Python analysis
    # -------------------------------------------------------------------------
    signals = analyze_message_signals(message)

    # -------------------------------------------------------------------------
    # 6. BUILD SYSTEM PROMPT
    # -------------------------------------------------------------------------
    # Combine current session summary + previous session summary
    # Current session summary: what was discussed in THIS session so far
    # Previous session summary: what was discussed LAST time (cross-session memory)
    combined_summary = ""
    if previous_session_summary:
        combined_summary += "Last conversation: " + previous_session_summary + "\n"
    if session_summary:
        combined_summary += f"This conversation so far: {session_summary}"

    # Build role layer — injected FIRST in system prompt
    agent_role = user_profile.get("agent_role", "friend")
    role_layer = get_role_system_prompt(agent_role)

    system_prompt = build_system_prompt(
        agent_type_code=agent_type_code,
        agent_state_prompt=agent_state_prompt,
        persona_prompt=PERSONA_PROMPTS.get(agent_type_code, PERSONA_PROMPTS["DEFAULT"]),
        mood_context=mood_context,
        memories=proactive_memories,  # populated by Qdrant on new session start
        session_summary=combined_summary,
        user_display_name=user_profile.get("display_name", "friend"),
        role_layer=role_layer,
    )

    # -------------------------------------------------------------------------
    # 7. ASSEMBLE INITIAL STATE
    # -------------------------------------------------------------------------
    return {
        # Input
        "user_id": user_id,
        "session_id": session_id,
        "message": message,
        "message_ts": datetime.now(timezone.utc).isoformat(),

        # User context
        "user_profile": user_profile,
        "persona_prompt": system_prompt,
        "recent_turns": recent_turns,
        "session_summary": session_summary,
        "mood_history": mood_history,
        "memories": [],   # populated by memory tool

        # Agent emotional state
        **agent_state,
        "agent_state_prompt": agent_state_prompt,

        # Message signals
        **signals,

        # Router defaults (overwritten by router node)
        "intent": "casual",
        "urgency": "low",
        "run_sentiment": False,
        "run_event_extractor": False,
        "run_memory_retriever": False,

        # Tool output defaults
        "sentiment": {},
        "extracted_event": {},
        "state_delta": {},

        # Response defaults
        "reply": "",
        "reply_ts": "",
        "langfuse_trace_id": "",
        "error": None,
    }