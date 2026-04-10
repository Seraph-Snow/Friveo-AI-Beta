# =============================================================================
# app/agent/tools/summariser.py — Session Summariser
# =============================================================================
# WHAT THIS DOES:
#   Compresses a session's raw turns into a single paragraph.
#   This paragraph becomes session.summary in MongoDB.
#
# WHEN IT RUNS:
#   Lazily — when a NEW session starts, we summarise the PREVIOUS session.
#   The user never waits for this because it runs during context loading
#   for the new session, in parallel with other DB reads.
#
# WHY SUMMARISE AT ALL?
#   A 50-turn session has ~5000 words of conversation. You can't inject all
#   of that into the LLM context window for future sessions — it's too slow
#   and too expensive. A 100-word summary preserves the important facts:
#   what topics were discussed, what events were mentioned, what emotional
#   state the user was in. That's enough for the agent to feel continuous.
#
# WHAT GOES INTO THE SUMMARY:
#   - Key topics discussed
#   - Events mentioned (exam, meeting, puja etc.)
#   - Emotional arc (started stressed, ended more positive)
#   - Important personal details (relationships, situations)
#   - Anything the agent should remember next time
# =============================================================================

import logging
from datetime import datetime, timezone
from langchain_core.messages import HumanMessage

from app.core.config import settings
from app.db.mongo import get_sessions_collection, get_mood_logs_collection

logger = logging.getLogger(__name__)

# Use the same provider as the rest of the agent
# Summarisation is a focused task — flash model is perfect
def _get_llm():
    from langchain_ollama import ChatOllama
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0.3,
    )


SUMMARISE_PROMPT = """You are summarising a conversation between a user and their AI companion.
Write a single concise paragraph (max 120 words) capturing:
- Main topics discussed
- Any events or plans mentioned (exams, meetings, trips, celebrations)
- The user's emotional state and how it evolved
- Important personal details mentioned (relationships, situations, concerns)
- Anything the companion should remember for next time

Write in third person past tense. Be specific, not generic.
Bad:  "The user talked about their life and felt stressed."
Good: "User was stressed about an upcoming exam on March 18th, mentioned studying all day.
       Also shared that there's a puja at home this Sunday. Mood improved slightly toward
       the end after talking through study plans."

Conversation:
{conversation}

Summary (one paragraph, no bullet points):"""


async def summarise_session(session_id: str, user_id: str) -> str:
    """
    Summarise a session's turns into a paragraph.
    Writes the summary back to MongoDB and returns it.

    Args:
        session_id: the session to summarise
        user_id:    for logging

    Returns:
        The summary string, or "" if session has no turns
    """
    sessions_col = get_sessions_collection()
    session_doc = await sessions_col.find_one({"session_id": session_id})

    if not session_doc:
        logger.warning(f"[Summariser] Session {session_id} not found")
        return ""

    turns = session_doc.get("turns", [])
    if not turns:
        logger.debug(f"[Summariser] Session {session_id} has no turns, skipping")
        return ""

    # Don't re-summarise if already done
    existing_summary = session_doc.get("summary", "")
    if existing_summary and existing_summary.strip():
        logger.debug(f"[Summariser] Session {session_id} already summarised")
        return existing_summary

    # Format turns as readable conversation
    conversation_lines = []
    for turn in turns:
        user_msg = turn.get("user_message", "").strip()
        agent_msg = turn.get("agent_reply", "").strip()
        if user_msg:
            conversation_lines.append(f"User: {user_msg}")
        if agent_msg:
            conversation_lines.append(f"Companion: {agent_msg}")

    if not conversation_lines:
        return ""

    conversation_text = "\n".join(conversation_lines)

    # Trim if very long — summariser doesn't need the full text
    # Take first 20 and last 10 turns to capture arc
    if len(turns) > 30:
        first_lines = conversation_lines[:40]   # first 20 turns = 40 lines
        last_lines  = conversation_lines[-20:]  # last 10 turns = 20 lines
        conversation_text = (
            "\n".join(first_lines) +
            "\n[...middle of conversation...]\n" +
            "\n".join(last_lines)
        )

    prompt = SUMMARISE_PROMPT.format(conversation=conversation_text)

    try:
        llm = _get_llm()
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        summary = response.content.strip()

        # Write summary back to MongoDB
        await sessions_col.update_one(
            {"session_id": session_id},
            {"$set": {
                "summary":      summary,
                "summarised_at": datetime.now(timezone.utc).isoformat(),
            }}
        )

        logger.info(
            f"[Summariser] Session {session_id} summarised "
            f"({len(turns)} turns → {len(summary)} chars)"
        )

        # ── Commit to Qdrant vector memory ───────────────────────────────────
        # Import here to avoid circular imports at module load time
        try:
            from app.agent.tools.memory_writer import write_session_memory
            await write_session_memory(
                user_id=user_id,
                session_id=session_id,
                summary=summary,
            )
        except Exception as e:
            # Never let Qdrant failure break summarisation
            logger.warning(f"[Summariser] Qdrant write failed (non-fatal): {e}")

        return summary

    except Exception as e:
        logger.error(f"[Summariser] Failed for session {session_id}: {e}")
        return ""


async def get_last_session_summary(user_id: str, current_session_id: str) -> str:
    """
    Find the most recent completed session for this user
    (not the current one) and return its summary.

    Called at the start of a new session to provide cross-session continuity.
    If the last session has no summary yet, we summarise it now.
    """
    sessions_col = get_sessions_collection()

    # Find the most recent session that is NOT the current one
    # and has at least one turn (i.e. was a real conversation)
    cursor = sessions_col.find(
        {
            "user_id":    user_id,
            "session_id": {"$ne": current_session_id},
            "turns.0":    {"$exists": True},  # at least one turn
        },
        sort=[("created_at", -1)],
        limit=1,
    )
    sessions = await cursor.to_list(length=1)

    if not sessions:
        logger.debug(f"[Summariser] No previous session found for user {user_id}")
        return ""

    last_session = sessions[0]
    last_session_id = last_session["session_id"]

    # Use existing summary if available
    summary = last_session.get("summary", "")
    if summary and summary.strip():
        logger.info(f"[Summariser] Using existing summary for session {last_session_id}")
        return summary

    # Summarise now (lazy — first time this session is referenced)
    logger.info(f"[Summariser] Summarising previous session {last_session_id} lazily")
    summary = await summarise_session(last_session_id, user_id)
    return summary