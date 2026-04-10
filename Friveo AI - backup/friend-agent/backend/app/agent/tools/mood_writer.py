# =============================================================================
# app/agent/tools/mood_writer.py — Mood Snapshot Writer
# =============================================================================
# WHAT THIS DOES:
#   After the sentiment tool runs, it returns a dict with valence, energy,
#   label etc. But nothing was writing that to MongoDB. This tool closes
#   that loop — it persists each sentiment result as a snapshot document
#   in the mood_logs collection.
#
# WHY A SEPARATE TOOL NOT INLINE IN graph.py?
#   Separation of concerns. The graph orchestrates, tools do the work.
#   Also makes it easy to test independently.
#
# WHAT mood_logs CONTAINS:
#   Two document types live in mood_logs:
#
#   doc_type: "sentiment_snapshot"
#     Written by this tool after each sentiment analysis
#     { user_id, session_id, valence, energy, label, tone,
#       sarcasm_detected, timestamp, doc_type }
#
#   doc_type: "agent_state"
#     Written by agent_state_updater node in graph.py
#     { user_id, mood, trust_level, openness, energy, last_impact, updated_at }
#
#   The context_builder reads sentiment_snapshots to build the mood trend
#   and reads agent_state to build the agent's current emotional context.
# =============================================================================

import logging
from datetime import datetime, timezone

from app.db.mongo import get_mood_logs_collection

logger = logging.getLogger(__name__)


async def write_mood_snapshot(
    user_id: str,
    session_id: str,
    sentiment: dict,
) -> bool:
    """
    Persist a sentiment analysis result to mood_logs.

    Args:
        user_id:    the user this snapshot belongs to
        session_id: which session this came from
        sentiment:  the full sentiment dict from the sentiment tool

    Returns:
        True if written successfully, False on error
    """
    if not sentiment:
        logger.debug("[MoodWriter] Empty sentiment dict, skipping write")
        return False

    # Only write if we have meaningful data
    valence = sentiment.get("valence")
    if valence is None:
        logger.debug("[MoodWriter] No valence in sentiment, skipping")
        return False

    try:
        mood_col = get_mood_logs_collection()

        snapshot = {
            "doc_type":          "sentiment_snapshot",
            "user_id":           user_id,
            "session_id":        session_id,
            "timestamp":         datetime.now(timezone.utc).isoformat(),
            # Core sentiment fields
            "valence":           valence,
            "energy":            sentiment.get("energy", "medium"),
            "label":             sentiment.get("label", "neutral"),
            "tone":              sentiment.get("tone", "neutral"),
            "intensity":         sentiment.get("intensity", "mild"),
            "sarcasm_detected":  sentiment.get("sarcasm_detected", False),
            "notes":             sentiment.get("notes", ""),
            # Message signal fields
            "caps_ratio":           sentiment.get("caps_ratio", 0.0),
            "punctuation_intensity": sentiment.get("punctuation_intensity", 0),
            "message_length_signal": sentiment.get("message_length_signal", "medium"),
        }

        await mood_col.insert_one(snapshot)

        logger.info(
            f"[MoodWriter] Snapshot written: "
            f"valence={valence} label={snapshot['label']} "
            f"user={user_id[:8]}..."
        )
        return True

    except Exception as e:
        logger.error(f"[MoodWriter] Failed to write snapshot: {e}")
        return False