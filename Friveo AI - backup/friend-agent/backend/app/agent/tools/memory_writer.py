# =============================================================================
# app/agent/tools/memory_writer.py — Vector Memory Storage (Step 6)
# =============================================================================

import logging
from datetime import datetime, timezone

from app.agent.tools.embeddings import embed_text, build_memory_text
from app.db.qdrant_client import get_qdrant_client, upsert_memory
from app.db.mongo import get_sessions_collection, get_mood_logs_collection

logger = logging.getLogger(__name__)


async def write_session_memory(
    user_id: str,
    session_id: str,
    summary: str,
) -> bool:
    """
    Embed a session summary and store it in Qdrant.
    Called by summariser.py after writing the summary to MongoDB.
    Enriches the embedding with events and mood from MongoDB.
    """
    if not summary or not summary.strip():
        logger.debug(f"[MemoryWriter] Empty summary for {session_id}, skipping")
        return False

    client = get_qdrant_client()
    if client is None:
        logger.warning("[MemoryWriter] Qdrant not available — skipping")
        return False

    try:
        sessions_col = get_sessions_collection()
        mood_col     = get_mood_logs_collection()

        # Collect events from all turns in this session
        session_doc = await sessions_col.find_one(
            {"session_id": session_id},
            projection={"turns": 1, "created_at": 1}
        )

        events = []
        if session_doc:
            for turn in session_doc.get("turns", []):
                event = turn.get("extracted_event", {})
                if event.get("has_event") and event.get("title"):
                    events.append({
                        "title":     event["title"],
                        "date":      event.get("date", ""),
                        "has_event": True,
                    })

        # Get dominant mood from this session
        mood_cursor = mood_col.find(
            {"user_id": user_id, "session_id": session_id, "doc_type": "sentiment_snapshot"},
            sort=[("timestamp", -1)],
            limit=1,
        )
        mood_docs  = await mood_cursor.to_list(length=1)
        mood_label = mood_docs[0].get("label", "") if mood_docs else ""
        valence    = mood_docs[0].get("valence", 0.0) if mood_docs else 0.0

        session_date = (
            session_doc.get("created_at", "")[:10]
            if session_doc else
            datetime.now(timezone.utc).strftime("%Y-%m-%d")
        )

        # Build enriched text: summary + events + mood
        embed_content = build_memory_text(
            summary=summary,
            events=events,
            mood_label=mood_label,
        )

        # Embed with fallback chain
        vector, provider = await embed_text(
            text=embed_content,
            source=f"session:{session_id[:12]}"
        )

        event_titles = [e["title"] for e in events if e.get("title")]

        success = await upsert_memory(
            user_id=user_id,
            session_id=session_id,
            vector=vector,
            summary=summary,
            date=session_date,
            mood_label=mood_label,
            valence=valence,
            events=event_titles,
            provider=provider,
        )

        if success:
            logger.info(
                f"[MemoryWriter] Session {session_id[:12]} → Qdrant "
                f"(mood: {mood_label}, events: {event_titles}, provider: {provider})"
            )
        return success

    except Exception as e:
        logger.error(f"[MemoryWriter] Failed for {session_id}: {e}")
        return False


async def run_memory_writer_tool(state: dict) -> dict:
    """LangGraph-compatible wrapper — kept for graph compatibility."""
    return {}