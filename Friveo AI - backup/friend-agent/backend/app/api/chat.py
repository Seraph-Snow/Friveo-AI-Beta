# =============================================================================
# app/api/chat.py — Chat Endpoint
# =============================================================================
# This is the single endpoint the frontend calls for every message.
# It orchestrates: auth → context building → graph execution → session save
# =============================================================================

import uuid
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres import get_db
from app.db.mongo import get_sessions_collection
from app.models.schemas import ChatMessageRequest, ChatMessageResponse
from app.api.auth import get_current_user
from app.models.sql_models import User
from app.agent.context_builder import build_context
from app.agent.graph import agent_graph
from app.agent.tools.mood_writer import write_mood_snapshot

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatMessageResponse)
async def chat(
    request: ChatMessageRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Send a message and receive the agent's response.

    Flow:
    1. Validate auth (get_current_user dependency)
    2. Get or create session ID
    3. Build context (DB reads, no LLM)
    4. Run LangGraph agent
    5. Save turn to MongoDB (async, doesn't block response)
    6. Return reply
    """

    # -------------------------------------------------------------------------
    # SESSION MANAGEMENT
    # -------------------------------------------------------------------------
    # If client sends a session_id, continue that session.
    # If not, create a new one.
    session_id = request.session_id or f"session_{uuid.uuid4().hex[:12]}"
    sessions_col = get_sessions_collection()

    # ── 30-minute session boundary check ─────────────────────────────────
    # If client sends a session_id but that session is > 30 minutes old,
    # treat it as a new session. Prevents stale session reuse.
    if request.session_id:
        existing = await sessions_col.find_one({"session_id": session_id})
        if existing:
            last_updated = existing.get("last_updated", existing.get("created_at"))
            if last_updated:
                try:
                    last_dt = datetime.fromisoformat(
                        last_updated.replace("Z", "+00:00")
                    )
                    age_seconds = (
                        datetime.now(timezone.utc) - last_dt
                    ).total_seconds()
                    if age_seconds > 1800:  # 30 minutes
                        session_id = f"session_{uuid.uuid4().hex[:12]}"
                        logger.info(
                            f"[Chat] Session expired after {age_seconds/60:.0f}min "
                            f"— creating new session {session_id}"
                        )
                except Exception:
                    pass  # If date parsing fails, keep original session_id

    # Ensure session document exists in MongoDB
    await sessions_col.update_one(
        {"session_id": session_id},
        {"$setOnInsert": {
            "session_id":  session_id,
            "user_id":     str(current_user.id),
            "turns":       [],
            "summary":     "",
            "created_at":  datetime.now(timezone.utc).isoformat(),
        }},
        upsert=True,
    )

    # -------------------------------------------------------------------------
    # BUILD CONTEXT — assemble everything before the graph runs
    # -------------------------------------------------------------------------
    try:
        initial_state = await build_context(
            user=current_user,
            message=request.content,
            session_id=session_id,
            db=db,
        )
    except Exception as e:
        logger.error(f"Context build failed for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load conversation context",
        )

    # -------------------------------------------------------------------------
    # RUN AGENT GRAPH
    # -------------------------------------------------------------------------
    try:
        result = await agent_graph.ainvoke(initial_state)
    except Exception as e:
        logger.error(f"Agent graph failed for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent encountered an error processing your message",
        )

    reply = result.get("reply", "I'm here — something went wrong on my end though.")
    message_id = f"msg_{uuid.uuid4().hex[:12]}"

    # -------------------------------------------------------------------------
    # SAVE TURN TO MONGODB — fire and don't wait
    # -------------------------------------------------------------------------
    # We don't await this — the user gets their response immediately.
    # MongoDB save happens in the background.
    # In production this would be a Celery task.
    import asyncio
    asyncio.create_task(_save_turn(
        session_id=session_id,
        user_id=str(current_user.id),
        user_message=request.content,
        agent_reply=reply,
        message_id=message_id,
        intent=result.get("intent", "casual"),
        sentiment=result.get("sentiment", {}),
        extracted_event=result.get("extracted_event", {}),
    ))

    # -------------------------------------------------------------------------
    # SCHEDULE EVENT REMINDER if event was extracted
    # -------------------------------------------------------------------------
    event = result.get("extracted_event", {})
    if event and event.get("has_event") and event.get("date"):
        asyncio.create_task(_schedule_reminder(
            user_id=str(current_user.id),
            event=event,
        ))

    return ChatMessageResponse(
        content=reply,
        session_id=session_id,
        message_id=message_id,
        timestamp=datetime.now(timezone.utc),
    )


async def _save_turn(
    session_id: str,
    user_id: str,
    user_message: str,
    agent_reply: str,
    message_id: str,
    intent: str,
    sentiment: dict,
    extracted_event: dict,
):
    """Save a conversation turn to MongoDB and write mood snapshot if sentiment ran."""
    try:
        sessions_col = get_sessions_collection()
        turn = {
            "message_id":      message_id,
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "user_message":    user_message,
            "agent_reply":     agent_reply,
            "intent":          intent,
            "sentiment":       sentiment,
            "extracted_event": extracted_event,
        }
        await sessions_col.update_one(
            {"session_id": session_id},
            {
                "$push": {"turns": turn},
                "$set":  {"last_updated": datetime.now(timezone.utc).isoformat()},
            }
        )
        logger.debug(f"Turn saved to session {session_id}")

        # Write mood snapshot if sentiment tool ran this turn
        # This is what populates the mood_logs collection that the
        # context_builder reads to build the 3-session mood trend
        if sentiment and sentiment.get("valence") is not None:
            await write_mood_snapshot(
                user_id=user_id,
                session_id=session_id,
                sentiment=sentiment,
            )

    except Exception as e:
        logger.error(f"Failed to save turn: {e}")


async def _schedule_reminder(user_id: str, event: dict):
    """Queue a reminder for a detected event."""
    try:
        from app.db.mongo import get_events_collection
        events_col = get_events_collection()
        await events_col.insert_one({
            "user_id":       user_id,
            "title":         event.get("title"),
            "event_date":    event.get("date"),
            "event_time":    event.get("time"),
            "event_type":    event.get("event_type"),
            "reminder_sent": False,
            "created_at":    datetime.now(timezone.utc).isoformat(),
        })
        logger.info(f"Event saved for user {user_id}: {event.get('title')}")
        # TODO Step 7: queue actual Celery reminder task here
    except Exception as e:
        logger.error(f"Failed to schedule reminder: {e}")