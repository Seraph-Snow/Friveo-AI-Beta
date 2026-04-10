# =============================================================================
# app/api/sessions.py — Session History Endpoints
# =============================================================================
# ENDPOINTS:
#   GET /sessions              → list of past sessions with summaries
#   GET /sessions/{session_id} → full turn history for one session
#   GET /sessions/{session_id}/summary → just the summary
#
# USED BY:
#   Frontend Step 8 — "Past conversations" sidebar
#   Journal agent Step 7 — needs today's sessions to write the journal
# =============================================================================

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional

from app.db.mongo import get_sessions_collection
from app.api.auth import get_current_user
from app.models.sql_models import User

router = APIRouter(prefix="/sessions", tags=["Sessions"])


@router.get("")
async def list_sessions(
    limit: int = Query(default=20, le=100),
    skip:  int = Query(default=0),
    current_user: User = Depends(get_current_user),
):
    """
    List past sessions for the current user, newest first.
    Returns summary only — not full turn history (that would be huge).
    """
    sessions_col = get_sessions_collection()
    user_id = str(current_user.id)

    cursor = sessions_col.find(
        {
            "user_id":    user_id,
            "turns.0":    {"$exists": True},  # only sessions with turns
        },
        # Project only what the frontend needs for the list view
        projection={
            "session_id":   1,
            "created_at":   1,
            "last_updated": 1,
            "summary":      1,
            "turns":        {"$slice": -1},   # last turn only for preview
            "_id":          0,
        },
        sort=[("created_at", -1)],
    ).skip(skip).limit(limit)

    sessions = await cursor.to_list(length=limit)

    # Build clean response
    result = []
    for s in sessions:
        last_turn = s.get("turns", [{}])[-1] if s.get("turns") else {}
        result.append({
            "session_id":   s.get("session_id"),
            "created_at":   s.get("created_at"),
            "last_updated": s.get("last_updated"),
            "summary":      s.get("summary", ""),
            "preview": {
                "user_message": last_turn.get("user_message", "")[:100],
                "agent_reply":  last_turn.get("agent_reply", "")[:100],
                "timestamp":    last_turn.get("timestamp"),
            }
        })

    return {"sessions": result, "total": len(result), "skip": skip}


@router.get("/{session_id}")
async def get_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    Get full turn history for a specific session.
    Validates the session belongs to the requesting user.
    """
    sessions_col = get_sessions_collection()
    user_id = str(current_user.id)

    session = await sessions_col.find_one(
        {
            "session_id": session_id,
            "user_id":    user_id,   # security: can only read own sessions
        },
        projection={"_id": 0}
    )

    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )

    return {
        "session_id":   session.get("session_id"),
        "created_at":   session.get("created_at"),
        "last_updated": session.get("last_updated"),
        "summary":      session.get("summary", ""),
        "turn_count":   len(session.get("turns", [])),
        "turns":        session.get("turns", []),
    }


@router.get("/{session_id}/summary")
async def get_session_summary(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    Get or generate the summary for a session.
    If no summary exists yet, generates one on demand.
    """
    from app.agent.tools.summariser import summarise_session

    sessions_col = get_sessions_collection()
    user_id = str(current_user.id)

    session = await sessions_col.find_one(
        {"session_id": session_id, "user_id": user_id},
        projection={"summary": 1, "turns": 1, "_id": 0}
    )

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    summary = session.get("summary", "")

    # Generate on demand if not yet summarised
    if not summary and session.get("turns"):
        summary = await summarise_session(session_id, user_id)

    return {
        "session_id": session_id,
        "summary":    summary,
        "generated":  not bool(session.get("summary")),
    }


@router.get("/today/all")
async def get_today_sessions(
    current_user: User = Depends(get_current_user),
):
    """
    Get all sessions from today.
    Used by the journal agent to write the daily journal.
    """
    from datetime import datetime, timezone

    sessions_col = get_sessions_collection()
    user_id = str(current_user.id)

    # Today's date in UTC
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    cursor = sessions_col.find(
        {
            "user_id":   user_id,
            "created_at": {"$regex": f"^{today}"},
            "turns.0":   {"$exists": True},
        },
        projection={"_id": 0},
        sort=[("created_at", 1)],
    )

    sessions = await cursor.to_list(length=50)

    # Flatten all today's turns into one timeline
    all_turns = []
    for session in sessions:
        for turn in session.get("turns", []):
            all_turns.append({
                "session_id": session["session_id"],
                **turn
            })

    all_turns.sort(key=lambda t: t.get("timestamp", ""))

    return {
        "date":     today,
        "sessions": len(sessions),
        "turns":    all_turns,
    }