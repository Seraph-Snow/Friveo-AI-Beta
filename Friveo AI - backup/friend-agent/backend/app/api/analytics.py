# =============================================================================
# app/api/analytics.py — Analytics, Journal, Goals, Reminders endpoints
# =============================================================================
# ENDPOINTS:
#   GET  /analytics/me              → full computed analytics for current user
#   GET  /analytics/me/summary      → concise summary (for agent context)
#   POST /analytics/me/compute      → trigger manual recompute
#
#   GET  /journal                   → list journal entries ("what I noticed")
#   GET  /journal/{date}            → single entry by date
#
#   GET  /goals                     → inferred goals + progress
#
#   GET  /reminders/pending         → undelivered reminders for this user
#   POST /reminders/{id}/delivered  → mark reminder as delivered (frontend calls this)
# =============================================================================

import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres import get_db
from app.api.auth import get_current_user
from app.models.sql_models import User
from app.db.mongo import (
    get_analytics_collection,
    get_journals_collection,
    get_pending_reminders_collection,
)
from app.agent.analytics_engine import (
    compute_user_analytics,
    save_analytics,
    get_latest_analytics,
    build_analytics_context_for_prompt,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analytics", tags=["Analytics"])


# =============================================================================
# ANALYTICS
# =============================================================================

@router.get("/me")
async def get_my_analytics(
    days:         int  = Query(default=90, le=365),
    current_user: User = Depends(get_current_user),
):
    """
    Get the latest computed analytics for the current user.
    Returns pre-computed results if available, otherwise triggers computation.
    """
    analytics = await get_latest_analytics(str(current_user.id))

    # If no analytics yet or stale (>7 days), compute now
    recompute = False
    if not analytics:
        recompute = True
    elif analytics.get("computed_at"):
        try:
            computed_dt = datetime.fromisoformat(analytics["computed_at"])
            age_days    = (datetime.now(timezone.utc) - computed_dt).days
            if age_days >= 7:
                recompute = True
        except Exception:
            recompute = True

    if recompute:
        analytics = await compute_user_analytics(str(current_user.id), days)
        week_start = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        await save_analytics(str(current_user.id), week_start, analytics)

    # Remove internal fields before returning
    if analytics:
        analytics.pop("_id", None)

    return analytics


@router.get("/me/summary")
async def get_analytics_summary(
    current_user: User = Depends(get_current_user),
):
    """Concise analytics summary — same format injected into agent context."""
    summary = await build_analytics_context_for_prompt(str(current_user.id))
    return {"summary": summary, "has_data": bool(summary)}


@router.post("/me/compute")
async def trigger_compute(
    current_user: User = Depends(get_current_user),
):
    """Manually trigger analytics recomputation."""
    analytics  = await compute_user_analytics(str(current_user.id))
    week_start = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    await save_analytics(str(current_user.id), week_start, analytics)
    return {
        "message":        "Analytics recomputed",
        "snapshot_count": analytics.get("snapshot_count", 0),
        "has_data":       analytics.get("has_data", False),
    }


# =============================================================================
# JOURNAL — "What I've noticed about you"
# =============================================================================

journal_router = APIRouter(prefix="/journal", tags=["Journal"])


@journal_router.get("")
async def list_journal_entries(
    limit:        int  = Query(default=30, le=100),
    skip:         int  = Query(default=0),
    current_user: User = Depends(get_current_user),
):
    """
    List journal entries — most recent first.
    This is the "What I've noticed about you" view.
    """
    journals_col = get_journals_collection()
    cursor = journals_col.find(
        {"user_id": str(current_user.id)},
        projection={"_id": 0},
        sort=[("journal_date", -1)],
    ).skip(skip).limit(limit)

    entries = await cursor.to_list(length=limit)
    return {
        "entries": entries,
        "total":   len(entries),
    }


@journal_router.get("/{date}")
async def get_journal_entry(
    date:         str,
    current_user: User = Depends(get_current_user),
):
    """Get journal entry for a specific date (YYYY-MM-DD)."""
    journals_col = get_journals_collection()
    entry = await journals_col.find_one(
        {"user_id": str(current_user.id), "journal_date": date},
        projection={"_id": 0},
    )
    if not entry:
        raise HTTPException(status_code=404, detail="No journal entry for this date")
    return entry


# =============================================================================
# GOALS
# =============================================================================

goals_router = APIRouter(prefix="/goals", tags=["Goals"])


@goals_router.get("")
async def get_goals(
    current_user: User = Depends(get_current_user),
):
    """
    Get inferred goals and progress for the current user.
    Returns goals if threshold met (30 days or 50 snapshots),
    otherwise returns eligibility status.
    """
    analytics = await get_latest_analytics(str(current_user.id))
    if not analytics:
        return {
            "eligible": False,
            "goals":    [],
            "message":  "Start chatting to unlock your wellbeing insights.",
        }
    return analytics.get("goals", {
        "eligible": False,
        "goals":    [],
        "message":  "Analytics not yet computed. Try again shortly.",
    })


# =============================================================================
# REMINDERS
# =============================================================================

reminders_router = APIRouter(prefix="/reminders", tags=["Reminders"])


@reminders_router.get("/pending")
async def get_pending_reminders(
    current_user: User = Depends(get_current_user),
):
    """
    Get undelivered reminders for the current user.
    Called by the frontend when a new session starts.
    Returns the reminder messages to show as agent opening messages.
    """
    reminders_col = get_pending_reminders_collection()
    cursor = reminders_col.find(
        {
            "user_id":   str(current_user.id),
            "delivered": False,
        },
        projection={"_id": 1, "message": 1, "event_type": 1, "event_title": 1},
        sort=[("deliver_at", 1)],
    )
    reminders = await cursor.to_list(length=10)

    # Convert ObjectId to string
    for r in reminders:
        r["id"] = str(r.pop("_id"))

    return {"reminders": reminders}


@reminders_router.post("/{reminder_id}/delivered")
async def mark_reminder_delivered(
    reminder_id:  str,
    current_user: User = Depends(get_current_user),
):
    """Mark a reminder as delivered after it's been shown to the user."""
    from bson import ObjectId
    reminders_col = get_pending_reminders_collection()

    result = await reminders_col.update_one(
        {
            "_id":      ObjectId(reminder_id),
            "user_id":  str(current_user.id),
        },
        {"$set": {
            "delivered":    True,
            "delivered_at": datetime.now(timezone.utc).isoformat(),
        }}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Reminder not found")

    return {"delivered": True}
    