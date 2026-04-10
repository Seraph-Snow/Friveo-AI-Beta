# =============================================================================
# app/tasks/tasks.py — All Celery Tasks
# =============================================================================
# THREE SCHEDULED TASKS:
#
#   check_event_reminders  → every 15 minutes
#     Checks events collection for upcoming events
#     Creates pending_reminder documents for events due within reminder window
#
#   write_daily_journals   → daily at 21:00 UTC (9pm)
#     Reads today's sessions for all active users
#     Writes agent-perspective journal entry per user
#     Stores in journals collection
#
#   compute_weekly_analytics → every Sunday at 22:00 UTC (10pm)
#     Runs full analytics suite for all users with data
#     Stores in analytics collection
#     Infers goals if threshold met
#
# ALL TASKS:
#   - Are idempotent (safe to re-run)
#   - Never crash the worker (all exceptions caught)
#   - Log clearly so you can debug from Flower (localhost:5555)
# =============================================================================

import asyncio
import logging
from datetime import datetime, timezone, timedelta

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


def run_async(coro):
    """Run an async coroutine from a sync Celery task."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# =============================================================================
# TASK 1 — EVENT REMINDERS (every 15 minutes)
# =============================================================================

async def _check_event_reminders_async():
    """
    Find events that need a reminder sent now.

    Logic:
    - For each unreminded event in the events collection
    - Check if (event_date - reminder_lead_hours) <= now <= event_date
    - If yes, create a pending_reminder document
    - Mark the event as reminder_sent: true
    """
    from app.db.mongo import get_events_collection, get_pending_reminders_collection
    from app.db.mongo_init import init_mongo_for_task

    await init_mongo_for_task()

    events_col    = get_events_collection()
    reminders_col = get_pending_reminders_collection()

    now = datetime.now(timezone.utc)

    # Find events not yet reminded, with a future date
    cursor = events_col.find({
        "reminder_sent": {"$ne": True},
        "event_date":    {"$gte": now.isoformat()},
    })
    events = await cursor.to_list(length=1000)

    reminded = 0
    for event in events:
        try:
            event_date_str = event.get("event_date", "")
            if not event_date_str:
                continue

            event_dt = datetime.fromisoformat(
                event_date_str.replace("Z", "+00:00")
            )
            lead_hours  = event.get("reminder_lead_hours", 12)
            remind_at   = event_dt - timedelta(hours=lead_hours)

            # Is it time to remind?
            if remind_at <= now <= event_dt:
                user_id    = event.get("user_id")
                event_type = event.get("event_type", "event")
                title      = event.get("title", "something")

                # Build reminder message based on event type
                messages = {
                    "exam":        f"Hey, your {title} is coming up — how are you feeling about it?",
                    "interview":   f"Big day coming up — {title}. How's the prep going?",
                    "appointment": f"Reminder: {title} is soon. Everything okay?",
                    "social":      f"Looking forward to {title}? Hope it's a good one.",
                    "deadline":    f"Heads up — {title} deadline is approaching. How are you tracking?",
                }
                message = messages.get(event_type, f"Just a heads up — {title} is coming up soon.")

                # Create pending reminder
                await reminders_col.insert_one({
                    "user_id":    user_id,
                    "event_id":   str(event.get("_id", "")),
                    "message":    message,
                    "deliver_at": now.isoformat(),
                    "delivered":  False,
                    "event_type": event_type,
                    "event_title": title,
                })

                # Mark event as reminded
                await events_col.update_one(
                    {"_id": event["_id"]},
                    {"$set": {"reminder_sent": True, "reminded_at": now.isoformat()}}
                )

                reminded += 1
                logger.info(f"[Reminders] Queued reminder for {user_id[:8]} — {title}")

        except Exception as e:
            logger.error(f"[Reminders] Error processing event {event.get('_id')}: {e}")

    if reminded > 0:
        logger.info(f"[Reminders] Queued {reminded} reminders")
    return reminded


@celery_app.task(name="tasks.check_event_reminders", bind=True, max_retries=3)
def check_event_reminders(self):
    """Celery task: check for upcoming events and queue reminders."""
    try:
        count = run_async(_check_event_reminders_async())
        return {"reminders_queued": count}
    except Exception as e:
        logger.error(f"[Reminders] Task failed: {e}")
        raise self.retry(exc=e, countdown=60)


# =============================================================================
# TASK 2 — DAILY JOURNAL (9pm UTC daily)
# =============================================================================

JOURNAL_PROMPT = """You are reflecting on your day with {user_name}, the person you companion.
Write a brief journal entry from YOUR perspective as their {agent_role}.

Today's conversations:
{conversations}

Write in first person as the companion. Be warm, observant, specific.
Focus on: what you noticed about their mood, what seemed to help or not,
any events mentioned, how they seem to be doing overall, what you'll
carry forward into future conversations.

Keep it to 3-4 sentences. No headers. Natural, reflective tone.
Example: "Arjun came to me stressed about his exam today, but by the end
of our conversation there was more lightness in his words. The wedding
chaos at home seems to be weighing on him too — worth keeping in mind.
He mentioned a Stripe interview next week which he's clearly excited about
underneath the nerves."

Journal entry (3-4 sentences, first person):"""


async def _write_daily_journals_async():
    """Write journal entries for all users who chatted today."""
    from app.db.mongo import (
        get_sessions_collection, get_journals_collection,
        get_mood_logs_collection
    )
    from app.db.mongo_init import init_mongo_for_task
    from app.db.postgres import get_db_for_task
    from app.models.sql_models import User
    from sqlalchemy import select

    await init_mongo_for_task()

    sessions_col = get_sessions_collection()
    journals_col = get_journals_collection()
    today_str    = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Find all users who had sessions today
    cursor = sessions_col.find(
        {
            "created_at": {"$regex": f"^{today_str}"},
            "turns.0":    {"$exists": True},
        },
        projection={"user_id": 1}
    )
    today_sessions = await cursor.to_list(length=10000)
    user_ids       = list(set(s["user_id"] for s in today_sessions))

    written = 0
    for user_id in user_ids:
        try:
            # Skip if journal already written today
            existing = await journals_col.find_one({
                "user_id":      user_id,
                "journal_date": today_str,
            })
            if existing:
                continue

            # Get today's conversations
            day_cursor = sessions_col.find({
                "user_id":    user_id,
                "created_at": {"$regex": f"^{today_str}"},
                "turns.0":    {"$exists": True},
            })
            day_sessions = await day_cursor.to_list(length=100)

            # Format conversations for the journal prompt
            conv_lines = []
            for session in day_sessions:
                for turn in session.get("turns", []):
                    user_msg  = turn.get("user_message", "")
                    agent_msg = turn.get("agent_reply", "")
                    if user_msg:
                        conv_lines.append(f"User: {user_msg}")
                    if agent_msg:
                        conv_lines.append(f"You: {agent_msg}")

            if not conv_lines:
                continue

            # Get user info from Postgres
            async with get_db_for_task() as db:
                result = await db.execute(
                    select(User).where(User.id == user_id)
                )
                user = result.scalar_one_or_none()

            user_name  = (user.display_name or "the user") if user else "the user"
            agent_role = getattr(user, "agent_role", "friend") if user else "friend"

            # Call LLM
            from app.core.config import settings
            from langchain_core.messages import HumanMessage

            from langchain_ollama import ChatOllama
            llm = ChatOllama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=0.7,
            )

            prompt = JOURNAL_PROMPT.format(
                user_name=user_name,
                agent_role=agent_role,
                conversations="\n".join(conv_lines[:60]),  # cap tokens
            )

            response  = await llm.ainvoke([HumanMessage(content=prompt)])
            entry_text = response.content.strip()

            # Save journal entry
            await journals_col.insert_one({
                "user_id":      user_id,
                "journal_date": today_str,
                "entry":        entry_text,
                "created_at":   datetime.now(timezone.utc).isoformat(),
                "session_count": len(day_sessions),
                "turn_count":    len(conv_lines) // 2,
            })

            # Also write to Qdrant for retrieval
            try:
                from app.agent.tools.memory_writer import write_session_memory
                # Use journal as a synthetic memory
                for session in day_sessions:
                    if session.get("summary"):
                        await write_session_memory(
                            user_id=user_id,
                            session_id=session["session_id"],
                            summary=session["summary"],
                        )
            except Exception:
                pass  # Non-fatal

            written += 1
            logger.info(f"[Journal] Written for user {user_id[:8]} on {today_str}")

        except Exception as e:
            logger.error(f"[Journal] Failed for user {user_id[:8]}: {e}")

    return written


@celery_app.task(name="tasks.write_daily_journals", bind=True, max_retries=2)
def write_daily_journals(self):
    """Celery task: write daily journal entries for all active users."""
    try:
        count = run_async(_write_daily_journals_async())
        return {"journals_written": count}
    except Exception as e:
        logger.error(f"[Journal] Task failed: {e}")
        raise self.retry(exc=e, countdown=300)


# =============================================================================
# TASK 3 — WEEKLY ANALYTICS (Sunday 10pm UTC)
# =============================================================================

async def _compute_weekly_analytics_async():
    """Compute analytics for all users who have data."""
    from app.db.mongo import get_mood_logs_collection
    from app.db.mongo_init import init_mongo_for_task
    from app.agent.analytics_engine import compute_user_analytics, save_analytics

    await init_mongo_for_task()

    mood_col = get_mood_logs_collection()

    # Find all users with mood data
    user_ids = await mood_col.distinct("user_id", {"doc_type": "sentiment_snapshot"})

    now        = datetime.now(timezone.utc)
    week_start = (now - timedelta(days=now.weekday())).strftime("%Y-%m-%d")

    computed = 0
    for user_id in user_ids:
        try:
            analytics = await compute_user_analytics(user_id)
            await save_analytics(user_id, week_start, analytics)
            computed += 1
        except Exception as e:
            logger.error(f"[Analytics] Failed for user {user_id[:8]}: {e}")

    logger.info(f"[Analytics] Computed for {computed}/{len(user_ids)} users")
    return {"computed": computed, "total_users": len(user_ids)}


@celery_app.task(name="tasks.compute_weekly_analytics", bind=True, max_retries=2)
def compute_weekly_analytics(self):
    """Celery task: compute weekly analytics for all users."""
    try:
        result = run_async(_compute_weekly_analytics_async())
        return result
    except Exception as e:
        logger.error(f"[Analytics] Weekly task failed: {e}")
        raise self.retry(exc=e, countdown=600)