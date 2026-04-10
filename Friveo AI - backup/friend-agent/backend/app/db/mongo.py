# =============================================================================
# app/db/mongo.py — MongoDB Async Connection
# =============================================================================
# WHY MOTOR NOT PYMONGO?
#   PyMongo is synchronous — calling it from async FastAPI code would block
#   the event loop just like synchronous SQLAlchemy would.
#   Motor is PyMongo's async wrapper — exact same API, but every operation
#   is awaitable. It's maintained by the MongoDB team, not a third party.
#
# COLLECTION DESIGN (what we'll store where):
#   sessions     — one document per conversation session
#                  { user_id, session_id, turns: [...], summary, created_at }
#   mood_logs    — one document per sentiment analysis snapshot
#                  { user_id, session_id, valence, energy, timestamp }
#   events       — extracted life events (exams, meetings, appointments)
#                  { user_id, title, date, reminder_sent, created_at }
#   journals     — end-of-day journal entries
#                  { user_id, date, highlights, user_additions, created_at }
#
# WHY NOT PUT SESSIONS IN POSTGRES?
#   A session has a turns array that grows unpredictably (5 turns or 500).
#   Each turn has nested metadata (sentiment, extracted entities, timestamps).
#   Modeling this in SQL requires 3-4 tables with joins on every read.
#   In MongoDB it's one document you fetch and write atomically.
# =============================================================================

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.core.config import settings
from typing import Optional


# Module-level client — created once, reused across all requests
# Motor manages its own connection pool internally
_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None


def get_mongo_client() -> AsyncIOMotorClient:
    """Return the Motor client, creating it if needed."""
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(
            settings.mongo_url,
            # How long to wait for a server response before raising an error
            serverSelectionTimeoutMS=5000,
        )
    return _client


def get_mongo_db() -> AsyncIOMotorDatabase:
    """Return the database handle."""
    global _db
    if _db is None:
        client = get_mongo_client()
        _db = client[settings.mongo_db]
    return _db


# =============================================================================
# COLLECTION ACCESSORS
# =============================================================================
# WHY FUNCTIONS NOT GLOBALS?
#   The DB connection isn't established at import time — only when first used.
#   Functions ensure the client is initialized before we try to access it.
#
# USAGE:
#   from app.db.mongo import get_sessions_collection
#   sessions = get_sessions_collection()
#   await sessions.insert_one({"user_id": "...", "turns": []})
# =============================================================================

def get_sessions_collection():
    """Chat sessions — one doc per session with turns array."""
    return get_mongo_db()["sessions"]


def get_mood_logs_collection():
    """Mood snapshots — written every N turns by sentiment tool."""
    return get_mongo_db()["mood_logs"]


def get_events_collection():
    """Extracted life events — exams, meetings, appointments."""
    return get_mongo_db()["events"]


def get_journals_collection():
    """Daily journal entries — agent-perspective, internal + readable."""
    return get_mongo_db()["journals"]


def get_analytics_collection():
    """Weekly analytics insights — mood trends, triggers, recovery patterns."""
    return get_mongo_db()["analytics"]


def get_goals_collection():
    """Inferred user goals + progress tracking."""
    return get_mongo_db()["goals"]


def get_pending_reminders_collection():
    """Event reminders queued for delivery on next session open."""
    return get_mongo_db()["pending_reminders"]


async def init_mongo():
    """
    Create indexes on startup.

    WHY INDEXES?
      Without indexes, every query scans ALL documents in a collection.
      With indexes, MongoDB jumps directly to matching documents.
      user_id is on every query — it must be indexed.
      Compound index on (user_id + created_at) supports
      "get this user's last 10 sessions sorted by time" in one fast lookup.
    """
    db = get_mongo_db()

    # Sessions: most queries are "get sessions for user X"
    await db["sessions"].create_index([("user_id", 1), ("created_at", -1)])
    await db["sessions"].create_index([("session_id", 1)], unique=True)

    # Mood logs: "get last 5 mood snapshots for user X"
    await db["mood_logs"].create_index([("user_id", 1), ("timestamp", -1)])

    # Events: "get upcoming events for user X"
    await db["events"].create_index([("user_id", 1), ("event_date", 1)])

    # Journals: "get journal for user X on date Y"
    await db["journals"].create_index(
        [("user_id", 1), ("journal_date", -1)], unique=True
    )

    # Analytics: weekly insights per user
    await db["analytics"].create_index([("user_id", 1), ("week_start", -1)])
    await db["analytics"].create_index(
        [("user_id", 1), ("week_start", 1)], unique=True
    )

    # Goals: one goals doc per user
    await db["goals"].create_index([("user_id", 1)], unique=True)

    # Pending reminders: find undelivered ones due soon
    await db["pending_reminders"].create_index([("user_id", 1), ("deliver_at", 1)])
    await db["pending_reminders"].create_index([("delivered", 1)])

    print("[MongoDB] All indexes created")


async def close_mongo():
    """Close the Motor client gracefully on app shutdown."""
    global _client
    if _client is not None:
        _client.close()
        _client = None