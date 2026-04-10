# =============================================================================
# app/db/redis_client.py — Redis Async Connection
# =============================================================================
# TWO USES FOR REDIS IN THIS PROJECT:
#
# 1. CELERY BROKER (handled by Celery itself, database /0)
#    When FastAPI fires a background task, it pushes a message to Redis /0.
#    The Celery worker reads it and executes the task. We don't touch /0 here.
#
# 2. APPLICATION CACHE (this file, database /1)
#    We cache the user's built context envelope so we don't hit Postgres +
#    Mongo + Qdrant on every single message. After the first message in a
#    session, subsequent messages reuse the cached context (TTL: 10 minutes).
#
# WHY CACHE THE CONTEXT ENVELOPE?
#   Building context requires:
#   - 1 Postgres query (user profile + personality)
#   - 1-3 Mongo queries (recent sessions, mood logs)
#   - 1 Qdrant vector search (relevant memories)
#   That's 3-5 round trips before we even call the LLM.
#   Cache it for 10 minutes and those trips happen once per session, not
#   once per message.
# =============================================================================

import redis.asyncio as aioredis
from app.core.config import settings
from typing import Optional
import json


# Use database /1 — separate from Celery's /0 broker
# This keeps cache keys and task messages from ever colliding
CACHE_DB_URL = settings.redis_url.replace("/0", "/1")

_redis: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    """Return async Redis client, creating it if needed."""
    global _redis
    if _redis is None:
        _redis = await aioredis.from_url(
            CACHE_DB_URL,
            encoding="utf-8",
            decode_responses=True,   # return strings not bytes
        )
    return _redis


async def close_redis():
    """Close Redis connection on app shutdown."""
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None


# =============================================================================
# CACHE HELPERS
# =============================================================================
# These are thin wrappers that add JSON serialization and namespaced keys.
# All user cache keys are prefixed with "user:{user_id}:" so we can
# bulk-delete a user's entire cache with a single pattern delete.
# =============================================================================

async def cache_set(key: str, value: dict, ttl_seconds: int = 600):
    """
    Store a dict in Redis as JSON.
    Default TTL: 600 seconds (10 minutes).

    WHY JSON NOT PICKLE?
      JSON is human-readable — you can inspect cached values in RedisInsight.
      Pickle is faster but opaque and has security concerns with untrusted data.
    """
    r = await get_redis()
    await r.setex(key, ttl_seconds, json.dumps(value))


async def cache_get(key: str) -> Optional[dict]:
    """
    Retrieve a cached dict from Redis.
    Returns None if key doesn't exist or has expired.
    """
    r = await get_redis()
    data = await r.get(key)
    if data is None:
        return None
    return json.loads(data)


async def cache_delete(key: str):
    """Explicitly invalidate a cache entry (e.g. after personality update)."""
    r = await get_redis()
    await r.delete(key)


async def cache_delete_user(user_id: str):
    """
    Delete ALL cache entries for a user.
    Used when a user's profile changes significantly
    (e.g. personality reassessment).
    """
    r = await get_redis()
    # SCAN is non-blocking unlike KEYS — safe to use in production
    async for key in r.scan_iter(f"user:{user_id}:*"):
        await r.delete(key)