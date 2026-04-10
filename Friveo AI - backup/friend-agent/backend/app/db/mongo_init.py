# =============================================================================
# app/db/mongo_init.py — DB initialization for Celery tasks
# =============================================================================
# WHY THIS EXISTS:
#   FastAPI lifespan handles DB init for the web server.
#   Celery tasks run in a separate process with no FastAPI context.
#   This module initializes just enough DB connection for tasks to work.
# =============================================================================

import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

_initialized = False


async def init_mongo_for_task():
    """Initialize MongoDB connection for a Celery task."""
    global _initialized
    if _initialized:
        return
    from app.db.mongo import init_mongo
    await init_mongo()
    _initialized = True
    logger.debug("[TaskDB] MongoDB initialized")


@asynccontextmanager
async def get_db_for_task():
    """
    Async context manager for Postgres session in a Celery task.
    Usage:
        async with get_db_for_task() as db:
            result = await db.execute(...)
    """
    from app.db.postgres import AsyncSessionLocal
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise