# =============================================================================
# app/db/postgres.py — PostgreSQL Async Connection
# =============================================================================
# WHY ASYNC SQLALCHEMY?
#   FastAPI is fully async. If your DB calls are synchronous (blocking),
#   the entire server freezes while waiting for Postgres to respond —
#   no other requests can be handled during that wait.
#
#   With async SQLAlchemy + asyncpg driver:
#   - FastAPI "suspends" the request coroutine while waiting for Postgres
#   - Meanwhile it handles other incoming requests
#   - When Postgres responds, the original request resumes
#   This is how you get high concurrency without multiple threads.
#
# THE CONNECTION POOL:
#   We don't create a new DB connection on every request — that's slow.
#   SQLAlchemy maintains a POOL of open connections (default: 5-10).
#   Each request borrows one from the pool and returns it when done.
#   pool_size=10 means 10 simultaneous DB operations max.
#   max_overflow=20 means up to 20 extra connections during traffic spikes.
# =============================================================================

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings


# =============================================================================
# ENGINE — the actual connection to Postgres
# =============================================================================
# echo=True in development prints every SQL query to the console.
# This is invaluable for learning — you see exactly what SQL your ORM
# code generates. Set echo=False in production (too noisy).
# =============================================================================
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,      # log SQL queries when DEBUG=true
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,       # test connection health before using from pool
)

# =============================================================================
# SESSION FACTORY — creates individual database sessions
# =============================================================================
# A "session" is a unit of work with the database.
# expire_on_commit=False means objects don't expire after commit —
# you can still access their attributes after saving them.
# This matters in async code where re-fetching after commit is expensive.
# =============================================================================
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# =============================================================================
# BASE CLASS — all SQLAlchemy models inherit from this
# =============================================================================
# DeclarativeBase is the modern SQLAlchemy 2.0 way to define models.
# Every table class in sql_models.py will inherit from Base.
# =============================================================================
class Base(DeclarativeBase):
    pass


# =============================================================================
# DEPENDENCY — used by FastAPI to inject DB sessions into endpoints
# =============================================================================
# WHY A DEPENDENCY FUNCTION?
#   FastAPI's dependency injection system calls this function for every
#   request that needs a DB session. The `yield` makes it a context manager:
#   - Code before yield: create and provide the session
#   - Code after yield: always runs (even on errors) to close the session
#
# USAGE IN AN ENDPOINT:
#   async def my_endpoint(db: AsyncSession = Depends(get_db)):
#       result = await db.execute(select(User))
#
# The session is automatically closed after the request, even if an
# exception occurs — no connection leaks.
# =============================================================================
async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# =============================================================================
# INIT — create all tables on startup
# =============================================================================
# WHY NOT JUST USE ALEMBIC?
#   Alembic handles migrations (changes to existing tables).
#   create_all handles initial table creation if they don't exist.
#   In development, we use create_all for simplicity.
#   In production, you'd use Alembic exclusively.
# =============================================================================
async def init_db():
    """Create all tables. Called once at app startup."""
    async with engine.begin() as conn:
        # Import models here to ensure they're registered with Base
        from app.models import sql_models  # noqa: F401
        await conn.run_sync(Base.metadata.create_all)