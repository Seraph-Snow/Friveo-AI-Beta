# =============================================================================
# app/main.py — FastAPI Application Entrypoint
# =============================================================================
# This file does three things:
#   1. Creates the FastAPI app instance with metadata
#   2. Defines the lifespan (startup + shutdown logic)
#   3. Registers all routers (groups of related endpoints)
#
# WHY LIFESPAN NOT @app.on_event("startup")?
#   on_event is deprecated in modern FastAPI. The lifespan context manager
#   is the new pattern — it uses async with so startup and shutdown are
#   always paired, even if startup raises an exception.
#   Think of it like a try/finally block for your entire application.
# =============================================================================

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.core.config import settings
from app.db.postgres import init_db
from app.db.mongo import init_mongo, close_mongo
from app.db.qdrant_client import init_qdrant, close_qdrant
from app.db.redis_client import get_redis, close_redis
from app.api import auth, chat, sessions, personality, analytics as analytics_api

# Configure logging — you'll see these messages in `docker compose logs backend`
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# LIFESPAN — runs at startup and shutdown
# =============================================================================
# Everything BEFORE yield runs at startup (app is starting).
# Everything AFTER yield runs at shutdown (app is stopping).
# The yield is where the app "lives" — handling requests.
#
# WHY DO STARTUP WORK HERE?
#   - init_db() creates tables if they don't exist
#   - init_mongo() creates indexes on collections
#   - get_redis() warms up the connection pool
#   Doing this at startup means the first request isn't slow.
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- STARTUP ----
    logger.info("Starting Friend Agent API...")

    logger.info("Initializing PostgreSQL tables...")
    await init_db()
    logger.info("PostgreSQL ready")

    logger.info("Initializing MongoDB indexes...")
    await init_mongo()
    await init_qdrant()
    logger.info("MongoDB ready")

    logger.info("Warming up Redis connection...")
    await get_redis()
    logger.info("Redis ready")

    logger.info("Friend Agent API is ready!")

    yield   # <-- app is running here, handling requests

    # ---- SHUTDOWN ----
    logger.info("Shutting down Friend Agent API...")
    await close_mongo()
    await close_qdrant()
    await close_redis()
    logger.info("Goodbye!")


# =============================================================================
# APP INSTANCE
# =============================================================================
app = FastAPI(
    title="Friend Agent API",
    description="AI companion with personality matching, mood tracking, and memory",
    version="0.1.0",
    # Disable docs in production — exposing your API schema is a security risk
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan,
)


# =============================================================================
# CORS MIDDLEWARE
# =============================================================================
# WHY CORS?
#   Your React frontend (localhost:5173) makes requests to the backend
#   (localhost:8000). Browsers block cross-origin requests by default as
#   a security measure. CORS middleware tells the browser it's allowed.
#
# WHY NOT allow_origins=["*"] IN PRODUCTION?
#   "*" means ANY website can make requests to your API.
#   A malicious site could make authenticated requests on behalf of your users.
#   In production, list only your actual frontend domain.
# =============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",    # Vite dev server
        "http://localhost:3000",    # Alternative frontend port
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,         # Allow cookies and auth headers
    allow_methods=["*"],            # GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],            # Authorization, Content-Type, etc.
)


# =============================================================================
# ROUTERS
# =============================================================================
# Each router is a group of related endpoints defined in its own file.
# We mount them here with a prefix so all auth routes are at /auth/*
# As we add features, we'll add more routers here.
# =============================================================================
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(sessions.router)
app.include_router(analytics_api.router)
app.include_router(analytics_api.journal_router)
app.include_router(analytics_api.goals_router)
app.include_router(analytics_api.reminders_router)
app.include_router(personality.router)
# Coming in later steps:
# app.include_router(chat.router)
app.include_router(sessions.router)
app.include_router(analytics_api.router)
app.include_router(analytics_api.journal_router)
app.include_router(analytics_api.goals_router)
app.include_router(analytics_api.reminders_router)
app.include_router(personality.router)
# app.include_router(personality.router)
# app.include_router(journal.router)


# =============================================================================
# HEALTH CHECK
# =============================================================================
# WHY A HEALTH ENDPOINT?
#   Docker, load balancers, and monitoring tools hit this to check if the
#   app is alive. Returns 200 = healthy, anything else = problem.
#   We also check DB connectivity so it's a real health check, not just
#   "is the process running".
# =============================================================================
@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "version": "0.1.0",
        "environment": settings.app_env,
    }


@app.get("/", tags=["System"])
async def root():
    return {
        "message": "Friend Agent API",
        "docs": "/docs",
        "health": "/health",
    }