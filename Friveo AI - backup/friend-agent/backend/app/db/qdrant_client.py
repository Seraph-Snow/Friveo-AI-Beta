# =============================================================================
# app/db/qdrant_client.py — Qdrant Vector Store
# =============================================================================
# COLLECTION DESIGN:
#   One collection: "memories"
#   Filtered by user_id payload field — all users share one collection
#   Each point = one session summary
#
# WHY ONE COLLECTION NOT PER-USER COLLECTIONS?
#   Per-user collections = thousands of tiny collections = operational nightmare
#   One collection with user_id filter = clean, scalable, standard pattern
#   Qdrant handles filtered search efficiently with payload indexes
#
# POINT STRUCTURE:
#   id:      UUID (deterministic from session_id so we can upsert)
#   vector:  768-dim float list (Gemini or nomic-embed-text)
#   payload: {
#     user_id:    str,
#     session_id: str,
#     summary:    str,       full summary text (returned with results)
#     date:       str,       ISO date of session
#     mood_label: str,       e.g. "stressed and anxious"
#     valence:    float,     -1.0 to 1.0
#     events:     list[str], event titles from this session
#     provider:   str,       which embedding model was used
#   }
# =============================================================================

import uuid
import logging
from typing import Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    PayloadSchemaType,
)

from app.core.config import settings
from app.agent.tools.embeddings import EMBEDDING_DIM

logger = logging.getLogger(__name__)

COLLECTION_NAME = "memories"

# Module-level client — initialized once on startup
_qdrant_client: Optional[AsyncQdrantClient] = None


def get_qdrant_client() -> AsyncQdrantClient:
    """Get the Qdrant client. Must call init_qdrant() first."""
    if _qdrant_client is None:
        raise RuntimeError("Qdrant not initialized — call init_qdrant() first")
    return _qdrant_client


async def init_qdrant() -> None:
    """
    Initialize Qdrant client and ensure the memories collection exists.
    Called once at application startup.
    """
    global _qdrant_client

    try:
        qdrant_url = getattr(settings, "qdrant_url", "http://qdrant:6333")
        _qdrant_client = AsyncQdrantClient(url=qdrant_url)

        # Check if collection exists
        collections = await _qdrant_client.get_collections()
        existing = [c.name for c in collections.collections]

        if COLLECTION_NAME not in existing:
            await _qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"[Qdrant] Created collection '{COLLECTION_NAME}' ({EMBEDDING_DIM}d cosine)")

            # Create payload index on user_id for fast filtered search
            await _qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="user_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info("[Qdrant] Created payload index on user_id")
        else:
            logger.info(f"[Qdrant] Collection '{COLLECTION_NAME}' already exists")

    except Exception as e:
        logger.error(f"[Qdrant] Initialization failed: {e}")
        logger.warning("[Qdrant] Continuing without vector memory — Step 6 features disabled")
        _qdrant_client = None


async def close_qdrant() -> None:
    """Close Qdrant connection on app shutdown."""
    global _qdrant_client
    if _qdrant_client:
        await _qdrant_client.close()
        _qdrant_client = None
        logger.info("[Qdrant] Connection closed")


def session_to_point_id(session_id: str) -> str:
    """
    Convert session_id to a deterministic UUID for Qdrant.
    Deterministic = same session_id always maps to same point_id.
    This enables upsert (update if exists, insert if not).
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"memory:{session_id}"))


async def upsert_memory(
    user_id: str,
    session_id: str,
    vector: list[float],
    summary: str,
    date: str,
    mood_label: str = "",
    valence: float = 0.0,
    events: list[str] = None,
    provider: str = "gemini",
) -> bool:
    """
    Upsert a session memory into Qdrant.
    Uses deterministic point_id so re-running is idempotent.
    """
    client = get_qdrant_client()
    if client is None:
        return False

    try:
        point_id = session_to_point_id(session_id)
        point = PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "user_id":    user_id,
                "session_id": session_id,
                "summary":    summary,
                "date":       date,
                "mood_label": mood_label,
                "valence":    valence,
                "events":     events or [],
                "provider":   provider,
            }
        )
        await client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point],
        )
        logger.info(f"[Qdrant] Memory upserted for session {session_id} (provider: {provider})")
        return True

    except Exception as e:
        logger.error(f"[Qdrant] Upsert failed for session {session_id}: {e}")
        return False


async def search_memories(
    user_id: str,
    query_vector: list[float],
    top_k: int = 3,
    score_threshold: float = 0.65,
    exclude_session_id: str = None,
) -> list[dict]:
    """
    Search for relevant memories for a user.

    Args:
        user_id:            only search this user's memories
        query_vector:       embedded query message
        top_k:              max results to return
        score_threshold:    minimum cosine similarity (0.65 = good match)
        exclude_session_id: exclude the current session from results

    Returns:
        list of { summary, date, mood_label, valence, events, score }
    """
    client = get_qdrant_client()
    if client is None:
        return []

    try:
        # Filter to this user's memories only
        user_filter = Filter(
            must=[FieldCondition(
                key="user_id",
                match=MatchValue(value=user_id)
            )]
        )

        results = await client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=user_filter,
            limit=top_k + 1,  # fetch one extra in case we exclude current session
            score_threshold=score_threshold,
            with_payload=True,
        )

        memories = []
        for r in results:
            payload = r.payload or {}
            # Exclude current session if specified
            if exclude_session_id and payload.get("session_id") == exclude_session_id:
                continue
            memories.append({
                "summary":    payload.get("summary", ""),
                "date":       payload.get("date", ""),
                "mood_label": payload.get("mood_label", ""),
                "valence":    payload.get("valence", 0.0),
                "events":     payload.get("events", []),
                "score":      round(r.score, 3),
                "session_id": payload.get("session_id", ""),
            })
            if len(memories) >= top_k:
                break

        if memories:
            logger.info(
                f"[Qdrant] Found {len(memories)} memories for user {user_id[:8]}... "
                f"(scores: {[m['score'] for m in memories]})"
            )
        else:
            logger.debug(f"[Qdrant] No memories above threshold {score_threshold}")

        return memories

    except Exception as e:
        logger.error(f"[Qdrant] Search failed: {e}")
        return []