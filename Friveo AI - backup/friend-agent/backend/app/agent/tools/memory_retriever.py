# =============================================================================
# app/agent/tools/memory_retriever.py — Vector Memory Retrieval (Step 6)
# =============================================================================

import logging
from app.agent.tools.embeddings import embed_text
from app.db.qdrant_client import get_qdrant_client, search_memories

logger = logging.getLogger(__name__)


def format_memories_for_prompt(memories: list[dict]) -> str:
    """
    Format retrieved memories into a block for injection into system prompt.

    Example output:
        Relevant memories from past conversations:
        • [2 weeks ago, stressed] Arjun was stressed about CS exam + sister's wedding.
          Events: CS exam (Mar 18), Sister's wedding
        • [last week, happy] Stripe interview came up, Arjun seemed excited.
    """
    if not memories:
        return ""

    lines = ["Relevant memories from past conversations:"]
    for m in memories:
        date    = m.get("date", "")
        mood    = m.get("mood_label", "")
        summary = m.get("summary", "").strip()
        events  = m.get("events", [])
        score   = m.get("score", 0)

        # Build context tag
        tag_parts = []
        if date:
            tag_parts.append(date)
        if mood:
            tag_parts.append(mood)
        tag = f"[{', '.join(tag_parts)}]" if tag_parts else ""

        lines.append(f"• {tag} {summary}")
        if events:
            lines.append(f"  Events mentioned: {', '.join(events)}")

    return "\n".join(lines)


async def retrieve_memories(
    message: str,
    user_id: str,
    session_id: str = None,
    top_k: int = 3,
    score_threshold: float = 0.65,
) -> list[dict]:
    """
    Retrieve semantically relevant past memories for this message.

    Args:
        message:         current user message (used as search query)
        user_id:         only search this user's memories
        session_id:      exclude the current session from results
        top_k:           max memories to return
        score_threshold: minimum cosine similarity

    Returns:
        list of memory dicts, empty list if Qdrant unavailable or no matches
    """
    client = get_qdrant_client()
    if client is None:
        logger.debug("[MemoryRetriever] Qdrant not available — skipping retrieval")
        return []

    try:
        # Embed the current message
        vector, provider = await embed_text(
            text=message,
            source="retrieval_query"
        )

        if provider == "hash_fallback":
            # Hash fallback doesn't produce semantic vectors — skip search
            logger.debug("[MemoryRetriever] Hash fallback — skipping semantic search")
            return []

        # Search Qdrant
        memories = await search_memories(
            user_id=user_id,
            query_vector=vector,
            top_k=top_k,
            score_threshold=score_threshold,
            exclude_session_id=session_id,
        )

        logger.info(
            f"[MemoryRetriever] Found {len(memories)} memories "
            f"(query: '{message[:50]}...', provider: {provider})"
        ) if memories else logger.debug("[MemoryRetriever] No relevant memories found")

        return memories

    except Exception as e:
        logger.error(f"[MemoryRetriever] Failed: {e}")
        return []


async def run_memory_retriever_tool(state: dict) -> dict:
    """LangGraph-compatible wrapper for the graph tool dispatcher."""
    memories = await retrieve_memories(
        message=state["message"],
        user_id=state["user_id"],
        session_id=state.get("session_id"),
    )
    return {"memories": memories}