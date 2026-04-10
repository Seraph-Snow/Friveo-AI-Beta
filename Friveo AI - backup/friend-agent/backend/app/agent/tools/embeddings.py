# =============================================================================
# app/agent/tools/embeddings.py — Embedding Engine
# =============================================================================
# PRIMARY:  Gemini text-embedding-004 (768 dimensions, high quality)
# FALLBACK: nomic-embed-text via Ollama (768 dimensions, local, free)
# FALLBACK: Simple TF-IDF bag-of-words (zero dependencies, always works)
#
# WHY A FALLBACK CHAIN?
#   Gemini free tier has limits. During dev/testing you'll hit them.
#   Ollama fallback works if you have it running locally.
#   TF-IDF is the last resort — no semantic understanding but never fails.
#   The system degrades gracefully rather than crashing.
#
# WHY 768 DIMENSIONS?
#   Both Gemini text-embedding-004 and nomic-embed-text output 768-dim vectors.
#   Qdrant collection is configured for 768. Switching providers needs no
#   schema change — just a config change.
# =============================================================================

import logging
import hashlib
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 768  # both Gemini and nomic-embed-text use 768


# =============================================================================
# PRIMARY: Gemini text-embedding-004
# =============================================================================

async def embed_gemini(text: str) -> Optional[list[float]]:
    """
    Embed text using Gemini text-embedding-004.
    Returns 768-dimensional vector or None on failure.
    """
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embedder = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=settings.gemini_api_key,
        )
        vector = await embedder.aembed_query(text)
        logger.debug(f"[Embed] Gemini ✓ dim={len(vector)}")
        return vector
    except Exception as e:
        err = str(e).lower()
        if "quota" in err or "429" in err or "rate" in err:
            logger.warning("[Embed] Gemini rate limited — trying fallback")
        else:
            logger.warning(f"[Embed] Gemini failed: {e} — trying fallback")
        return None


# =============================================================================
# FALLBACK 1: Ollama nomic-embed-text (local)
# =============================================================================

async def embed_ollama(text: str) -> Optional[list[float]]:
    """
    Embed text using nomic-embed-text via Ollama.
    Returns 768-dimensional vector or None if Ollama not running.
    """
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": text},
            )
            if response.status_code == 200:
                vector = response.json()["embedding"]
                logger.debug(f"[Embed] Ollama ✓ dim={len(vector)}")
                return vector
            else:
                logger.warning(f"[Embed] Ollama returned {response.status_code}")
                return None
    except Exception as e:
        logger.warning(f"[Embed] Ollama unavailable: {e}")
        return None


# =============================================================================
# FALLBACK 2: Deterministic hash-based pseudo-embedding
# =============================================================================
# NOT semantically meaningful — just a last resort so the system doesn't crash.
# Similarity search with this will return random results, but at least
# writes won't fail and the app stays running.

def embed_hash_fallback(text: str) -> list[float]:
    """
    Deterministic pseudo-embedding from text hash.
    Zero dependencies, always works, not semantically meaningful.
    Only used when both Gemini and Ollama are unavailable.
    """
    logger.warning("[Embed] Using hash fallback — semantic search disabled")
    words = text.lower().split()
    vector = [0.0] * EMBEDDING_DIM
    for i, word in enumerate(words[:EMBEDDING_DIM]):
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vector[h % EMBEDDING_DIM] += 1.0 / (i + 1)
    # Normalize
    magnitude = sum(x**2 for x in vector) ** 0.5
    if magnitude > 0:
        vector = [x / magnitude for x in vector]
    return vector


# =============================================================================
# PUBLIC API — always returns a vector, never raises
# =============================================================================

async def embed_text(text: str, source: str = "") -> tuple[list[float], str]:
    """
    Embed text using the best available provider.
    Returns (vector, provider_name) — provider_name for logging.

    Fallback chain:
        1. Gemini text-embedding-004 (primary)
        2. Ollama nomic-embed-text (local fallback)
        3. Hash-based pseudo-embedding (last resort)

    Never raises — always returns a vector of length EMBEDDING_DIM.
    """
    label = f"[{source}] " if source else ""

    # Truncate very long text — embeddings degrade on huge inputs
    text = text.strip()
    if len(text) > 2000:
        text = text[:2000]
        logger.debug(f"{label}Text truncated to 2000 chars for embedding")

    if not text:
        logger.warning(f"{label}Empty text — returning zero vector")
        return [0.0] * EMBEDDING_DIM, "zero"

    # Try Gemini first
    if getattr(settings, "gemini_api_key", ""):
        vector = await embed_gemini(text)
        if vector and len(vector) == EMBEDDING_DIM:
            return vector, "gemini"

    # Try Ollama
    vector = await embed_ollama(text)
    if vector:
        # nomic-embed-text returns 768 dims — pad/truncate just in case
        if len(vector) > EMBEDDING_DIM:
            vector = vector[:EMBEDDING_DIM]
        elif len(vector) < EMBEDDING_DIM:
            vector = vector + [0.0] * (EMBEDDING_DIM - len(vector))
        return vector, "ollama"

    # Last resort
    return embed_hash_fallback(text), "hash_fallback"


def build_memory_text(
    summary: str,
    events: list[dict] = None,
    mood_label: str = "",
    key_topics: list[str] = None,
) -> str:
    """
    Build the text block that gets embedded for a session memory.
    Richer text = better semantic retrieval later.

    We include: summary + events + mood + topics
    This lets Qdrant find this memory when someone says "how did the exam go"
    even if the summary uses different words.
    """
    parts = [summary.strip()]

    if events:
        event_strs = []
        for e in events:
            if e.get("has_event") and e.get("title"):
                date_str = f" on {e['date']}" if e.get("date") else ""
                event_strs.append(f"{e['title']}{date_str}")
        if event_strs:
            parts.append(f"Events mentioned: {', '.join(event_strs)}")

    if mood_label:
        parts.append(f"Emotional state: {mood_label}")

    if key_topics:
        parts.append(f"Topics: {', '.join(key_topics)}")

    return "\n".join(parts)