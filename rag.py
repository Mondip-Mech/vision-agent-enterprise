"""
Semantic post retrieval for Voxly.

When a brand has more example posts than the prompt budget allows, naive
truncation (``posts[:5]``) may discard the most relevant ones. This module
retrieves the top-k posts that are semantically closest to the current
generation context using dense embeddings and cosine similarity.

Model
-----
``all-MiniLM-L6-v2`` — 22 MB, runs fully offline, no API key required.
First call downloads and caches the model via sentence-transformers.

Embedding caching
-----------------
Call ``embed_posts(posts)`` after saving a brand and store the result via
``database.save_brand_embeddings()``. Pass the cached vectors as
``cached_embeddings`` to ``retrieve_relevant_posts()`` so the model is not
re-run on every caption generation call.

Install (optional dependency)
-----------------------------
    pip install -r requirements-rag.txt

If the package is not installed, all functions fall back gracefully so the
rest of the app is unaffected.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional import — graceful degradation if sentence-transformers not present
# ---------------------------------------------------------------------------
try:
    import numpy as np  # noqa: F401  (used inside retrieve_relevant_posts)
    from sentence_transformers import SentenceTransformer

    _HAS_ST: bool = True
except ImportError:  # pragma: no cover
    _HAS_ST = False

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: Any = None  # lazy-loaded singleton


def _get_model() -> Any:
    """Return the cached SentenceTransformer model, loading it on first call."""
    global _model
    if _model is None:
        logger.info("Loading sentence-transformers model '%s' (first call only)", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_posts(posts: list[str]) -> list[list[float]]:
    """
    Embed a list of brand posts and return as serialisable float vectors.

    The result can be persisted via ``database.save_brand_embeddings()`` so
    that subsequent ``retrieve_relevant_posts()`` calls skip the encode step.

    Args:
        posts: Brand example posts to embed.

    Returns:
        List of L2-normalised float vectors, one per post.

    Raises:
        ImportError: If sentence-transformers is not installed.
    """
    if not _HAS_ST:
        raise ImportError(
            "sentence-transformers is required for embedding. "
            "Run: pip install -r requirements-rag.txt"
        )
    model = _get_model()
    return model.encode(posts, normalize_embeddings=True).tolist()


def retrieve_relevant_posts(
    query: str,
    posts: list[str],
    cached_embeddings: list[list[float]] | None = None,
    top_k: int = 5,
) -> list[str]:
    """
    Return the ``top_k`` brand posts most semantically relevant to ``query``.

    Uses cosine similarity between dense embeddings. If ``cached_embeddings``
    is provided (and length matches ``posts``), the encode step for posts is
    skipped — only the query is encoded. This avoids re-running the model on
    every caption generation call.

    Falls back to ``posts[:top_k]`` when:
    - sentence-transformers is not installed
    - ``cached_embeddings`` length mismatches ``posts``
    - the model raises an unexpected error

    Args:
        query:             Retrieval context — brand name + tone + context hint.
        posts:             All available brand example posts.
        cached_embeddings: Pre-computed post embeddings from ``embed_posts()``.
        top_k:             Maximum number of posts to return.

    Returns:
        Up to ``top_k`` posts ordered by descending relevance.
    """
    if len(posts) <= top_k:
        return list(posts)

    if not _HAS_ST:
        logger.debug(
            "sentence-transformers not installed — returning first %d posts. "
            "Run `pip install -r requirements-rag.txt` to enable semantic retrieval.",
            top_k,
        )
        return posts[:top_k]

    try:
        model = _get_model()

        # Use cached embeddings when available and valid — avoids re-encoding posts
        if cached_embeddings and len(cached_embeddings) == len(posts):
            doc_embeddings = np.array(cached_embeddings)
            logger.debug("RAG: using cached embeddings for %d posts", len(posts))
        else:
            doc_embeddings = model.encode(posts, normalize_embeddings=True)
            if cached_embeddings:
                logger.warning(
                    "RAG: cached_embeddings length %d != posts length %d — re-encoding",
                    len(cached_embeddings), len(posts),
                )

        query_embedding = model.encode([query], normalize_embeddings=True)[0]
        scores          = doc_embeddings @ query_embedding   # cosine sim (normalised vectors)
        top_indices     = scores.argsort()[::-1][:top_k]
        retrieved       = [posts[int(i)] for i in top_indices]

        logger.debug(
            "RAG: retrieved %d / %d posts for query %r (cache=%s)",
            len(retrieved), len(posts), query[:60], cached_embeddings is not None,
        )
        return retrieved

    except Exception as exc:
        logger.warning("RAG retrieval failed (%s) — falling back to first %d posts", exc, top_k)
        return posts[:top_k]
