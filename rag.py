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

Install (optional dependency)
-----------------------------
    pip install sentence-transformers

If the package is not installed, ``retrieve_relevant_posts`` silently falls
back to ``posts[:top_k]`` so the rest of the app is unaffected.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional import — graceful degradation if sentence-transformers not present
# ---------------------------------------------------------------------------
try:
    import numpy as np  # noqa: F401  (used in retrieve_relevant_posts)
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


def retrieve_relevant_posts(
    query: str,
    posts: list[str],
    top_k: int = 5,
) -> list[str]:
    """
    Return the ``top_k`` brand posts most semantically relevant to ``query``.

    Uses cosine similarity between dense embeddings. Falls back to
    ``posts[:top_k]`` if sentence-transformers is not installed or if
    the model raises an unexpected error.

    Args:
        query:  Retrieval context — typically brand name + tone + context hint.
        posts:  All available brand example posts.
        top_k:  Maximum number of posts to return.

    Returns:
        Up to ``top_k`` posts ordered by descending relevance.
    """
    if len(posts) <= top_k:
        return list(posts)

    if not _HAS_ST:
        logger.debug(
            "sentence-transformers not installed — returning first %d posts. "
            "Run `pip install sentence-transformers` to enable semantic retrieval.",
            top_k,
        )
        return posts[:top_k]

    try:
        model = _get_model()
        doc_embeddings   = model.encode(posts, normalize_embeddings=True)
        query_embedding  = model.encode([query], normalize_embeddings=True)[0]
        scores           = doc_embeddings @ query_embedding          # cosine sim (normalised)
        top_indices      = scores.argsort()[::-1][:top_k]
        retrieved        = [posts[int(i)] for i in top_indices]
        logger.debug(
            "RAG: retrieved %d / %d posts for query %r",
            len(retrieved), len(posts), query[:60],
        )
        return retrieved

    except Exception as exc:
        logger.warning(
            "RAG retrieval failed (%s) — falling back to first %d posts", exc, top_k,
        )
        return posts[:top_k]
