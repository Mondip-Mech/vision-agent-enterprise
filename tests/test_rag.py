"""
Tests for rag.py — semantic post retrieval.

sentence-transformers is mocked throughout so the tests run without the
optional dependency and without any model download.
"""
from __future__ import annotations

import numpy as np

import rag

# ── helpers ───────────────────────────────────────────────────────────────────

class _FakeModel:
    """
    Deterministic fake SentenceTransformer.

    Encodes each text as a unit vector derived from its index in the input list
    so we can predict which post will rank highest for a given query.
    The query is always mapped to the first basis vector [1, 0, 0, …],
    so the post whose embedding has the largest first component ranks highest.
    """

    def encode(self, texts: list[str], normalize_embeddings: bool = True):
        dim = max(len(texts), 4)
        vecs = np.zeros((len(texts), dim))
        for i, _ in enumerate(texts):
            vecs[i, i % dim] = 1.0  # each text gets a different basis vector
        return vecs


def _patch_st(monkeypatch, available: bool = True, model=None):
    monkeypatch.setattr(rag, "_HAS_ST", available)
    monkeypatch.setattr(rag, "_model", None)  # reset cached singleton
    if available:
        monkeypatch.setattr(rag, "_get_model", lambda: model or _FakeModel())


# ── retrieve_relevant_posts ───────────────────────────────────────────────────

class TestRetrieveRelevantPosts:

    def test_returns_all_when_fewer_than_top_k(self, monkeypatch):
        _patch_st(monkeypatch, available=True)
        posts = ["Post A", "Post B", "Post C"]
        result = rag.retrieve_relevant_posts("query", posts, top_k=5)
        assert result == posts

    def test_returns_all_when_exactly_top_k(self, monkeypatch):
        _patch_st(monkeypatch, available=True)
        posts = ["A", "B", "C", "D", "E"]
        result = rag.retrieve_relevant_posts("query", posts, top_k=5)
        assert len(result) == 5

    def test_fallback_when_st_not_installed(self, monkeypatch):
        _patch_st(monkeypatch, available=False)
        posts = [f"Post {i}" for i in range(10)]
        result = rag.retrieve_relevant_posts("query", posts, top_k=3)
        assert result == posts[:3]

    def test_returns_top_k_with_st(self, monkeypatch):
        _patch_st(monkeypatch, available=True)
        posts = [f"Post {i}" for i in range(10)]
        result = rag.retrieve_relevant_posts("query", posts, top_k=4)
        assert len(result) == 4

    def test_result_is_subset_of_posts(self, monkeypatch):
        _patch_st(monkeypatch, available=True)
        posts = [f"Post {i}" for i in range(8)]
        result = rag.retrieve_relevant_posts("query", posts, top_k=3)
        for p in result:
            assert p in posts

    def test_no_duplicates_in_result(self, monkeypatch):
        _patch_st(monkeypatch, available=True)
        posts = [f"Post {i}" for i in range(8)]
        result = rag.retrieve_relevant_posts("query", posts, top_k=4)
        assert len(result) == len(set(result))

    def test_fallback_on_model_error(self, monkeypatch):
        """If the model raises unexpectedly, should return posts[:top_k]."""
        class _BrokenModel:
            def encode(self, *a, **kw):
                raise RuntimeError("GPU out of memory")

        _patch_st(monkeypatch, available=True, model=_BrokenModel())
        posts = [f"Post {i}" for i in range(10)]
        result = rag.retrieve_relevant_posts("query", posts, top_k=3)
        assert result == posts[:3]

    def test_empty_posts_returns_empty(self, monkeypatch):
        _patch_st(monkeypatch, available=True)
        assert rag.retrieve_relevant_posts("query", [], top_k=5) == []

    def test_top_k_zero_returns_empty(self, monkeypatch):
        _patch_st(monkeypatch, available=True)
        posts = ["A", "B", "C"]
        result = rag.retrieve_relevant_posts("query", posts, top_k=0)
        assert result == []
