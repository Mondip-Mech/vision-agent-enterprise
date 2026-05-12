"""
Tests for analytics.py
"""
from __future__ import annotations

import pytest

import analytics
import database as db


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    test_db = str(tmp_path / "test_voxly.db")
    monkeypatch.setattr(db, "_DB_PATH", test_db)
    db.init_db()
    yield


def _make_perf_row(**overrides):
    base = {
        "likes": 100, "comments": 5, "reach": 500, "saves": 20,
        "caption": "Test caption", "platform": "Instagram",
        "image_name": "img.jpg", "brand_name": "Brand",
    }
    base.update(overrides)
    return base


# ── summary_stats ─────────────────────────────────────────────────────────────

class TestSummaryStats:
    def test_empty_rows_returns_zeros(self):
        stats = analytics.summary_stats([])
        assert stats["total_posts"] == 0
        assert stats["avg_likes"] == 0.0
        assert stats["best_post"] is None

    def test_single_row(self):
        rows = [_make_perf_row(likes=50, reach=200)]
        stats = analytics.summary_stats(rows)
        assert stats["total_posts"] == 1
        assert stats["avg_likes"] == 50.0
        assert stats["avg_reach"] == 200.0
        assert stats["best_post"]["likes"] == 50

    def test_multiple_rows_averages(self):
        rows = [
            _make_perf_row(likes=100, reach=1000),
            _make_perf_row(likes=200, reach=2000),
        ]
        stats = analytics.summary_stats(rows)
        assert stats["avg_likes"] == 150.0
        assert stats["avg_reach"] == 1500.0

    def test_best_post_is_highest_likes(self):
        rows = [
            _make_perf_row(likes=10, image_name="low.jpg"),
            _make_perf_row(likes=999, image_name="high.jpg"),
            _make_perf_row(likes=50, image_name="mid.jpg"),
        ]
        stats = analytics.summary_stats(rows)
        assert stats["best_post"]["image_name"] == "high.jpg"

    def test_missing_likes_key_defaults_to_zero(self):
        rows = [{"caption": "x", "platform": "IG", "image_name": "x.jpg"}]
        stats = analytics.summary_stats(rows)
        assert stats["avg_likes"] == 0.0


# ── get_ai_insights ───────────────────────────────────────────────────────────

class TestGetAiInsights:
    def _seed_performance(self, n: int, monkeypatch):
        """Insert n performance rows via the real DB."""
        brand = db.save_brand("B", "both", {"brand_name": "B"}, [])
        for i in range(n):
            cap = db.save_caption(
                brand_id=brand["id"], brand_name="B",
                image_name=f"img{i}.jpg", platform="Instagram",
                caption=f"Caption number {i}", hashtags="#x",
                score=float(i),
            )
            db.save_performance(cap["id"], likes=i * 10, comments=i, reach=i * 100, saves=i)

    def test_returns_message_when_too_few_rows(self, monkeypatch):
        profile = {"brand_name": "TestBrand"}
        result = analytics.get_ai_insights(profile)
        assert "at least" in result.lower()

    def test_calls_api_with_enough_data(self, monkeypatch):
        self._seed_performance(5, monkeypatch)

        captured = {}
        def fake_post_text(prompt, **kwargs):
            captured["prompt"] = prompt
            return {"choices": [{"message": {"content": "1. Do this\n2. Do that\n3. Try this"}}]}

        monkeypatch.setattr(analytics, "_post_text", fake_post_text)
        monkeypatch.setattr(analytics, "_get_content",
                            lambda data: data["choices"][0]["message"]["content"])

        result = analytics.get_ai_insights({"brand_name": "TestBrand"})
        assert "1." in result
        assert "TestBrand" in captured["prompt"]

    def test_top_and_bottom_do_not_overlap_with_small_dataset(self, monkeypatch):
        """With only 3 rows, top and bottom samples should not include the same posts."""
        self._seed_performance(3, monkeypatch)

        call_count = {"n": 0}
        def fake_post_text(prompt, **kwargs):
            call_count["n"] += 1
            return {"choices": [{"message": {"content": "1. Tip one"}}]}

        monkeypatch.setattr(analytics, "_post_text", fake_post_text)
        monkeypatch.setattr(analytics, "_get_content",
                            lambda data: data["choices"][0]["message"]["content"])

        analytics.get_ai_insights({"brand_name": "X"})
        # Should not crash and should have called the API exactly once
        assert call_count["n"] == 1
