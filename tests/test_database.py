"""
Tests for database.py

Uses an in-memory SQLite database (via monkeypatching _DB_PATH) so tests
are fully isolated and leave no files on disk.
"""
from __future__ import annotations

import os

import pytest

import database as db


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    """Redirect every test to a fresh temporary database file."""
    test_db = str(tmp_path / "test_voxly.db")
    monkeypatch.setattr(db, "_DB_PATH", test_db)
    db.init_db()
    yield
    if os.path.exists(test_db):
        os.unlink(test_db)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_profile(**overrides):
    base = {
        "tone": "friendly confident",
        "personality": "A brand that feels like your best friend.",
        "style_traits": ["casual", "punchy", "emoji-forward"],
        "emoji_usage": "moderate",
        "preferred_emojis": ["✨", "🔥"],
        "sentence_style": "short",
        "signature_words": ["vibe", "drop"],
        "cta_style": "ends with a question",
        "audience": "Gen Z fashion shoppers",
        "brand_name": "TestBrand",
        "platform": "both",
    }
    base.update(overrides)
    return base


def _make_brand(**overrides):
    kwargs = dict(
        name="TestBrand",
        platform="both",
        profile=_make_profile(),
        example_posts=["Great post 1", "Great post 2"],
    )
    kwargs.update(overrides)
    return db.save_brand(**kwargs)


# ── Settings ──────────────────────────────────────────────────────────────────

class TestSettings:
    def test_get_missing_key_returns_empty_string(self):
        assert db.get_setting("nonexistent") == ""

    def test_set_and_get(self):
        db.set_setting("buffer_token", "tok_abc123")
        assert db.get_setting("buffer_token") == "tok_abc123"

    def test_upsert_overwrites_existing_value(self):
        db.set_setting("key", "first")
        db.set_setting("key", "second")
        assert db.get_setting("key") == "second"

    def test_set_empty_string(self):
        db.set_setting("key", "")
        assert db.get_setting("key") == ""


# ── Brands ────────────────────────────────────────────────────────────────────

class TestBrands:
    def test_save_and_retrieve(self):
        brand = _make_brand()
        assert brand["id"] is not None
        assert brand["name"] == "TestBrand"
        assert brand["platform"] == "both"
        assert brand["profile"]["tone"] == "friendly confident"
        assert brand["example_posts"] == ["Great post 1", "Great post 2"]

    def test_load_brands_empty(self):
        assert db.load_brands() == []

    def test_load_brands_returns_all(self):
        _make_brand(name="BrandA")
        _make_brand(name="BrandB")
        brands = db.load_brands()
        assert len(brands) == 2
        names = {b["name"] for b in brands}
        assert names == {"BrandA", "BrandB"}

    def test_load_brands_newest_first(self):
        _make_brand(name="First")
        _make_brand(name="Second")
        brands = db.load_brands()
        assert brands[0]["name"] == "Second"

    def test_get_brand_by_id(self):
        brand = _make_brand()
        fetched = db.get_brand(brand["id"])
        assert fetched["id"] == brand["id"]

    def test_get_brand_missing_returns_none(self):
        assert db.get_brand("nonexistent-id") is None

    def test_update_brand(self):
        brand = _make_brand()
        updated = db.update_brand(
            brand["id"], "UpdatedName", "instagram", _make_profile(), ["new post"]
        )
        assert updated["name"] == "UpdatedName"
        assert updated["platform"] == "instagram"
        assert updated["example_posts"] == ["new post"]

    def test_delete_brand(self):
        brand = _make_brand()
        db.delete_brand(brand["id"])
        assert db.get_brand(brand["id"]) is None
        assert db.load_brands() == []

    def test_profile_json_round_trips(self):
        profile = _make_profile(tone="bold and edgy")
        brand = db.save_brand("RoundTrip", "instagram", profile, [])
        fetched = db.get_brand(brand["id"])
        assert fetched["profile"]["tone"] == "bold and edgy"

    def test_corrupted_profile_json_returns_empty_dict(self, monkeypatch):
        """_hydrate_brand should not crash on corrupt JSON — returns {}."""
        brand = _make_brand()
        # Directly corrupt the stored JSON
        with db._conn() as con:
            con.execute(
                "UPDATE brands SET profile_json = ? WHERE id = ?",
                ("NOT VALID JSON {{{", brand["id"]),
            )
        fetched = db.get_brand(brand["id"])
        assert fetched["profile"] == {}


# ── Captions ──────────────────────────────────────────────────────────────────

class TestCaptions:
    def setup_method(self):
        self.brand = _make_brand()

    def _save(self, **overrides):
        kwargs = dict(
            brand_id=self.brand["id"],
            brand_name=self.brand["name"],
            image_name="product.jpg",
            platform="Instagram",
            caption="Amazing new drop ✨",
            hashtags="#fashion #style",
            score=8.5,
        )
        kwargs.update(overrides)
        return db.save_caption(**kwargs)

    def test_save_and_retrieve(self):
        cap = self._save()
        assert cap["id"] is not None
        assert cap["caption"] == "Amazing new drop ✨"
        assert cap["score"] == 8.5
        assert cap["published"] == 0

    def test_load_captions_empty(self):
        assert db.load_captions() == []

    def test_load_captions_returns_all(self):
        self._save(image_name="a.jpg")
        self._save(image_name="b.jpg")
        assert len(db.load_captions()) == 2

    def test_load_captions_filter_by_brand(self):
        other_brand = _make_brand(name="OtherBrand")
        self._save(image_name="mine.jpg")
        db.save_caption(
            brand_id=other_brand["id"], brand_name="OtherBrand",
            image_name="theirs.jpg", platform="Facebook",
            caption="Other caption", hashtags="",
        )
        mine = db.load_captions(brand_id=self.brand["id"])
        assert len(mine) == 1
        assert mine[0]["image_name"] == "mine.jpg"

    def test_load_captions_filter_published(self):
        cap = self._save()
        db.update_caption(cap["id"], published=1)
        assert len(db.load_captions(published=True)) == 1
        assert len(db.load_captions(published=False)) == 0

    def test_update_caption_fields(self):
        cap = self._save()
        db.update_caption(cap["id"], caption="Edited caption", published=1)
        updated = db.get_caption(cap["id"])
        assert updated["caption"] == "Edited caption"
        assert updated["published"] == 1

    def test_update_caption_ignores_unknown_fields(self):
        cap = self._save()
        # Should not raise even though 'evil_field' is not allowed
        db.update_caption(cap["id"], evil_field="DROP TABLE captions")
        assert db.get_caption(cap["id"]) is not None

    def test_delete_caption(self):
        cap = self._save()
        db.delete_caption(cap["id"])
        assert db.get_caption(cap["id"]) is None

    def test_schedule_date_stored(self):
        cap = self._save(scheduled_date="2025-12-01T10:00:00+00:00")
        assert "2025-12-01" in db.get_caption(cap["id"])["scheduled_date"]


# ── Performance ───────────────────────────────────────────────────────────────

class TestPerformance:
    def setup_method(self):
        brand = _make_brand()
        self.cap = db.save_caption(
            brand_id=brand["id"], brand_name=brand["name"],
            image_name="x.jpg", platform="Instagram",
            caption="Cap", hashtags="#x",
        )

    def test_save_and_load(self):
        db.save_performance(self.cap["id"], likes=120, comments=5, reach=800, saves=30)
        rows = db.load_performance()
        assert len(rows) == 1
        assert rows[0]["likes"] == 120
        assert rows[0]["image_name"] == "x.jpg"

    def test_has_performance_false_before_save(self):
        assert db.has_performance(self.cap["id"]) is False

    def test_has_performance_true_after_save(self):
        db.save_performance(self.cap["id"], 10, 1, 100, 5)
        assert db.has_performance(self.cap["id"]) is True

    def test_load_performance_joins_caption_data(self):
        db.save_performance(self.cap["id"], 50, 2, 300, 10)
        row = db.load_performance()[0]
        assert "caption" in row
        assert "platform" in row
        assert "brand_name" in row
