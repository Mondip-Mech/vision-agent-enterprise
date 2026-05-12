"""
Tests for summarizer.py

All NVIDIA API calls are mocked — no real HTTP requests are made.
"""
from __future__ import annotations

import json

import pytest

import summarizer

# ── Fixtures & helpers ────────────────────────────────────────────────────────

def _api_response(content: str) -> dict:
    """Build a minimal NVIDIA-style chat completion response."""
    return {
        "choices": [
            {"message": {"content": content, "role": "assistant"}}
        ]
    }


def _profile(**overrides) -> dict:
    base = {
        "brand_name": "TestBrand",
        "platform": "both",
        "tone": "fun casual",
        "personality": "The cool friend who always knows what's trending.",
        "style_traits": ["punchy", "emoji-forward", "conversational"],
        "emoji_usage": "moderate",
        "preferred_emojis": ["✨", "🔥"],
        "sentence_style": "short",
        "signature_words": ["vibe", "drop"],
        "cta_style": "ends with a question",
        "audience": "Gen Z shoppers",
    }
    base.update(overrides)
    return base


# ── _parse_json ───────────────────────────────────────────────────────────────

class TestParseJson:
    def test_parses_clean_json(self):
        obj = summarizer._parse_json('{"tone": "bold"}')
        assert obj == {"tone": "bold"}

    def test_parses_json_in_markdown_block(self):
        text = '```json\n{"tone": "bold"}\n```'
        assert summarizer._parse_json(text) == {"tone": "bold"}

    def test_parses_json_embedded_in_prose(self):
        text = 'Here is the result:\n{"tone": "bold"}\nEnd.'
        assert summarizer._parse_json(text) == {"tone": "bold"}

    def test_returns_none_for_plain_text(self):
        assert summarizer._parse_json("This is just plain text.") is None

    def test_returns_none_for_empty_string(self):
        assert summarizer._parse_json("") is None

    def test_returns_none_for_array_json(self):
        # We only want dicts, not lists
        assert summarizer._parse_json("[1, 2, 3]") is None


# ── _parse_variations ─────────────────────────────────────────────────────────

class TestParseVariations:
    def _make_data(self, variations: list) -> dict:
        return {"image_analysis": "A red jacket.", "variations": variations}

    def test_parses_full_variations(self):
        data = self._make_data([
            {
                "facebook_caption": "FB cap",
                "instagram_caption": "IG cap",
                "hashtags": ["#fashion", "#style"],
                "score": 9,
                "score_reason": "Great tone",
            }
        ])
        result = summarizer._parse_variations(data, "both")
        assert len(result) == 1
        assert result[0]["facebook_caption"] == "FB cap"
        assert result[0]["score"] == 9.0
        assert "#fashion" in result[0]["hashtags"]

    def test_adds_hash_prefix_to_hashtags(self):
        data = self._make_data([
            {"hashtags": ["fashion", "#style"], "score": 7}
        ])
        result = summarizer._parse_variations(data, "both")
        assert "#fashion" in result[0]["hashtags"]
        assert "#style" in result[0]["hashtags"]

    def test_handles_string_hashtags(self):
        data = self._make_data([{"hashtags": "#a #b #c", "score": 7}])
        result = summarizer._parse_variations(data, "both")
        assert len(result[0]["hashtags"]) == 3

    def test_defaults_score_to_7_on_missing(self):
        data = self._make_data([{"hashtags": []}])
        result = summarizer._parse_variations(data, "both")
        assert result[0]["score"] == 7.0

    def test_defaults_score_to_7_on_invalid(self):
        data = self._make_data([{"hashtags": [], "score": "great"}])
        result = summarizer._parse_variations(data, "both")
        assert result[0]["score"] == 7.0

    def test_returns_empty_list_when_no_variations_key(self):
        assert summarizer._parse_variations({}, "both") == []

    def test_skips_non_dict_entries(self):
        data = self._make_data(["not a dict", {"hashtags": [], "score": 8}])
        result = summarizer._parse_variations(data, "both")
        assert len(result) == 1


# ── _get_content ──────────────────────────────────────────────────────────────

class TestGetContent:
    def test_extracts_string_content(self):
        data = _api_response("Hello world")
        assert summarizer._get_content(data) == "Hello world"

    def test_extracts_list_content(self):
        data = {"choices": [{"message": {"content": [
            {"type": "text", "text": "Part one "},
            {"type": "text", "text": "part two"},
        ]}}]}
        assert summarizer._get_content(data) == "Part one  part two"

    def test_raises_on_empty_choices(self):
        with pytest.raises(ValueError, match="no choices"):
            summarizer._get_content({"choices": []})

    def test_raises_on_empty_content(self):
        with pytest.raises(ValueError, match="empty content"):
            summarizer._get_content(_api_response("   "))

    def test_strips_whitespace(self):
        assert summarizer._get_content(_api_response("  hello  ")) == "hello"


# ── _compress_image ───────────────────────────────────────────────────────────

class TestCompressImage:
    def _make_image(self, size=(200, 200)):
        from PIL import Image
        img = Image.new("RGB", size, color=(200, 100, 50))
        return img

    def test_returns_jpeg_mime(self):
        _, mime = summarizer._compress_image(self._make_image())
        assert mime == "image/jpeg"

    def test_output_is_valid_jpeg(self):
        from io import BytesIO

        from PIL import Image
        data, _ = summarizer._compress_image(self._make_image())
        img = Image.open(BytesIO(data))
        assert img.format == "JPEG"

    def test_large_image_is_reduced(self):
        # A 2000×2000 image should be thumbnailed and compressed
        data, _ = summarizer._compress_image(self._make_image((2000, 2000)))
        assert len(data) <= 1024 * 1024  # well under 1 MB


# ── _open_image ───────────────────────────────────────────────────────────────

class TestOpenImage:
    def _jpeg_bytes(self, size=(100, 100)):
        from io import BytesIO

        from PIL import Image
        buf = BytesIO()
        Image.new("RGB", size, color=(10, 20, 30)).save(buf, format="JPEG")
        return buf.getvalue()

    def test_opens_valid_jpeg(self):
        img = summarizer._open_image(self._jpeg_bytes())
        assert img.mode == "RGB"

    def test_raises_on_corrupt_bytes(self):
        with pytest.raises(ValueError, match="could not be decoded"):
            summarizer._open_image(b"not an image")


# ── analyze_brand_voice ───────────────────────────────────────────────────────

class TestAnalyzeBrandVoice:
    VALID_PROFILE = {
        "tone": "fun casual",
        "personality": "Like your best friend.",
        "style_traits": ["punchy"],
        "emoji_usage": "moderate",
        "preferred_emojis": ["✨"],
        "sentence_style": "short",
        "signature_words": ["drop"],
        "cta_style": "ends with question",
        "audience": "Gen Z",
    }

    def test_returns_profile_with_brand_metadata(self, monkeypatch):
        monkeypatch.setattr(summarizer, "_post_text",
                            lambda *a, **kw: _api_response(json.dumps(self.VALID_PROFILE)))
        monkeypatch.setattr(summarizer, "_require_api_key", lambda: None)

        result = summarizer.analyze_brand_voice("Nike", "instagram", ["Post 1", "Post 2"])
        assert result["tone"] == "fun casual"
        assert result["brand_name"] == "Nike"
        assert result["platform"] == "instagram"

    def test_raises_when_api_key_missing(self, monkeypatch):
        monkeypatch.setattr(summarizer, "_require_api_key",
                            lambda: (_ for _ in ()).throw(ValueError("NVIDIA_API_KEY is not set")))
        with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
            summarizer.analyze_brand_voice("Brand", "both", ["p1", "p2"])

    def test_raises_when_profile_unparseable(self, monkeypatch):
        monkeypatch.setattr(summarizer, "_post_text",
                            lambda *a, **kw: _api_response("sorry, I cannot do that"))
        monkeypatch.setattr(summarizer, "_require_api_key", lambda: None)
        with pytest.raises(ValueError, match="voice profile"):
            summarizer.analyze_brand_voice("Brand", "both", ["p1", "p2"])


# ── generate_caption_variations ──────────────────────────────────────────────

class TestGenerateCaptionVariations:
    def _jpeg_bytes(self):
        from io import BytesIO

        from PIL import Image
        buf = BytesIO()
        Image.new("RGB", (100, 100)).save(buf, format="JPEG")
        return buf.getvalue()

    def _valid_response(self):
        payload = {
            "image_analysis": "A red jacket on a white background.",
            "variations": [
                {
                    "facebook_caption": "New drop just landed 🔥",
                    "instagram_caption": "Your next fave. ✨",
                    "hashtags": ["#fashion", "#ootd"],
                    "score": 9,
                    "score_reason": "Strong voice match",
                },
                {
                    "facebook_caption": "Style meets comfort.",
                    "instagram_caption": "Elevate your look 🚀",
                    "hashtags": ["#style", "#newdrop"],
                    "score": 7,
                    "score_reason": "Good but generic opener",
                },
            ],
        }
        return _api_response(json.dumps(payload))

    def test_returns_variations_sorted_by_score(self, monkeypatch):
        monkeypatch.setattr(summarizer, "_require_api_key", lambda: None)
        monkeypatch.setattr(summarizer, "_post_vision_with_retry",
                            lambda *a, **kw: self._valid_response())

        result = summarizer.generate_caption_variations(
            self._jpeg_bytes(), _profile(), ["Post 1", "Post 2"]
        )
        assert result["error"] is None
        assert len(result["variations"]) == 2
        assert result["variations"][0]["score"] >= result["variations"][1]["score"]

    def test_includes_image_analysis(self, monkeypatch):
        monkeypatch.setattr(summarizer, "_require_api_key", lambda: None)
        monkeypatch.setattr(summarizer, "_post_vision_with_retry",
                            lambda *a, **kw: self._valid_response())

        result = summarizer.generate_caption_variations(
            self._jpeg_bytes(), _profile(), ["Post 1"]
        )
        assert "red jacket" in result["image_analysis"]

    def test_returns_error_dict_on_api_failure(self, monkeypatch):
        import requests
        monkeypatch.setattr(summarizer, "_require_api_key", lambda: None)
        monkeypatch.setattr(summarizer, "_post_vision_with_retry",
                            lambda *a, **kw: (_ for _ in ()).throw(
                                requests.exceptions.Timeout("timed out")))

        result = summarizer.generate_caption_variations(
            self._jpeg_bytes(), _profile(), ["Post 1"]
        )
        assert result["error"] is not None
        assert result["variations"] == []

    def test_raises_when_api_key_missing(self, monkeypatch):
        monkeypatch.setattr(summarizer, "_require_api_key",
                            lambda: (_ for _ in ()).throw(ValueError("NVIDIA_API_KEY is not set")))
        with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
            summarizer.generate_caption_variations(self._jpeg_bytes(), _profile(), [])


# ── _build_variations_prompt ─────────────────────────────────────────────────

class TestBuildVariationsPrompt:
    def test_includes_brand_name(self):
        prompt = summarizer._build_variations_prompt(
            _profile(brand_name="Zara"), ["Post 1"], "both", "", 3, "medium"
        )
        assert "Zara" in prompt

    def test_includes_context_when_provided(self):
        prompt = summarizer._build_variations_prompt(
            _profile(), ["Post 1"], "both", "Winter jacket £89", 3, "medium"
        )
        assert "Winter jacket £89" in prompt

    def test_excludes_context_block_when_empty(self):
        prompt = summarizer._build_variations_prompt(
            _profile(), ["Post 1"], "both", "", 3, "medium"
        )
        assert "SPECIFIC CONTEXT" not in prompt

    def test_facebook_only_schema(self):
        prompt = summarizer._build_variations_prompt(
            _profile(platform="facebook"), ["Post 1"], "facebook", "", 1, "short"
        )
        assert '"instagram_caption": null' in prompt

    def test_instagram_only_schema(self):
        prompt = summarizer._build_variations_prompt(
            _profile(platform="instagram"), ["Post 1"], "instagram", "", 1, "short"
        )
        assert '"facebook_caption": null' in prompt

    def test_length_note_injected(self):
        prompt = summarizer._build_variations_prompt(
            _profile(), ["Post 1"], "both", "", 3, "long"
        )
        assert "80" in prompt  # from the "long" length note

    def test_caps_example_posts_at_max(self):
        posts = [f"Post {i}" for i in range(20)]
        prompt = summarizer._build_variations_prompt(
            _profile(), posts, "both", "", 3, "medium"
        )
        # Only _MAX_EXAMPLE_POSTS (5) posts should appear
        assert "Post 5" not in prompt
