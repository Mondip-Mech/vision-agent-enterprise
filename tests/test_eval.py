"""
Tests for eval_captions.py — independent caption judge framework.

sentence-transformers and real API calls are mocked throughout.
"""
from __future__ import annotations

import pytest

import eval_captions
import summarizer
from config import Settings

# ── helpers ───────────────────────────────────────────────────────────────────

def _openai_response(text: str) -> dict:
    return {"choices": [{"message": {"content": text, "role": "assistant"}}]}


def _fake_score_response(score: float, reasoning: str) -> dict:
    import json
    payload = json.dumps({"score": score, "reasoning": reasoning})
    return _openai_response(payload)


def _brand(name: str = "TestBrand") -> dict:
    return {
        "brand_name": name,
        "tone": "warm, friendly",
        "personality": "approachable",
        "style_traits": ["conversational", "emoji-friendly"],
        "emoji_usage": "moderate",
        "audience": "young adults",
        "cta_style": "soft ask",
    }


# ── judge_provider ────────────────────────────────────────────────────────────

class TestJudgeProvider:
    def test_returns_different_provider_when_available(self, monkeypatch):
        """With nvidia as generator and anthropic key set, judge should be anthropic."""
        fake = Settings(
            caption_provider="nvidia",
            nvidia_api_key="nvapi-x",
            anthropic_api_key="sk-ant-x",
        )
        monkeypatch.setattr(eval_captions, "settings", fake)
        monkeypatch.setattr(summarizer, "settings", fake)
        assert eval_captions.judge_provider() != "nvidia"

    def test_falls_back_to_generator_when_single_key(self, monkeypatch):
        """With only nvidia key, judge falls back to nvidia (self-evaluation)."""
        fake = Settings(caption_provider="nvidia", nvidia_api_key="nvapi-x")
        monkeypatch.setattr(eval_captions, "settings", fake)
        monkeypatch.setattr(summarizer, "settings", fake)
        assert eval_captions.judge_provider() == "nvidia"

    def test_openai_generator_picks_anthropic_judge(self, monkeypatch):
        fake = Settings(
            caption_provider="openai",
            openai_api_key="sk-x",
            anthropic_api_key="sk-ant-x",
        )
        monkeypatch.setattr(eval_captions, "settings", fake)
        monkeypatch.setattr(summarizer, "settings", fake)
        judge = eval_captions.judge_provider()
        assert judge == "anthropic"

    def test_anthropic_generator_picks_openai_judge(self, monkeypatch):
        fake = Settings(
            caption_provider="anthropic",
            anthropic_api_key="sk-ant-x",
            openai_api_key="sk-x",
        )
        monkeypatch.setattr(eval_captions, "settings", fake)
        monkeypatch.setattr(summarizer, "settings", fake)
        judge = eval_captions.judge_provider()
        assert judge == "openai"


# ── score_caption ─────────────────────────────────────────────────────────────

class TestScoreCaption:
    def _patch_judge(self, monkeypatch, score: float, reasoning: str):
        """Patch _call_for_provider so it returns a fixed evaluation."""
        monkeypatch.setattr(
            summarizer,
            "_call_for_provider",
            lambda p, m, **kw: _fake_score_response(score, reasoning),
        )
        fake = Settings(caption_provider="nvidia", nvidia_api_key="nvapi-x")
        monkeypatch.setattr(eval_captions, "settings", fake)
        monkeypatch.setattr(summarizer, "settings", fake)

    def test_returns_score_and_reasoning(self, monkeypatch):
        self._patch_judge(monkeypatch, 8.5, "Great brand voice match.")
        result = eval_captions.score_caption("Great caption!", _brand())
        assert result["score"] == 8.5
        assert result["reasoning"] == "Great brand voice match."
        assert result["error"] is None

    def test_returns_original_caption(self, monkeypatch):
        self._patch_judge(monkeypatch, 7.0, "Decent.")
        cap = "Shop our new collection ✨"
        result = eval_captions.score_caption(cap, _brand())
        assert result["caption"] == cap

    def test_score_clamped_to_10(self, monkeypatch):
        self._patch_judge(monkeypatch, 99.0, "Way too high.")
        result = eval_captions.score_caption("cap", _brand())
        assert result["score"] == 10.0

    def test_score_clamped_to_1(self, monkeypatch):
        self._patch_judge(monkeypatch, -5.0, "Way too low.")
        result = eval_captions.score_caption("cap", _brand())
        assert result["score"] == 1.0

    def test_judge_field_populated(self, monkeypatch):
        self._patch_judge(monkeypatch, 7.0, "OK")
        result = eval_captions.score_caption("cap", _brand())
        assert result["judge"] in ("nvidia", "openai", "anthropic")

    def test_graceful_error_on_api_failure(self, monkeypatch):
        """When the judge API raises, score_caption returns a default 5.0 with error."""
        monkeypatch.setattr(
            summarizer,
            "_call_for_provider",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("API down")),
        )
        fake = Settings(caption_provider="nvidia", nvidia_api_key="nvapi-x")
        monkeypatch.setattr(eval_captions, "settings", fake)
        monkeypatch.setattr(summarizer, "settings", fake)
        result = eval_captions.score_caption("cap", _brand())
        assert result["score"] == 5.0
        assert result["error"] is not None

    def test_image_description_included_in_prompt(self, monkeypatch):
        """Verify image_description reaches the prompt (captured via call args)."""
        captured = {}
        def capture_call(p, m, **kw):
            captured["prompt"] = m[0]["content"]
            return _fake_score_response(7.0, "ok")

        monkeypatch.setattr(summarizer, "_call_for_provider", capture_call)
        fake = Settings(caption_provider="nvidia", nvidia_api_key="nvapi-x")
        monkeypatch.setattr(eval_captions, "settings", fake)
        monkeypatch.setattr(summarizer, "settings", fake)

        eval_captions.score_caption("cap", _brand(), image_description="A red jacket on white background")
        assert "A red jacket on white background" in captured["prompt"]


# ── evaluate_batch ────────────────────────────────────────────────────────────

class TestEvaluateBatch:
    def _patch_sequential_scores(self, monkeypatch, scores: list[float]):
        """Return successive scores from the scores list on each call."""
        import json
        call_idx = {"i": 0}

        def fake_call(p, m, **kw):
            idx = call_idx["i"]
            call_idx["i"] += 1
            score = scores[idx] if idx < len(scores) else 5.0
            payload = json.dumps({"score": score, "reasoning": f"Score {score}"})
            return _openai_response(payload)

        monkeypatch.setattr(summarizer, "_call_for_provider", fake_call)
        fake = Settings(caption_provider="nvidia", nvidia_api_key="nvapi-x")
        monkeypatch.setattr(eval_captions, "settings", fake)
        monkeypatch.setattr(summarizer, "settings", fake)

    def test_returns_all_captions_scored(self, monkeypatch):
        self._patch_sequential_scores(monkeypatch, [7.0, 8.5, 6.0])
        results = eval_captions.evaluate_batch(
            ["Caption A", "Caption B", "Caption C"], _brand()
        )
        assert len(results) == 3

    def test_sorted_by_score_descending(self, monkeypatch):
        self._patch_sequential_scores(monkeypatch, [6.0, 9.0, 4.0])
        results = eval_captions.evaluate_batch(
            ["Low", "High", "Lowest"], _brand()
        )
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_list_returns_empty(self, monkeypatch):
        fake = Settings(caption_provider="nvidia", nvidia_api_key="nvapi-x")
        monkeypatch.setattr(eval_captions, "settings", fake)
        monkeypatch.setattr(summarizer, "settings", fake)
        assert eval_captions.evaluate_batch([], _brand()) == []

    def test_single_caption_batch(self, monkeypatch):
        self._patch_sequential_scores(monkeypatch, [8.0])
        results = eval_captions.evaluate_batch(["Solo caption"], _brand())
        assert len(results) == 1
        assert results[0]["score"] == 8.0


# ── stream hardening ──────────────────────────────────────────────────────────

class TestStreamHardening:
    """Verify that empty streams raise ConnectionError (triggering the fallback)."""

    def _make_resp(self, lines: list[bytes]):
        """Build a mock streaming response that yields the given lines."""
        class FakeResp:
            status_code = 200
            def raise_for_status(self): pass
            def iter_lines(self): return iter(lines)

        return FakeResp()

    def test_openai_compat_raises_on_empty_stream(self, monkeypatch):
        import requests as req
        # Stream that returns no content tokens (only keep-alive lines)
        monkeypatch.setattr(req, "post", lambda *a, **kw: self._make_resp([b": keep-alive", b""]))
        fake = Settings(nvidia_api_key="nvapi-x")
        monkeypatch.setattr(summarizer, "settings", fake)
        gen = summarizer._stream_openai_compat(
            [{"role": "user", "content": "hi"}],
            api_key="key", base_url="https://api.example.com/v1",
            model="test-model", max_tokens=100, temperature=0.5,
        )
        with pytest.raises(ConnectionError, match="no content"):
            list(gen)  # exhaust generator

    def test_anthropic_raises_on_empty_stream(self, monkeypatch):
        import requests as req
        # Stream with no content_block_delta events
        monkeypatch.setattr(req, "post", lambda *a, **kw: self._make_resp([b"data: {\"type\": \"ping\"}"]))
        fake = Settings(anthropic_api_key="sk-ant-x")
        monkeypatch.setattr(summarizer, "settings", fake)
        gen = summarizer._stream_anthropic(
            [{"role": "user", "content": "hi"}],
            max_tokens=100, temperature=0.5,
        )
        with pytest.raises(ConnectionError, match="no content"):
            list(gen)

    def test_openai_compat_does_not_raise_when_content_received(self, monkeypatch):
        import json

        import requests as req
        chunk = json.dumps({
            "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]
        }).encode()
        lines = [b"data: " + chunk, b"data: [DONE]"]
        monkeypatch.setattr(req, "post", lambda *a, **kw: self._make_resp(lines))
        fake = Settings(nvidia_api_key="nvapi-x")
        monkeypatch.setattr(summarizer, "settings", fake)
        tokens = list(summarizer._stream_openai_compat(
            [{"role": "user", "content": "hi"}],
            api_key="key", base_url="https://api.example.com/v1",
            model="test-model", max_tokens=100, temperature=0.5,
        ))
        assert tokens == ["Hello"]
