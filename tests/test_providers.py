"""
Tests for the model-agnostic provider layer in summarizer.py

Covers:
  - _require_api_key() per provider
  - _call() dispatch to the correct backend
  - _to_anthropic_messages() message-format conversion
  - _call_anthropic() response normalisation
  - _call_openai_compat() parameter forwarding
"""
from __future__ import annotations

import base64

import pytest
import requests

import summarizer
from config import Settings

# ── helpers ───────────────────────────────────────────────────────────────────

def _openai_response(text: str) -> dict:
    return {"choices": [{"message": {"content": text, "role": "assistant"}}]}


def _anthropic_response(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}]}


# ── _require_api_key ──────────────────────────────────────────────────────────

class TestRequireApiKey:
    def _patch_provider(self, monkeypatch, provider: str):
        fake = Settings(
            caption_provider=provider,
            nvidia_api_key=None,
            openai_api_key=None,
            anthropic_api_key=None,
        )
        monkeypatch.setattr(summarizer, "settings", fake)

    def test_nvidia_raises_when_key_missing(self, monkeypatch):
        self._patch_provider(monkeypatch, "nvidia")
        with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
            summarizer._require_api_key()

    def test_openai_raises_when_key_missing(self, monkeypatch):
        self._patch_provider(monkeypatch, "openai")
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            summarizer._require_api_key()

    def test_anthropic_raises_when_key_missing(self, monkeypatch):
        self._patch_provider(monkeypatch, "anthropic")
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            summarizer._require_api_key()

    def test_nvidia_passes_when_key_set(self, monkeypatch):
        fake = Settings(caption_provider="nvidia", nvidia_api_key="nvapi-test")
        monkeypatch.setattr(summarizer, "settings", fake)
        summarizer._require_api_key()  # should not raise

    def test_openai_passes_when_key_set(self, monkeypatch):
        fake = Settings(caption_provider="openai", openai_api_key="sk-test")
        monkeypatch.setattr(summarizer, "settings", fake)
        summarizer._require_api_key()  # should not raise

    def test_anthropic_passes_when_key_set(self, monkeypatch):
        fake = Settings(caption_provider="anthropic", anthropic_api_key="sk-ant-test")
        monkeypatch.setattr(summarizer, "settings", fake)
        summarizer._require_api_key()  # should not raise


# ── _call dispatch ────────────────────────────────────────────────────────────

class TestCallDispatch:
    """_call() should route to the correct backend function."""

    def _messages(self):
        return [{"role": "user", "content": "hello"}]

    def test_dispatch_to_openai_compat_for_nvidia(self, monkeypatch):
        called_with = {}
        def fake_openai_compat(messages, *, api_key, base_url, model, max_tokens, temperature):
            called_with.update({"provider": "nvidia_compat", "base_url": base_url})
            return _openai_response("ok")

        fake_settings = Settings(caption_provider="nvidia", nvidia_api_key="nvapi-x")
        monkeypatch.setattr(summarizer, "settings", fake_settings)
        monkeypatch.setattr(summarizer, "_call_openai_compat", fake_openai_compat)

        summarizer._call(self._messages(), max_tokens=100, temperature=0.5)
        assert called_with["provider"] == "nvidia_compat"
        assert "nvidia" in called_with["base_url"] or "integrate" in called_with["base_url"]

    def test_dispatch_to_openai_compat_for_openai(self, monkeypatch):
        called_with = {}
        def fake_openai_compat(messages, *, api_key, base_url, model, max_tokens, temperature):
            called_with.update({"provider": "openai_compat", "base_url": base_url})
            return _openai_response("ok")

        fake_settings = Settings(caption_provider="openai", openai_api_key="sk-test")
        monkeypatch.setattr(summarizer, "settings", fake_settings)
        monkeypatch.setattr(summarizer, "_call_openai_compat", fake_openai_compat)

        summarizer._call(self._messages(), max_tokens=100, temperature=0.5)
        assert "openai.com" in called_with["base_url"]

    def test_dispatch_to_anthropic(self, monkeypatch):
        called = {"n": 0}
        def fake_anthropic(messages, *, max_tokens, temperature):
            called["n"] += 1
            return _openai_response("ok")

        fake_settings = Settings(caption_provider="anthropic", anthropic_api_key="sk-ant-x")
        monkeypatch.setattr(summarizer, "settings", fake_settings)
        monkeypatch.setattr(summarizer, "_call_anthropic", fake_anthropic)

        summarizer._call(self._messages(), max_tokens=100, temperature=0.5)
        assert called["n"] == 1


# ── _to_anthropic_messages ────────────────────────────────────────────────────

class TestToAnthropicMessages:
    def test_plain_text_message_unchanged(self):
        msgs = [{"role": "user", "content": "Hello world"}]
        result = summarizer._to_anthropic_messages(msgs)
        assert result == [{"role": "user", "content": "Hello world"}]

    def test_text_block_preserved(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "Describe this"}]}]
        result = summarizer._to_anthropic_messages(msgs)
        assert result[0]["content"][0] == {"type": "text", "text": "Describe this"}

    def test_image_url_converted_to_anthropic_format(self):
        b64 = base64.b64encode(b"fakeimagedata").decode()
        msgs = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }]
        result = summarizer._to_anthropic_messages(msgs)
        img_block = result[0]["content"][0]
        assert img_block["type"] == "image"
        assert img_block["source"]["type"] == "base64"
        assert img_block["source"]["media_type"] == "image/jpeg"
        assert img_block["source"]["data"] == b64

    def test_mixed_content_converts_all_blocks(self):
        b64 = base64.b64encode(b"px").decode()
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Caption this"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }]
        result = summarizer._to_anthropic_messages(msgs)
        blocks = result[0]["content"]
        assert len(blocks) == 2
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "image"

    def test_remote_url_converted_to_url_source(self):
        msgs = [{
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}],
        }]
        result = summarizer._to_anthropic_messages(msgs)
        src = result[0]["content"][0]["source"]
        assert src["type"] == "url"
        assert src["url"] == "https://example.com/img.jpg"


# ── _call_anthropic response normalisation ────────────────────────────────────

class TestCallAnthropicNormalisation:
    """Verify _call_anthropic() converts Anthropic wire response → OpenAI format."""

    def _make_mock_response(self, anthropic_resp: dict, status_code: int = 200):
        """Create a mock requests.Response."""
        mock = type("MockResp", (), {
            "status_code": status_code,
            "json": lambda self: anthropic_resp,
            "raise_for_status": lambda self: None,
        })()
        return mock

    def test_single_text_block_normalised(self, monkeypatch):
        raw = _anthropic_response("Here is your caption.")
        monkeypatch.setattr(requests, "post", lambda *a, **kw: self._make_mock_response(raw))
        fake_settings = Settings(caption_provider="anthropic", anthropic_api_key="sk-ant-x")
        monkeypatch.setattr(summarizer, "settings", fake_settings)

        result = summarizer._call_anthropic(
            [{"role": "user", "content": "Hello"}], max_tokens=100, temperature=0.5
        )
        assert result["choices"][0]["message"]["content"] == "Here is your caption."

    def test_multiple_text_blocks_joined(self, monkeypatch):
        raw = {"content": [
            {"type": "text", "text": "Part one."},
            {"type": "text", "text": "Part two."},
        ]}
        monkeypatch.setattr(requests, "post", lambda *a, **kw: self._make_mock_response(raw))
        fake_settings = Settings(caption_provider="anthropic", anthropic_api_key="sk-ant-x")
        monkeypatch.setattr(summarizer, "settings", fake_settings)

        result = summarizer._call_anthropic(
            [{"role": "user", "content": "Hello"}], max_tokens=100, temperature=0.5
        )
        assert "Part one." in result["choices"][0]["message"]["content"]
        assert "Part two." in result["choices"][0]["message"]["content"]

    def test_non_text_blocks_ignored(self, monkeypatch):
        raw = {"content": [
            {"type": "tool_use", "id": "x", "name": "y"},
            {"type": "text", "text": "Final answer"},
        ]}
        monkeypatch.setattr(requests, "post", lambda *a, **kw: self._make_mock_response(raw))
        fake_settings = Settings(caption_provider="anthropic", anthropic_api_key="sk-ant-x")
        monkeypatch.setattr(summarizer, "settings", fake_settings)

        result = summarizer._call_anthropic(
            [{"role": "user", "content": "Hello"}], max_tokens=100, temperature=0.5
        )
        assert result["choices"][0]["message"]["content"] == "Final answer"


# ── _call_openai_compat parameter forwarding ──────────────────────────────────

class TestCallOpenAICompat:
    def _capture_request(self, monkeypatch):
        """Patches requests.post and returns a dict that captures the call args."""
        captured = {}

        class FakeResp:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return _openai_response("ok")

        def fake_post(url, *, headers, json, timeout):
            captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
            return FakeResp()

        monkeypatch.setattr(requests, "post", fake_post)
        return captured

    def test_uses_correct_base_url_and_model(self, monkeypatch):
        captured = self._capture_request(monkeypatch)
        summarizer._call_openai_compat(
            [{"role": "user", "content": "hi"}],
            api_key="test-key",
            base_url="https://custom.api.com/v1",
            model="my-model-v2",
            max_tokens=50,
            temperature=0.3,
        )
        assert captured["url"] == "https://custom.api.com/v1/chat/completions"
        assert captured["json"]["model"] == "my-model-v2"

    def test_bearer_auth_header(self, monkeypatch):
        captured = self._capture_request(monkeypatch)
        summarizer._call_openai_compat(
            [{"role": "user", "content": "hi"}],
            api_key="sk-secret",
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
            max_tokens=50,
            temperature=0.3,
        )
        assert captured["headers"]["Authorization"] == "Bearer sk-secret"

    def test_stream_is_false(self, monkeypatch):
        captured = self._capture_request(monkeypatch)
        summarizer._call_openai_compat(
            [{"role": "user", "content": "hi"}],
            api_key="key",
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
            max_tokens=50,
            temperature=0.3,
        )
        assert captured["json"]["stream"] is False
