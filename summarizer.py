"""
Core AI logic for Voxly.

Public API
----------
analyze_brand_voice(brand_name, platform, posts)
    Calls the configured AI provider (text-only) to extract a structured brand
    voice profile from real example posts. Returns a dict stored in SQLite and
    injected into every subsequent caption call.

generate_caption_variations(image_bytes, brand_profile, example_posts, ...)
    Calls the configured AI provider (vision) with a compressed image and the
    brand voice profile. Returns up to `num_variations` caption variations, each
    scored 1–10, plus a one-sentence image analysis. Retries transient errors
    with exponential backoff before surfacing to the caller.

Provider selection
------------------
Set CAPTION_PROVIDER in .env to switch backends at startup (no code changes):
    CAPTION_PROVIDER=nvidia      → NVIDIA NIM  (default, free key at build.nvidia.com)
    CAPTION_PROVIDER=openai      → OpenAI gpt-4o
    CAPTION_PROVIDER=anthropic   → Anthropic claude-3-5-sonnet

All three providers use identical public APIs — the abstraction is internal.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import time
from collections.abc import Generator
from io import BytesIO
from typing import Any

import requests
from PIL import Image, UnidentifiedImageError

from config import settings
from rag import retrieve_relevant_posts

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Characters and patterns that must never reach the model prompt.
_INJECT_RE = re.compile(
    r"ignore\s+(previous|all|above)\s+instructions?|"
    r"you\s+are\s+now\s+|"
    r"new\s+system\s+prompt|"
    r"<\|.*?\|>|"            # token delimiters (e.g. <|im_start|>)
    r"\[INST\]|\[/?SYS\]",   # LLaMA / Mistral instruction tokens
    re.IGNORECASE,
)
_MAX_POST_CHARS = 500   # per-post hard cap before prompt injection

# Retry behaviour for transient API failures (timeouts, 5xx).
_MAX_RETRIES: int = 3
_RETRY_BACKOFF_BASE: float = 2.0   # wait = base^attempt → 1 s, 2 s, 4 s

# Hashtag counts match platform best-practice guidelines.
_HASHTAG_COUNT: dict[str, int] = {"instagram": 10, "facebook": 5, "both": 10}

# Caption length guidance injected into the prompt.
_LENGTH_NOTES: dict[str, str] = {
    "short":  "Keep all captions concise — under 40 words. Punchy and direct.",
    "medium": "Keep captions 40–80 words. Balanced detail and readability.",
    "long":   "Write detailed captions, 80–150 words. Tell a story or include product details.",
}

# Number of real example posts to show the model (avoids prompt bloat).
_MAX_EXAMPLE_POSTS: int = 5


# ── Input sanitization ───────────────────────────────────────────────────────

def _sanitize_text(text: str) -> str:
    """
    Sanitize user-supplied text before injecting it into a model prompt.

    Applies three defences in order:
    1. Strip C0/C1 control characters (keep \\n and \\t for readability).
    2. Hard-truncate to ``_MAX_POST_CHARS`` characters.
    3. Replace known prompt-injection patterns with ``[removed]``.

    Args:
        text: Raw user-supplied string (brand post, context hint, etc.).

    Returns:
        Cleaned string safe for prompt interpolation.
    """
    # 1. Strip control characters except newline (0x0A) and tab (0x09)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # 2. Hard cap
    text = text[:_MAX_POST_CHARS]
    # 3. Injection patterns
    text = _INJECT_RE.sub("[removed]", text)
    return text.strip()


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_brand_voice(
    brand_name: str,
    platform: str,
    posts: list[str],
) -> dict[str, Any]:
    """
    Analyse example posts and return a structured brand voice profile.

    The profile captures tone, personality, style traits, emoji usage,
    preferred emojis, sentence style, signature words, CTA style, and
    target audience. It is stored once and reused for all caption calls.

    Args:
        brand_name: Human-readable brand or account name.
        platform:   Target platform — "instagram", "facebook", or "both".
        posts:      List of real posts to analyse (minimum 2, ideally 3–5).

    Returns:
        Profile dict with keys: tone, personality, style_traits, emoji_usage,
        preferred_emojis, sentence_style, signature_words, cta_style,
        audience, brand_name, platform.

    Raises:
        ValueError: If API key is missing, the model returns empty content,
                    or a valid voice profile cannot be parsed.
    """
    _require_api_key()
    logger.info("Analysing brand voice for '%s' from %d posts", brand_name, len(posts))

    formatted = "\n\n".join(
        f'Post {i + 1}: "{_sanitize_text(p)}"' for i, p in enumerate(posts)
    )

    prompt = (
        f"You are analyzing the social media voice of a brand called '{brand_name}'.\n\n"
        f"Study these {len(posts)} real posts carefully:\n\n"
        f"{formatted}\n\n"
        "Extract the brand voice and return ONLY a valid JSON object — no markdown, no extra text:\n"
        "{\n"
        '  "tone": "<2-4 word description of overall tone>",\n'
        '  "personality": "<one sentence describing brand personality>",\n'
        '  "style_traits": ["<trait 1>", "<trait 2>", "<trait 3>"],\n'
        '  "emoji_usage": "<none | minimal | moderate | heavy>",\n'
        '  "preferred_emojis": ["<emoji>", "<emoji>"],\n'
        '  "sentence_style": "<short | medium | long>",\n'
        '  "signature_words": ["<word/phrase>", "<word/phrase>"],\n'
        '  "cta_style": "<how they typically end posts>",\n'
        '  "audience": "<who they seem to be talking to>"\n'
        "}"
    )

    data    = _post_text(prompt, max_tokens=500, temperature=0.2)
    text    = _get_content(data)
    profile = _parse_json(text)

    if not profile or "tone" not in profile:
        logger.warning("Failed to parse voice profile from model response")
        raise ValueError(
            "Could not extract a voice profile. "
            "Add more varied example posts and try again."
        )

    profile["brand_name"] = brand_name
    profile["platform"]   = platform
    logger.info("Brand voice profile extracted successfully for '%s'", brand_name)
    return profile


def generate_caption_variations(
    image_bytes: bytes,
    brand_profile: dict[str, Any],
    example_posts: list[str],
    context: str = "",
    num_variations: int = 3,
    length: str = "medium",
    cached_embeddings: list[list[float]] | None = None,
) -> dict[str, Any]:
    """
    Generate multiple caption variations for a single image.

    Each variation includes Facebook and/or Instagram captions (based on the
    brand's platform setting), a set of hashtags, a 1–10 brand-voice score,
    and a short score reason. An image analysis sentence is also returned.

    API calls are retried up to _MAX_RETRIES times with exponential backoff
    on transient errors (timeouts, 5xx). Client errors (4xx) are not retried.

    Args:
        image_bytes:    Raw bytes of the uploaded image.
        brand_profile:  Voice profile dict from analyze_brand_voice().
        example_posts:  Real posts used as few-shot examples.
        context:        Optional one-line product/campaign hint
                        (e.g. "Winter jacket, price £89").
        num_variations: Number of distinct caption variations to generate.
        length:         Caption length preference — "short", "medium", or "long".

    Returns:
        {
            "image_analysis": str,     # 1-2 sentence description of the image
            "variations": [
                {
                    "facebook_caption":  str | None,
                    "instagram_caption": str | None,
                    "hashtags":          list[str],
                    "score":             float,   # 1–10
                    "score_reason":      str,
                }
            ],
            "error": str | None,       # set only on failure
        }
    """
    _require_api_key()
    logger.info(
        "Generating %d caption variation(s) for image (%d bytes), length=%s",
        num_variations, len(image_bytes), length,
    )

    try:
        image    = _open_image(image_bytes)
        nv_bytes, mime = _compress_image(image)
        platform = brand_profile.get("platform", "both")

        prompt = _build_variations_prompt(
            brand_profile, example_posts, platform,
            context.strip(), num_variations, length,
            cached_embeddings=cached_embeddings,
        )
        data   = _post_vision_with_retry(nv_bytes, mime, prompt=prompt, max_tokens=1600, temperature=0.55)
        text   = _get_content(data)

        parsed         = _parse_json(text) or {}
        image_analysis = parsed.get("image_analysis", "")
        variations     = _parse_variations(parsed, platform)

        if not variations:
            # Last-resort fallback: try parsing as old single-caption format
            old = _parse_captions(text)
            if old.get("facebook_caption") or old.get("instagram_caption"):
                variations = [{
                    "facebook_caption":  old.get("facebook_caption"),
                    "instagram_caption": old.get("instagram_caption"),
                    "hashtags":          old.get("hashtags", []),
                    "score":             7.0,
                    "score_reason":      "Generated caption",
                }]
                logger.warning("Fell back to legacy caption parser; model may have returned non-standard JSON")

        # Highest score first
        variations.sort(key=lambda v: v.get("score", 0), reverse=True)

        logger.info("Generated %d variation(s) successfully", len(variations))
        return {"image_analysis": image_analysis, "variations": variations, "error": None}

    except (requests.RequestException, ValueError) as exc:
        logger.error("Caption generation failed: %s", exc, exc_info=True)
        return {"image_analysis": "", "variations": [], "error": str(exc)}


def generate_caption(
    image_bytes: bytes,
    brand_profile: dict[str, Any],
    example_posts: list[str],
    context: str = "",
) -> dict[str, Any]:
    """Backwards-compatible single-caption wrapper around generate_caption_variations."""
    result = generate_caption_variations(
        image_bytes, brand_profile, example_posts, context=context, num_variations=1
    )
    if result["error"]:
        return {"facebook_caption": None, "instagram_caption": None, "hashtags": [], "error": result["error"]}
    best = result["variations"][0] if result["variations"] else {}
    return {
        "facebook_caption":  best.get("facebook_caption"),
        "instagram_caption": best.get("instagram_caption"),
        "hashtags":          best.get("hashtags", []),
        "error":             None,
    }


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_variations_prompt(
    profile: dict[str, Any],
    posts: list[str],
    platform: str,
    context: str,
    num_variations: int,
    length: str,
    cached_embeddings: list[list[float]] | None = None,
) -> str:
    """Build the vision-model prompt requesting `num_variations` caption variations."""
    brand_name  = profile.get("brand_name", "this brand")
    tone        = profile.get("tone", "")
    personality = profile.get("personality", "")
    traits      = ", ".join(profile.get("style_traits") or [])
    emojis      = profile.get("emoji_usage", "")
    fav_emojis  = " ".join(profile.get("preferred_emojis") or [])
    sentences   = profile.get("sentence_style", "")
    sig_words   = ", ".join(f'"{w}"' for w in (profile.get("signature_words") or []))
    cta         = profile.get("cta_style", "")
    audience    = profile.get("audience", "")
    # Semantic retrieval: pick the most contextually relevant posts.
    # cached_embeddings avoids re-running the model on every generation call.
    sanitized_posts = [_sanitize_text(p) for p in posts]
    rag_query       = " ".join(filter(None, [brand_name, tone, context]))
    selected        = retrieve_relevant_posts(
        rag_query, sanitized_posts,
        cached_embeddings=cached_embeddings,
        top_k=_MAX_EXAMPLE_POSTS,
    )
    examples    = "\n".join(
        f'  {i+1}. "{p.strip()}"'
        for i, p in enumerate(selected)
    )
    length_note   = _LENGTH_NOTES.get(length, _LENGTH_NOTES["medium"])
    hashtag_count = _HASHTAG_COUNT.get(platform, _HASHTAG_COUNT["both"])

    if platform == "facebook":
        cap_schema = '"facebook_caption": "<2-3 sentence post>", "instagram_caption": null'
    elif platform == "instagram":
        cap_schema = '"facebook_caption": null, "instagram_caption": "<1-3 sentence caption with emojis>"'
    else:
        cap_schema = '"facebook_caption": "<2-3 sentence post>", "instagram_caption": "<1-3 sentence caption>"'

    context_block = (
        f"\nSPECIFIC CONTEXT: {context}\n"
        "Use this to make captions concrete — mention the product, price, or detail if relevant.\n"
    ) if context else ""

    # JSON schema shown inline so the model has a concrete target to match.
    schema = (
        "{\n"
        '  "image_analysis": "<1-2 sentence image description>",\n'
        '  "variations": [\n'
        "    {\n"
        f'      {cap_schema},\n'
        f'      "hashtags": ["#tag1", "..." ({hashtag_count} total)],\n'
        '      "score": <1-10>,\n'
        '      "score_reason": "<brief reason>"\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    return (
        f"You are the social media voice of '{brand_name}'.\n\n"
        f"BRAND VOICE PROFILE:\n"
        f"  Tone: {tone}\n"
        f"  Personality: {personality}\n"
        f"  Style: {traits}\n"
        f"  Emoji Usage: {emojis} — preferred: {fav_emojis}\n"
        f"  Sentence Length: {sentences}\n"
        f"  Signature Words: {sig_words}\n"
        f"  How Posts End: {cta}\n"
        f"  Audience: {audience}\n\n"
        f"REAL EXAMPLES (study the rhythm, language, and energy):\n"
        f"{examples}\n"
        f"{context_block}\n"
        f"LENGTH: {length_note}\n\n"
        f"Look at this image carefully and:\n"
        f"1. Describe what you see in 1-2 sentences (product, setting, mood, colors).\n"
        f"2. Write {num_variations} DISTINCT caption variations — different openings, angles, "
        f"or structures — each unmistakably sounding like {brand_name}.\n"
        f"3. Score each 1–10 for brand voice match + audience appeal.\n\n"
        f"Return ONLY valid JSON — no markdown, no code fences:\n"
        f"{schema}"
    )


# ── Provider-agnostic API layer ───────────────────────────────────────────────
#
# All public functions call _post_text() or _post_vision_with_retry(), which
# both route to _call_with_retry() → _call() → provider-specific implementation.
#
# Adding a new provider:
#   1. Add its API key / model settings to config.py
#   2. Implement _call_<provider>(messages, *, max_tokens, temperature)
#      returning a normalised OpenAI-format dict {"choices": [...]}
#   3. Add a branch in _call() and _require_api_key()

def _post_vision_with_retry(
    image_bytes: bytes,
    mime: str,
    *,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """Call the vision endpoint with exponential backoff retry.

    Builds an OpenAI-compatible message with an image_url content block;
    _call() converts it to the wire format expected by the active provider.
    """
    b64      = base64.b64encode(image_bytes).decode("ascii")
    messages = [{
        "role": "user",
        "content": [
            {"type": "text",      "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ],
    }]
    return _call_with_retry(messages, max_tokens=max_tokens, temperature=temperature)


def _post_text(prompt: str, *, max_tokens: int, temperature: float) -> dict[str, Any]:
    """Call the text-only endpoint (no retry — used for brand voice extraction only)."""
    return _call(
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )


def stream_text(
    prompt: str,
    *,
    max_tokens: int = 800,
    temperature: float = 0.4,
) -> Generator[str, None, None]:
    """
    Stream a text-only response token by token from the active provider.

    Yields successive text chunks as they arrive from the API, enabling
    ``st.write_stream()`` to render tokens in real time.

    Falls back to a single yielded string if the provider does not support
    streaming or if streaming fails mid-response.

    Args:
        prompt:      The user prompt to send.
        max_tokens:  Maximum tokens to generate.
        temperature: Sampling temperature.

    Yields:
        str — successive text fragments from the model.
    """
    _require_api_key()
    provider = settings.caption_provider
    messages: list[dict] = [{"role": "user", "content": prompt}]
    try:
        if provider == "anthropic":
            yield from _stream_anthropic(messages, max_tokens=max_tokens, temperature=temperature)
        else:
            api_key  = settings.openai_api_key or "" if provider == "openai" else settings.nvidia_api_key or ""
            base_url = "https://api.openai.com/v1" if provider == "openai" else settings.nvidia_api_base_url.rstrip("/")
            model    = settings.openai_model if provider == "openai" else settings.nvidia_vision_model
            yield from _stream_openai_compat(
                messages, api_key=api_key, base_url=base_url,
                model=model, max_tokens=max_tokens, temperature=temperature,
            )
    except Exception as exc:
        logger.warning("Streaming failed (%s) — falling back to batch call", exc)
        data = _post_text(prompt, max_tokens=max_tokens, temperature=temperature)
        yield _get_content(data)


def _stream_openai_compat(
    messages: list[dict],
    *,
    api_key: str,
    base_url: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> Generator[str, None, None]:
    """Stream tokens from an OpenAI-compatible SSE endpoint.

    Raises:
        ConnectionError: If the stream connects but yields no content tokens,
                         which typically indicates a silent API failure.
    """
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model, "messages": messages,  # type: ignore[dict-item]
            "temperature": temperature, "max_tokens": max_tokens, "stream": True,
        },
        stream=True,
        timeout=settings.request_timeout_seconds,
    )
    resp.raise_for_status()
    content_received = False
    for raw in resp.iter_lines():
        if not raw or not raw.startswith(b"data: "):
            continue
        payload = raw[6:]
        if payload == b"[DONE]":
            break
        try:
            chunk   = json.loads(payload)
            content = chunk["choices"][0]["delta"].get("content") or ""
            if content:
                content_received = True
                yield content
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
    if not content_received:
        raise ConnectionError("Stream connected but yielded no content — treating as transient failure")


def _stream_anthropic(
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float,
) -> Generator[str, None, None]:
    """Stream tokens from the Anthropic Messages SSE endpoint.

    Raises:
        ConnectionError: If the stream connects but yields no content tokens,
                         which typically indicates a silent API failure.
    """
    anthropic_messages = _to_anthropic_messages(messages)
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": settings.anthropic_api_key or "",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.anthropic_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anthropic_messages,  # type: ignore[dict-item]
            "stream": True,
        },
        stream=True,
        timeout=settings.request_timeout_seconds,
    )
    resp.raise_for_status()
    content_received = False
    for raw in resp.iter_lines():
        if not raw or not raw.startswith(b"data: "):
            continue
        try:
            event = json.loads(raw[6:])
            if event.get("type") == "content_block_delta":
                text = event.get("delta", {}).get("text") or ""
                if text:
                    content_received = True
                    yield text
        except (json.JSONDecodeError, KeyError):
            continue
    if not content_received:
        raise ConnectionError("Stream connected but yielded no content — treating as transient failure")


# ── Provider failover helpers ─────────────────────────────────────────────────

_PROVIDER_PRIORITY: tuple[str, ...] = ("nvidia", "openai", "anthropic")


def _has_key(provider: str) -> bool:
    """Return True if the given provider has an API key configured."""
    if provider == "nvidia":
        return bool(settings.nvidia_api_key)
    if provider == "openai":
        return bool(settings.openai_api_key)
    if provider == "anthropic":
        return bool(settings.anthropic_api_key)
    return False


def _failover_chain() -> list[str]:
    """
    Return the ordered list of providers to try, starting with the primary.

    Only providers that have an API key configured are included.
    This means a single-key setup has no failover; multi-key setups get
    automatic resilience when the primary is unavailable.
    """
    primary = settings.caption_provider
    return [primary] + [
        p for p in _PROVIDER_PRIORITY
        if p != primary and _has_key(p)
    ]


def _call_for_provider(
    provider: str,
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """Route a single call to the provider-specific implementation."""
    if provider == "openai":
        return _call_openai_compat(
            messages,
            api_key=settings.openai_api_key or "",
            base_url="https://api.openai.com/v1",
            model=settings.openai_model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    if provider == "anthropic":
        return _call_anthropic(messages, max_tokens=max_tokens, temperature=temperature)
    # Default: nvidia
    return _call_openai_compat(
        messages,
        api_key=settings.nvidia_api_key or "",
        base_url=settings.nvidia_api_base_url.rstrip("/"),
        model=settings.nvidia_vision_model,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _call(
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """Single dispatch call to the configured provider — no retry or failover."""
    return _call_for_provider(
        settings.caption_provider, messages,
        max_tokens=max_tokens, temperature=temperature,
    )


def _call_with_retry(
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """
    Retry + automatic provider failover for transient API errors.

    Retry policy per provider:
        - Timeout      → retry up to _MAX_RETRIES with exponential backoff
        - HTTP 5xx     → retry up to _MAX_RETRIES with exponential backoff
        - HTTP 4xx     → raise immediately (client error, no retry/failover)

    Failover:
        When all retries for the primary provider are exhausted, the next
        configured provider (by _PROVIDER_PRIORITY) is tried automatically.
        Failover only happens when more than one provider has a key set.
    """
    chain    = _failover_chain()
    last_exc: Exception | None = None

    for p_idx, provider in enumerate(chain):
        is_last_provider = p_idx == len(chain) - 1

        for attempt in range(_MAX_RETRIES):
            try:
                return _call_for_provider(
                    provider, messages,
                    max_tokens=max_tokens, temperature=temperature,
                )

            except requests.exceptions.Timeout as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    wait = _RETRY_BACKOFF_BASE ** attempt
                    logger.warning(
                        "Timeout on %s attempt %d/%d — retrying in %.1f s",
                        provider, attempt + 1, _MAX_RETRIES, wait,
                    )
                    time.sleep(wait)

            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                if status < 500:
                    logger.error("Non-retryable HTTP %d from %s", status, provider)
                    raise  # 4xx — never retry or failover
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    wait = _RETRY_BACKOFF_BASE ** attempt
                    logger.warning(
                        "HTTP %d on %s attempt %d/%d — retrying in %.1f s",
                        status, provider, attempt + 1, _MAX_RETRIES, wait,
                    )
                    time.sleep(wait)

        # All retries for this provider exhausted
        if not is_last_provider:
            next_provider = chain[p_idx + 1]
            logger.warning(
                "Provider '%s' exhausted all %d retries — failing over to '%s'",
                provider, _MAX_RETRIES, next_provider,
            )

    assert last_exc is not None
    raise last_exc


def _call_openai_compat(
    messages: list[dict],
    *,
    api_key: str,
    base_url: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """
    Call any OpenAI-compatible /chat/completions endpoint.

    Used for both NVIDIA NIM and OpenAI — they share the same wire format
    (image_url content blocks, Bearer auth, choices response).
    """
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        },
        json={
            "model":       model,
            "messages":    messages,  # type: ignore[dict-item]
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "stream":      False,
        },
        timeout=settings.request_timeout_seconds,
    )
    resp.raise_for_status()
    return resp.json()


def _call_anthropic(
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """
    Call the Anthropic Messages API and return a normalised OpenAI-format dict.

    Converts OpenAI-style ``image_url`` content blocks to Anthropic's
    ``image / source / base64`` format, then normalises the response back to
    ``{"choices": [{"message": {"content": str, "role": "assistant"}}]}``
    so all downstream code stays provider-agnostic.
    """
    anthropic_messages = _to_anthropic_messages(messages)
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key":         settings.anthropic_api_key or "",
            "anthropic-version": "2023-06-01",
            "Content-Type":      "application/json",
        },
        json={
            "model":       settings.anthropic_model,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "messages":    anthropic_messages,  # type: ignore[dict-item]
        },
        timeout=settings.request_timeout_seconds,
    )
    resp.raise_for_status()
    data = resp.json()

    # Normalise Anthropic response → OpenAI choices format
    text_parts = [
        block["text"]
        for block in data.get("content", [])
        if block.get("type") == "text"
    ]
    content = " ".join(text_parts)
    return {"choices": [{"message": {"content": content, "role": "assistant"}}]}


def _to_anthropic_messages(messages: list[dict]) -> list[dict]:
    """
    Convert OpenAI-format messages (including image_url blocks) to Anthropic format.

    OpenAI image block:
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<b64>"}}

    Anthropic image block:
        {"type": "image", "source": {"type": "base64",
                                     "media_type": "image/jpeg",
                                     "data": "<b64>"}}
    """
    result: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            result.append({"role": msg["role"], "content": content})
        elif isinstance(content, list):
            blocks: list[dict] = []
            for block in content:
                if block.get("type") == "text":
                    blocks.append({"type": "text", "text": block["text"]})
                elif block.get("type") == "image_url":
                    url = (block.get("image_url") or {}).get("url", "")
                    if url.startswith("data:"):
                        # data:<media_type>;base64,<data>
                        _, rest   = url.split(":", 1)
                        media_type, b64_data = rest.split(";base64,", 1)
                        blocks.append({
                            "type":   "image",
                            "source": {
                                "type":       "base64",
                                "media_type": media_type,
                                "data":       b64_data,
                            },
                        })
                    else:
                        # Remote URL — pass as-is (Anthropic supports this too)
                        blocks.append({
                            "type":   "image",
                            "source": {"type": "url", "url": url},
                        })
            result.append({"role": msg["role"], "content": blocks})
    return result


def _get_content(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("Model returned no choices — the API response may be malformed.")
    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        content = " ".join(
            item.get("text", "")
            for item in content
            if item.get("type") in {"text", "output_text"}
        )
    if not content.strip():
        raise ValueError("Model returned empty content.")
    return content.strip()


# ── JSON helpers ──────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict[str, Any] | None:
    """
    Try to parse a JSON dict from `text`.

    Attempts (in order):
        1. Parse the raw text directly.
        2. Extract from a ```json ... ``` markdown code block.
        3. Extract the first {...} block with a regex.
    """
    for candidate in _json_candidates(text):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _parse_variations(data: dict[str, Any], platform: str) -> list[dict[str, Any]]:
    """Normalise the 'variations' list from the model response."""
    raw = data.get("variations")
    if not isinstance(raw, list):
        return []

    result: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        tags = item.get("hashtags") or []
        if isinstance(tags, str):
            tags = tags.split()
        try:
            score = float(item.get("score") or 7.0)
        except (TypeError, ValueError):
            score = 7.0

        result.append({
            "facebook_caption":  item.get("facebook_caption") or None,
            "instagram_caption": item.get("instagram_caption") or None,
            "hashtags": [
                t if t.startswith("#") else f"#{t}"
                for t in tags if isinstance(t, str)
            ],
            "score":        score,
            "score_reason": item.get("score_reason") or "",
        })
    return result


def _parse_captions(text: str) -> dict[str, Any]:
    """Legacy single-caption parser — fallback when variations parsing fails."""
    data = _parse_json(text) or {}
    tags = data.get("hashtags") or []
    if isinstance(tags, str):
        tags = tags.split()
    return {
        "facebook_caption":  data.get("facebook_caption") or None,
        "instagram_caption": data.get("instagram_caption") or None,
        "hashtags": [
            t if t.startswith("#") else f"#{t}"
            for t in tags if isinstance(t, str)
        ],
    }


def _json_candidates(text: str) -> list[str]:
    candidates = [text]
    if m := re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text):
        candidates.append(m.group(1))
    if m := re.search(r"\{[\s\S]*\}", text):
        candidates.append(m.group(0))
    return candidates


# ── Image helpers ─────────────────────────────────────────────────────────────

def _open_image(image_bytes: bytes) -> Image.Image:
    """Decode image bytes and convert to RGB, preserving original format metadata."""
    try:
        raw = Image.open(BytesIO(image_bytes))
        img = raw.convert("RGB")
        img.format = raw.format
        return img
    except UnidentifiedImageError as exc:
        raise ValueError("Image could not be decoded — unsupported or corrupt file.") from exc


def _compress_image(image: Image.Image) -> tuple[bytes, str]:
    """
    Resize and compress an image to fit within NVIDIA_MAX_IMAGE_BYTES.

    Strategy:
        1. Thumbnail to 1024×1024 (preserves aspect ratio).
        2. Try JPEG quality levels 85→75→65→55→45 until size fits.
        3. If still too large, reduce to 768×768 at quality 40.

    Returns:
        (compressed_bytes, mime_type)
    """
    prepared = image.copy()
    prepared.thumbnail((1024, 1024))

    for quality in (85, 75, 65, 55, 45):
        buf = BytesIO()
        prepared.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
        if len(data) <= settings.nvidia_max_image_bytes:
            logger.debug("Image compressed to %d bytes at JPEG quality %d", len(data), quality)
            return data, "image/jpeg"

    # Last resort — aggressive resize + minimum quality
    buf = BytesIO()
    prepared.thumbnail((768, 768))
    prepared.save(buf, format="JPEG", quality=40, optimize=True)
    logger.warning(
        "Image could not be reduced below %d KB at normal quality; "
        "used 768×768 / q40 fallback (%d bytes)",
        settings.nvidia_max_image_bytes // 1024, len(buf.getvalue()),
    )
    return buf.getvalue(), "image/jpeg"


# ── Guards ────────────────────────────────────────────────────────────────────

def _require_api_key() -> None:
    """Raise ValueError if the API key for the active provider is missing."""
    provider = settings.caption_provider
    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file (see .env.example) and restart the app."
            )
    elif provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file (see .env.example) and restart the app."
            )
    else:  # nvidia (default)
        if not settings.nvidia_api_key:
            raise ValueError(
                "NVIDIA_API_KEY is not set. "
                "Add it to your .env file (see .env.example) and restart the app."
            )
