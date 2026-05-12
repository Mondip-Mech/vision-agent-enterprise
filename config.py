"""
Application settings loaded from environment variables.

Provider selection
------------------
Set CAPTION_PROVIDER to one of:  nvidia (default) | openai | anthropic

Depending on the provider, supply the corresponding API key:
    NVIDIA_API_KEY    — free key at https://build.nvidia.com/
    OPENAI_API_KEY    — https://platform.openai.com/api-keys
    ANTHROPIC_API_KEY — https://console.anthropic.com/settings/keys

All other variables are optional and have sensible defaults.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

_VALID_PROVIDERS = ("nvidia", "openai", "anthropic")


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name!r} must be a float, got {value!r}") from exc


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name!r} must be an integer, got {value!r}") from exc


def _provider_env() -> str:
    raw = os.getenv("CAPTION_PROVIDER", "nvidia").lower().strip()
    if raw not in _VALID_PROVIDERS:
        raise ValueError(
            f"CAPTION_PROVIDER={raw!r} is not valid. "
            f"Choose one of: {', '.join(_VALID_PROVIDERS)}"
        )
    return raw


@dataclass(frozen=True)
class Settings:
    # ── Provider selection ────────────────────────────────────────────────────
    # Controls which AI backend is used for all caption + brand-voice calls.
    # Valid values: "nvidia" (default) | "openai" | "anthropic"
    caption_provider: str = _provider_env()

    # ── NVIDIA NIM ────────────────────────────────────────────────────────────
    nvidia_api_key: str | None = os.getenv("NVIDIA_API_KEY")
    nvidia_api_base_url: str = os.getenv(
        "NVIDIA_API_BASE_URL",
        "https://integrate.api.nvidia.com/v1",
    )
    nvidia_vision_model: str = os.getenv(
        "NVIDIA_VISION_MODEL",
        "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    # ── Anthropic ─────────────────────────────────────────────────────────────
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

    # ── Shared ────────────────────────────────────────────────────────────────

    # Max bytes of a compressed JPEG sent to the vision API (~180 KB keeps latency low).
    # The image is re-compressed at progressively lower quality until it fits.
    nvidia_max_image_bytes: int = _int_env("NVIDIA_MAX_IMAGE_BYTES", 180 * 1024)

    # Per-request timeout; vision calls can be slow on cold start.
    request_timeout_seconds: float = _float_env("REQUEST_TIMEOUT_SECONDS", 60.0)


settings = Settings()
