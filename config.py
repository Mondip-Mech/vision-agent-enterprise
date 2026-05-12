"""
Application settings loaded from environment variables.

Required:
    NVIDIA_API_KEY — NVIDIA NIM API key (get one at build.nvidia.com)

All other variables are optional and have sensible defaults.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


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


@dataclass(frozen=True)
class Settings:
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

    # Max bytes of a compressed JPEG sent to NVIDIA (~180 KB keeps latency low).
    # The image is re-compressed at progressively lower quality until it fits.
    nvidia_max_image_bytes: int = _int_env("NVIDIA_MAX_IMAGE_BYTES", 180 * 1024)

    # Per-request timeout; NVIDIA vision calls can be slow on first load.
    request_timeout_seconds: float = _float_env("REQUEST_TIMEOUT_SECONDS", 60.0)


settings = Settings()
