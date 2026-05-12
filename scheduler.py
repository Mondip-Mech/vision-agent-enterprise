from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests

_BASE = "https://api.bufferapp.com/1"
_TIMEOUT = 15


def get_profiles(access_token: str) -> list[dict]:
    """Return the user's connected Buffer profiles."""
    resp = requests.get(
        f"{_BASE}/profiles.json",
        params={"access_token": access_token},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def publish_post(
    access_token: str,
    profile_id: str,
    text: str,
    scheduled_at: str | None = None,
) -> dict[str, Any]:
    """
    Post text to a single Buffer profile.
    scheduled_at: ISO-8601 string (UTC). If None, adds to the queue immediately.
    Returns Buffer's API response dict.
    """
    payload: dict[str, Any] = {
        "access_token": access_token,
        "profile_ids[]": profile_id,
        "text": text,
    }
    if scheduled_at:
        dt = datetime.fromisoformat(scheduled_at.replace("Z", "+00:00"))
        payload["scheduled_at"] = dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    resp = requests.post(f"{_BASE}/updates/create.json", data=payload, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_pending_posts(access_token: str, profile_id: str) -> list[dict]:
    """Return scheduled (pending) posts for a Buffer profile."""
    resp = requests.get(
        f"{_BASE}/profiles/{profile_id}/updates/pending.json",
        params={"access_token": access_token},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("updates", [])
