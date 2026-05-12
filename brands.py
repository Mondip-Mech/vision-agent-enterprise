from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

_FILE = os.path.join(os.path.dirname(__file__), "brands.json")


def load_all() -> list[dict[str, Any]]:
    if not os.path.exists(_FILE):
        return []
    try:
        with open(_FILE, encoding="utf-8") as f:
            return json.load(f).get("brands", [])
    except (json.JSONDecodeError, OSError):
        return []


def save_brand(name: str, platform: str, profile: dict, example_posts: list[str]) -> dict[str, Any]:
    brands = load_all()
    brand: dict[str, Any] = {
        "id": str(uuid4()),
        "name": name,
        "platform": platform,
        "profile": profile,
        "example_posts": example_posts,
        "created_at": datetime.now(UTC).isoformat(),
    }
    brands.append(brand)
    _write(brands)
    return brand


def update_brand(brand_id: str, name: str, platform: str, profile: dict, example_posts: list[str]) -> dict[str, Any]:
    brands = load_all()
    for i, b in enumerate(brands):
        if b["id"] == brand_id:
            brands[i] = {
                **b,
                "name": name,
                "platform": platform,
                "profile": profile,
                "example_posts": example_posts,
                "updated_at": datetime.now(UTC).isoformat(),
            }
            _write(brands)
            return brands[i]
    raise ValueError(f"Brand {brand_id} not found.")


def delete_brand(brand_id: str) -> None:
    _write([b for b in load_all() if b["id"] != brand_id])


def get_by_id(brand_id: str) -> dict[str, Any] | None:
    return next((b for b in load_all() if b["id"] == brand_id), None)


def _write(brands: list) -> None:
    with open(_FILE, "w", encoding="utf-8") as f:
        json.dump({"brands": brands}, f, indent=2, ensure_ascii=False)
