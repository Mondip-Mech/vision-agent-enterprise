"""
SQLite data layer for Voxly.

Tables
------
brands      — Named brand voice profiles with example posts.
captions    — Generated captions saved to the content calendar.
performance — Engagement metrics logged per published caption.
settings    — Simple key/value store for app-level config (e.g. Buffer token).

All tables are created automatically on first run via init_db().
The database file is written to the project directory as ``voxly.db``
and is excluded from version control via .gitignore.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

_DB_PATH = os.path.join(os.path.dirname(__file__), "voxly.db")


@contextmanager
def _conn():
    """Yield a committed SQLite connection; roll back and close on error."""
    con = sqlite3.connect(_DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db() -> None:
    """Create all tables if they do not already exist."""
    with _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS brands (
                id           TEXT PRIMARY KEY,
                name         TEXT NOT NULL,
                platform     TEXT NOT NULL,
                profile_json TEXT NOT NULL,
                posts_json   TEXT NOT NULL,
                created_at   TEXT NOT NULL,
                updated_at   TEXT
            );

            CREATE TABLE IF NOT EXISTS captions (
                id             TEXT PRIMARY KEY,
                brand_id       TEXT NOT NULL,
                brand_name     TEXT NOT NULL DEFAULT '',
                image_name     TEXT NOT NULL,
                platform       TEXT NOT NULL,
                caption        TEXT NOT NULL,
                hashtags       TEXT NOT NULL DEFAULT '',
                score          REAL DEFAULT NULL,
                context        TEXT DEFAULT '',
                length_pref    TEXT DEFAULT 'medium',
                scheduled_date TEXT DEFAULT NULL,
                published      INTEGER NOT NULL DEFAULT 0,
                created_at     TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS performance (
                id          TEXT PRIMARY KEY,
                caption_id  TEXT NOT NULL,
                likes       INTEGER DEFAULT 0,
                comments    INTEGER DEFAULT 0,
                reach       INTEGER DEFAULT 0,
                saves       INTEGER DEFAULT 0,
                recorded_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT
            );
        """)
    logger.debug("Database initialised at %s", _DB_PATH)


# ── Settings ──────────────────────────────────────────────────────────────────

def get_setting(key: str) -> str:
    """Return the value for *key*, or an empty string if not set."""
    with _conn() as con:
        row = con.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else ""


def set_setting(key: str, value: str) -> None:
    """Upsert a key/value pair in the settings table."""
    with _conn() as con:
        con.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value or ""),
        )


# ── Brands ────────────────────────────────────────────────────────────────────

def load_brands() -> list[dict[str, Any]]:
    """Return all saved brand profiles, newest first."""
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM brands ORDER BY created_at DESC"
        ).fetchall()
    return [_hydrate_brand(dict(r)) for r in rows]


def save_brand(
    name: str,
    platform: str,
    profile: dict[str, Any],
    example_posts: list[str],
) -> dict[str, Any]:
    """Insert a new brand and return the full saved record."""
    bid = str(uuid4())
    with _conn() as con:
        con.execute(
            "INSERT INTO brands (id, name, platform, profile_json, posts_json, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (bid, name, platform, json.dumps(profile), json.dumps(example_posts), _now()),
        )
    logger.info("Brand saved: id=%s name=%r", bid, name)
    return get_brand(bid)


def update_brand(
    brand_id: str,
    name: str,
    platform: str,
    profile: dict[str, Any],
    example_posts: list[str],
) -> dict[str, Any]:
    """Update an existing brand and return the updated record."""
    with _conn() as con:
        con.execute(
            "UPDATE brands SET name=?, platform=?, profile_json=?, posts_json=?, updated_at=?"
            " WHERE id=?",
            (name, platform, json.dumps(profile), json.dumps(example_posts), _now(), brand_id),
        )
    logger.info("Brand updated: id=%s name=%r", brand_id, name)
    return get_brand(brand_id)


def delete_brand(brand_id: str) -> None:
    """Delete a brand by ID."""
    with _conn() as con:
        con.execute("DELETE FROM brands WHERE id = ?", (brand_id,))
    logger.info("Brand deleted: id=%s", brand_id)


def get_brand(brand_id: str) -> dict[str, Any] | None:
    """Return a single brand by ID, or None if not found."""
    with _conn() as con:
        row = con.execute("SELECT * FROM brands WHERE id = ?", (brand_id,)).fetchone()
    return _hydrate_brand(dict(row)) if row else None


def _hydrate_brand(row: dict[str, Any]) -> dict[str, Any]:
    """Deserialise JSON columns into Python objects."""
    try:
        row["profile"] = json.loads(row.pop("profile_json"))
    except (json.JSONDecodeError, KeyError):
        logger.warning("Could not parse profile_json for brand id=%s", row.get("id"))
        row["profile"] = {}
    try:
        row["example_posts"] = json.loads(row.pop("posts_json"))
    except (json.JSONDecodeError, KeyError):
        logger.warning("Could not parse posts_json for brand id=%s", row.get("id"))
        row["example_posts"] = []
    return row


# ── Captions ──────────────────────────────────────────────────────────────────

def save_caption(
    brand_id: str,
    brand_name: str,
    image_name: str,
    platform: str,
    caption: str,
    hashtags: str,
    score: float | None = None,
    context: str = "",
    length_pref: str = "medium",
    scheduled_date: str | None = None,
) -> dict[str, Any]:
    """Insert a new caption and return the saved record."""
    cid = str(uuid4())
    with _conn() as con:
        con.execute(
            """INSERT INTO captions
               (id, brand_id, brand_name, image_name, platform, caption, hashtags,
                score, context, length_pref, scheduled_date, published, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
            (
                cid, brand_id, brand_name, image_name, platform,
                caption, hashtags, score, context, length_pref, scheduled_date, _now(),
            ),
        )
    logger.debug("Caption saved: id=%s platform=%s image=%s", cid, platform, image_name)
    return get_caption(cid)


def get_caption(caption_id: str) -> dict[str, Any] | None:
    """Return a single caption by ID, or None if not found."""
    with _conn() as con:
        row = con.execute("SELECT * FROM captions WHERE id = ?", (caption_id,)).fetchone()
    return dict(row) if row else None


def update_caption(caption_id: str, **kwargs: Any) -> None:
    """
    Update specific fields on a caption row.

    Only the fields listed in *_UPDATABLE_FIELDS* are accepted; all others
    are silently ignored to prevent SQL injection via column name injection.
    """
    _UPDATABLE_FIELDS = {"caption", "hashtags", "scheduled_date", "published"}
    updates = {k: v for k, v in kwargs.items() if k in _UPDATABLE_FIELDS}
    if not updates:
        return
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    with _conn() as con:
        con.execute(
            f"UPDATE captions SET {set_clause} WHERE id = ?",
            (*updates.values(), caption_id),
        )


def delete_caption(caption_id: str) -> None:
    """Delete a caption by ID."""
    with _conn() as con:
        con.execute("DELETE FROM captions WHERE id = ?", (caption_id,))


def load_captions(
    brand_id: str | None = None,
    published: bool | None = None,
) -> list[dict[str, Any]]:
    """
    Return captions ordered by scheduled date (then creation date).

    Args:
        brand_id:  Filter to a specific brand. None returns all brands.
        published: True = published only; False = unpublished only; None = all.
    """
    query  = "SELECT * FROM captions WHERE 1=1"
    params: list[Any] = []
    if brand_id:
        query += " AND brand_id = ?"
        params.append(brand_id)
    if published is not None:
        query += " AND published = ?"
        params.append(1 if published else 0)
    query += " ORDER BY COALESCE(scheduled_date, created_at) ASC"
    with _conn() as con:
        rows = con.execute(query, params).fetchall()
    return [dict(r) for r in rows]


# ── Performance ───────────────────────────────────────────────────────────────

def save_performance(
    caption_id: str,
    likes: int,
    comments: int,
    reach: int,
    saves: int,
) -> None:
    """Record engagement metrics for a published caption."""
    with _conn() as con:
        con.execute(
            "INSERT INTO performance (id, caption_id, likes, comments, reach, saves, recorded_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(uuid4()), caption_id, likes, comments, reach, saves, _now()),
        )
    logger.debug("Performance recorded for caption_id=%s", caption_id)


def load_performance() -> list[dict[str, Any]]:
    """Return all performance rows joined with their caption metadata."""
    with _conn() as con:
        rows = con.execute(
            """SELECT p.*, c.caption, c.platform, c.image_name, c.brand_name, c.brand_id
               FROM performance p
               JOIN captions c ON p.caption_id = c.id
               ORDER BY p.recorded_at DESC"""
        ).fetchall()
    return [dict(r) for r in rows]


def has_performance(caption_id: str) -> bool:
    """Return True if any performance data has been recorded for this caption."""
    with _conn() as con:
        row = con.execute(
            "SELECT 1 FROM performance WHERE caption_id = ? LIMIT 1", (caption_id,)
        ).fetchone()
    return row is not None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(UTC).isoformat()
