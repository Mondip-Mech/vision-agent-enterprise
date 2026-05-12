"""
Performance analytics and AI-powered insights for Voxly.

Functions
---------
get_ai_insights(brand_profile)
    Sends the top and bottom performing posts to the NVIDIA API and returns
    3 specific, actionable recommendations as plain text.

summary_stats(rows)
    Returns aggregate engagement stats (total posts, avg likes, avg reach,
    best post) from a list of performance rows.
"""
from __future__ import annotations

import logging
from typing import Any

import database as db
from summarizer import _get_content, _post_text

logger = logging.getLogger(__name__)

# Minimum rows required before AI analysis is meaningful.
_MIN_ROWS_FOR_INSIGHTS: int = 3
# How many top / bottom posts to include in the insight prompt.
_INSIGHT_SAMPLE_SIZE: int = 3


def get_ai_insights(brand_profile: dict[str, Any]) -> str:
    """
    Generate AI-powered caption improvement recommendations.

    Loads all performance data, selects the top and bottom
    _INSIGHT_SAMPLE_SIZE posts by likes, and sends them to the NVIDIA
    text API with a structured analysis prompt.

    Args:
        brand_profile: Voice profile dict from analyze_brand_voice().

    Returns:
        A numbered list of 3 recommendations as a plain-text string.

    Raises:
        ValueError: If fewer than _MIN_ROWS_FOR_INSIGHTS data points exist,
                    or if the API call fails.
    """
    rows = db.load_performance()

    if len(rows) < _MIN_ROWS_FOR_INSIGHTS:
        return (
            f"Add performance data for at least {_MIN_ROWS_FOR_INSIGHTS} posts "
            "to unlock AI insights."
        )

    rows_by_likes = sorted(rows, key=lambda r: r.get("likes", 0), reverse=True)

    # Guard against overlap when there are fewer rows than 2 × sample size.
    half       = max(1, len(rows_by_likes) // 2)
    sample     = min(_INSIGHT_SAMPLE_SIZE, half)
    top        = rows_by_likes[:sample]
    bottom     = rows_by_likes[-sample:]

    def _fmt(r: dict[str, Any]) -> str:
        # Truncate to ~140 chars so the prompt stays compact.
        preview = r["caption"][:140].replace("\n", " ")
        return (
            f'  Platform: {r["platform"]} | Image: {r["image_name"]}\n'
            f'  Caption: "{preview}"\n'
            f'  Likes: {r["likes"]} | Comments: {r["comments"]}'
            f' | Reach: {r["reach"]} | Saves: {r["saves"]}'
        )

    top_block    = "\n\n".join(_fmt(r) for r in top)
    bottom_block = "\n\n".join(_fmt(r) for r in bottom)
    brand_name   = brand_profile.get("brand_name", "this brand")

    logger.info(
        "Requesting AI insights for '%s' — %d top, %d bottom posts",
        brand_name, len(top), len(bottom),
    )

    prompt = (
        f"You are a social media analyst for '{brand_name}'.\n\n"
        f"TOP PERFORMING POSTS (highest likes):\n{top_block}\n\n"
        f"LOWEST PERFORMING POSTS:\n{bottom_block}\n\n"
        "Based on these real results, give exactly 3 specific, actionable "
        "recommendations to improve future captions. Reference the actual language, "
        "structure, or topics that worked vs. didn't. "
        "Format as a numbered list. Each point: max 2 sentences."
    )

    data = _post_text(prompt, max_tokens=450, temperature=0.3)
    result = _get_content(data)
    logger.info("AI insights generated successfully")
    return result


def summary_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute aggregate engagement statistics from a list of performance rows.

    Args:
        rows: Performance dicts as returned by database.load_performance().

    Returns:
        {
            "total_posts": int,
            "avg_likes":   float,
            "avg_reach":   float,
            "best_post":   dict | None,   # row with highest likes
        }
    """
    if not rows:
        return {"total_posts": 0, "avg_likes": 0.0, "avg_reach": 0.0, "best_post": None}

    total   = len(rows)
    avg_l   = round(sum(r.get("likes", 0) for r in rows) / total, 1)
    avg_r   = round(sum(r.get("reach", 0) for r in rows) / total, 1)
    best    = max(rows, key=lambda r: r.get("likes", 0))

    return {"total_posts": total, "avg_likes": avg_l, "avg_reach": avg_r, "best_post": best}
