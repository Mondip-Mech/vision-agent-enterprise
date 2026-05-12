"""
Caption evaluation framework for Voxly.

Motivation
----------
The caption generation model self-scores its own output (1–10).  That score
suffers from self-serving bias: the same model that wrote the caption also
judges it, so it consistently overestimates quality.

This module provides an independent evaluation layer that routes scoring to a
*different* provider than the one that generated the caption.  A cross-model
judge gives an unbiased signal for:
  - A/B testing captions before publishing
  - Identifying which provider produces the best brand-voice match
  - Surfacing low-quality outputs that slipped past the generation score

Public API
----------
score_caption(caption, brand_profile, image_description="")
    Ask a judge model to rate a single caption and return a score + reasoning.

evaluate_batch(captions, brand_profile, image_description="")
    Score a list of captions in one batch call; returns them sorted by score.

judge_provider()
    Return the provider being used for evaluation (for observability).

CLI
---
    python eval_captions.py --caption "Your caption here" --brand-name "Nike"

or pipe a JSON file:
    python eval_captions.py --file captions.json
"""
from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import summarizer as _summarizer
from config import settings

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Priority order when choosing a judge — we deliberately pick a *different*
# provider from the generator to avoid self-scoring bias.
_PROVIDER_PRIORITY: tuple[str, ...] = ("anthropic", "openai", "nvidia")

# Evaluation prompt template.
_EVAL_PROMPT = """\
You are an independent social media content evaluator. Your job is to rate \
how well a caption matches a brand's voice profile, NOT to rewrite it.

BRAND VOICE PROFILE:
  Name:        {brand_name}
  Tone:        {tone}
  Personality: {personality}
  Style:       {traits}
  Emoji usage: {emoji_usage}
  Audience:    {audience}
  CTA style:   {cta_style}

{image_block}\
CAPTION TO EVALUATE:
"{caption}"

Rate this caption on a scale of 1.0–10.0 for brand-voice match and audience \
appeal. Be critical and unbiased — do not inflate scores.

Return ONLY a JSON object — no markdown, no extra text:
{{
  "score": <number between 1.0 and 10.0>,
  "reasoning": "<one sentence explaining the score>"
}}"""


# ── Judge selection ───────────────────────────────────────────────────────────

def judge_provider() -> str:
    """
    Return the provider used for evaluation.

    Selection logic:
        1. Prefer any provider *other* than the current generator.
        2. If only one provider has a key, fall back to it (self-evaluation
           is unavoidable in single-key setups, but the separate prompt still
           adds structure).
    """
    generator = settings.caption_provider

    def _has_key(p: str) -> bool:
        if p == "nvidia":
            return bool(settings.nvidia_api_key)
        if p == "openai":
            return bool(settings.openai_api_key)
        if p == "anthropic":
            return bool(settings.anthropic_api_key)
        return False

    # Try to find a configured provider that is different from the generator
    for p in _PROVIDER_PRIORITY:
        if p != generator and _has_key(p):
            return p

    # Fallback: use the generator itself (single-key setup)
    return generator


# ── Core evaluation ───────────────────────────────────────────────────────────

def score_caption(
    caption: str,
    brand_profile: dict[str, Any],
    image_description: str = "",
) -> dict[str, Any]:
    """
    Ask an independent judge model to score a single caption.

    Uses a provider different from the configured generator whenever possible
    to avoid self-scoring bias.  See ``judge_provider()`` for selection logic.

    Args:
        caption:           The caption text to evaluate.
        brand_profile:     Voice profile dict from analyze_brand_voice().
        image_description: Optional 1–2 sentence description of the image
                           (improves relevance scoring when provided).

    Returns:
        {
            "caption":    str,   # the original caption
            "score":      float, # 1.0–10.0
            "reasoning":  str,   # one-sentence explanation
            "judge":      str,   # which provider scored it
            "error":      str | None,
        }
    """
    provider = judge_provider()
    image_block = (
        f"IMAGE DESCRIPTION:\n{image_description}\n\n"
        if image_description.strip() else ""
    )
    prompt = _EVAL_PROMPT.format(
        brand_name  = brand_profile.get("brand_name", "this brand"),
        tone        = brand_profile.get("tone", "—"),
        personality = brand_profile.get("personality", "—"),
        traits      = ", ".join(brand_profile.get("style_traits") or []),
        emoji_usage = brand_profile.get("emoji_usage", "—"),
        audience    = brand_profile.get("audience", "—"),
        cta_style   = brand_profile.get("cta_style", "—"),
        image_block = image_block,
        caption     = caption,
    )

    messages = [{"role": "user", "content": prompt}]

    try:
        data      = _summarizer._call_for_provider(provider, messages, max_tokens=200, temperature=0.1)
        text      = _summarizer._get_content(data)
        parsed    = _summarizer._parse_json(text) or {}
        score     = float(parsed.get("score") or 5.0)
        score     = max(1.0, min(10.0, score))   # clamp to valid range
        reasoning = str(parsed.get("reasoning") or "").strip()

        logger.info(
            "Eval: caption scored %.1f/10 by %s judge", score, provider
        )
        return {
            "caption":   caption,
            "score":     score,
            "reasoning": reasoning,
            "judge":     provider,
            "error":     None,
        }

    except Exception as exc:
        logger.warning("Eval failed (%s) — returning default score", exc)
        return {
            "caption":   caption,
            "score":     5.0,
            "reasoning": "Evaluation failed",
            "judge":     provider,
            "error":     str(exc),
        }


def evaluate_batch(
    captions: list[str],
    brand_profile: dict[str, Any],
    image_description: str = "",
) -> list[dict[str, Any]]:
    """
    Score a list of captions and return them sorted by score (highest first).

    Each caption is scored independently by the same judge provider.  Scores
    are comparable within a batch, making this suitable for A/B ranking.

    Args:
        captions:          List of caption strings to compare.
        brand_profile:     Voice profile dict from analyze_brand_voice().
        image_description: Shared image context for all captions in the batch.

    Returns:
        List of score dicts (same shape as score_caption()) sorted by
        descending score.
    """
    if not captions:
        return []

    logger.info(
        "Evaluating batch of %d caption(s) with '%s' judge",
        len(captions), judge_provider(),
    )

    results = [
        score_caption(caption, brand_profile, image_description)
        for caption in captions
    ]
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ── CLI interface ─────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Voxly captions with an independent judge model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
# Score a single caption:
  python eval_captions.py --caption "Just dropped our new winter jacket ❄️" --brand-name "Zara"

# Score captions from a JSON file (list of strings):
  python eval_captions.py --file captions.json --brand-name "Nike" --tone "energetic, bold"

# Include an image description for better relevance scoring:
  python eval_captions.py --caption "Shop now →" --brand-name "H&M" \\
      --image "A flat-lay of a cream wool sweater on a marble surface"
""",
    )
    parser.add_argument("--caption",    help="Single caption string to evaluate")
    parser.add_argument("--file",       help="JSON file — list of caption strings")
    parser.add_argument("--brand-name", default="this brand", help="Brand name")
    parser.add_argument("--tone",       default="",           help="Brand tone (optional)")
    parser.add_argument("--audience",   default="",           help="Target audience (optional)")
    parser.add_argument("--image",      default="",           help="Image description (optional)")
    return parser


def _minimal_profile(brand_name: str, tone: str, audience: str) -> dict[str, Any]:
    """Build a minimal brand profile from CLI args."""
    return {
        "brand_name": brand_name,
        "tone":       tone,
        "personality": "",
        "style_traits": [],
        "emoji_usage": "",
        "audience":   audience,
        "cta_style":  "",
    }


def main() -> None:
    """CLI entry point."""
    parser = _build_arg_parser()
    args   = parser.parse_args()

    if not args.caption and not args.file:
        parser.error("Provide --caption or --file.")

    profile = _minimal_profile(args.brand_name, args.tone, args.audience)

    if args.file:
        with open(args.file) as fh:
            captions = json.load(fh)
        if not isinstance(captions, list):
            parser.error("--file must contain a JSON array of strings.")
        results = evaluate_batch(captions, profile, image_description=args.image)
    else:
        results = [score_caption(args.caption, profile, image_description=args.image)]

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
