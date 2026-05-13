# Voxly — Codebase Guide

Voxly is a Streamlit app that extracts a brand's social media voice from example posts and uses it to generate bulk caption variations for product images via multimodal AI (NVIDIA / OpenAI / Anthropic).

## Architecture

Single-process Streamlit app. No backend server, no task queue. All I/O is synchronous.

```
app.py             → UI (4 tabs) + RAG model preload via @st.cache_resource
summarizer.py      → AI calls, prompt logic, provider dispatch, failover, streaming, sanitization
database.py        → SQLite CRUD + embedding cache
rag.py             → Semantic post retrieval (sentence-transformers, cosine similarity)
eval_captions.py   → Independent caption judge (cross-provider scoring, CLI)
analytics.py       → Performance analytics + streaming AI insights
scheduler.py       → Buffer API wrapper
config.py          → Settings from env vars
```

## Key Data Flows

### 1. Brand voice extraction + embedding cache
```
User pastes 3-5 posts
→ summarizer.analyze_brand_voice()
  → _sanitize_text() on each post  [control chars, truncation, injection patterns]
  → _post_text() [text-only API call]
  → _parse_json() extracts structured profile
→ database.save_brand() stores to SQLite
→ rag.embed_posts(posts)            [all-MiniLM-L6-v2, optional]
→ database.save_brand_embeddings()  [persist vectors as JSON in brands.embeddings_json]
→ st.session_state.active_brand set
```

### 2. Caption generation (per image)
```
User uploads image(s) + clicks Generate
→ database.get_brand_embeddings()   [load cached vectors — no re-encoding]
→ summarizer.generate_caption_variations()
  → _open_image() + _compress_image()              [PIL]
  → _build_variations_prompt()
      → _sanitize_text() on all posts              [defence-in-depth]
      → rag.retrieve_relevant_posts(               [cosine sim, top-5]
            query=brand+tone+context,
            cached_embeddings=cached_emb)
      → injects selected posts + voice profile into prompt
  → _post_vision_with_retry()                      [image + text API call]
      → _call_with_retry()
          → _failover_chain()                      [primary + other keyed providers]
          → per-provider: up to 3 retries (exp backoff) on 5xx / Timeout
          → 4xx → raise immediately, no failover
  → _get_content() + _parse_json()
  → _parse_variations() normalises output
→ Returns {image_analysis, variations[], error}
→ Results stored in st.session_state.results
```

### 3. Independent evaluation (eval_captions.py)
```
captions = [v["facebook_caption"] for v in result["variations"]]
→ eval_captions.evaluate_batch(captions, brand_profile, image_description)
  → judge_provider()              [picks provider ≠ generator; falls back if single key]
  → score_caption() per caption   [structured eval prompt → score 1-10 + reasoning]
→ Returns captions sorted by unbiased score
```

### 4. Calendar / analytics save
```
User clicks "Save to Calendar"
→ database.save_caption()    [SQLite insert]
User logs performance
→ database.save_performance() [SQLite insert]
User clicks "Get AI Recommendations"
→ analytics.get_ai_insights_stream(brand_profile)
  → database.load_performance()
  → summarizer.stream_text()   [SSE streaming, token-by-token]
      → _stream_openai_compat() or _stream_anthropic()
      → raises ConnectionError if 0 tokens received (triggers batch fallback)
→ st.write_stream() renders tokens as they arrive
```

## Session State Keys

| Key | Type | Description |
|-----|------|-------------|
| `active_brand` | `dict \| None` | Currently selected brand (full record from DB) |
| `results` | `list[dict] \| None` | Latest generation results |
| `show_brand_form` | `bool` | Whether add/edit brand form is visible |
| `editing_brand_id` | `str \| None` | Brand ID being edited |
| `buffer_profiles` | `list \| None` | Cached Buffer API profiles |
| `ai_insights` | `str \| None` | Last streamed AI insights text |

## Provider Failover

`_failover_chain()` in `summarizer.py`:
- Returns `[primary] + [other providers with API keys configured]`
- Only providers with keys are included — single-key setups have no failover
- `_call_with_retry()` iterates the chain: 3 retries per provider (backoff 1/2/4 s) on 5xx/Timeout, then moves to next. 4xx always raises immediately.

Adding a new provider:
1. Add API key + model name to `config.py`
2. Add a branch in `_call_for_provider()` calling your wire-format function
3. Add to `_PROVIDER_PRIORITY` tuple in both `summarizer.py` and `eval_captions.py`
4. Add key check in `_has_key()` and `_require_api_key()`

## Database Schema

```sql
brands      (id, name, platform, profile_json, posts_json, created_at, updated_at,
             embeddings_json)   -- JSON array of float vectors, one per example post
captions    (id, brand_id, brand_name, image_name, platform, caption, hashtags,
             score, context, length_pref, scheduled_date, published, created_at)
performance (id, caption_id, likes, comments, reach, saves, recorded_at)
settings    (key, value)
```

`profile_json`, `posts_json`, and `embeddings_json` are JSON-encoded.
`_hydrate_brand()` deserialises the first two; errors are caught and logged rather than raised.
`embeddings_json` is written by `save_brand_embeddings()` and read by `get_brand_embeddings()`.

Migration: `init_db()` attempts `ALTER TABLE brands ADD COLUMN embeddings_json TEXT` after every startup; the exception is silently swallowed on subsequent runs (column already exists).

## Prompt Injection Sanitization

`_sanitize_text(text)` in `summarizer.py` — applied to every user-supplied post before it reaches a model prompt:
1. Strip C0/C1 control characters (keep `\n`, `\t`)
2. Hard-truncate to `_MAX_POST_CHARS` (500)
3. Regex-replace injection patterns (`_INJECT_RE`): `ignore previous instructions`, `you are now`, `new system prompt`, `[INST]`, `[SYS]`, `<|...|>` token delimiters

## Image Compression Pipeline

Images are compressed before sending to NVIDIA to stay within `NVIDIA_MAX_IMAGE_BYTES` (default ~180 KB):

1. Thumbnail to 1024×1024 (preserves aspect ratio)
2. Save as JPEG at quality 85, 75, 65, 55, 45 — stop when size fits
3. If still too large: thumbnail to 768×768 at quality 40

## RAG Model

`all-MiniLM-L6-v2` (22 MB, sentence-transformers). Downloaded on first call, cached in-process via `_get_model()` singleton. Preloaded at Streamlit startup via `@st.cache_resource _preload_rag_model()` in `app.py`.

Optional dependency (`requirements-rag.txt`). Falls back to `posts[:top_k]` if not installed.

## Environment Variables

See `config.py` for the full list. Only the API key for your chosen provider is required.
Copy `.env.example` → `.env` and fill in your key.

## Common Tasks

**Add a new setting:**
Edit `config.py` — add a field to `Settings` dataclass with `os.getenv("VAR", default)`.

**Change the AI model:**
Set `NVIDIA_VISION_MODEL` in `.env` to any NVIDIA NIM vision model name.

**Extend the database schema:**
Edit `init_db()` in `database.py`. For existing deployments, add the ALTER TABLE in the migration block after `executescript()`.

**Change prompt behaviour:**
All prompts are in `summarizer.py`. `_build_variations_prompt()` constructs the main caption prompt. `_LENGTH_NOTES` and `_HASHTAG_COUNT` are the key constants to adjust.

**Run the eval CLI:**
```bash
python eval_captions.py --caption "Your caption" --brand-name "Nike"
python eval_captions.py --file captions.json --brand-name "Zara" --image "cream sweater flat-lay"
```
