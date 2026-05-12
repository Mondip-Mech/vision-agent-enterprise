# Voxly — Codebase Guide

Voxly is a Streamlit app that extracts a brand's social media voice from example posts and uses it to generate bulk caption variations for product images via NVIDIA's vision API.

## Architecture

Single-process Streamlit app. No backend server, no task queue. All I/O is synchronous.

```
app.py          → UI (4 tabs)
summarizer.py   → NVIDIA API calls + prompt logic
database.py     → SQLite CRUD
analytics.py    → AI insights
scheduler.py    → Buffer API
config.py       → Settings from env vars
```

## Key Data Flows

### 1. Brand voice extraction
```
User pastes 3-5 posts
→ summarizer.analyze_brand_voice()
  → _post_text() [NVIDIA text-only call]
  → _parse_json() extracts structured profile
→ database.save_brand() stores to SQLite
→ st.session_state.active_brand set
```

### 2. Caption generation (per image)
```
User uploads image(s) + clicks Generate
→ summarizer.generate_caption_variations()
  → _open_image() + _compress_image()     [PIL]
  → _build_variations_prompt()            [injects profile + examples]
  → _post_vision_with_retry()             [NVIDIA vision call, up to 3 retries]
  → _get_content() + _parse_json()
  → _parse_variations() normalises output
→ Returns {image_analysis, variations[], error}
→ Results stored in st.session_state.results
```

### 3. Calendar / analytics save
```
User clicks "Save to Calendar"
→ database.save_caption()    [SQLite insert]
User logs performance
→ database.save_performance() [SQLite insert]
User clicks "Get AI Recommendations"
→ analytics.get_ai_insights()
  → database.load_performance()
  → summarizer._post_text()  [NVIDIA text-only call]
```

## Session State Keys

| Key | Type | Description |
|-----|------|-------------|
| `active_brand` | `dict \| None` | Currently selected brand (full record from DB) |
| `results` | `list[dict] \| None` | Latest generation results |
| `show_brand_form` | `bool` | Whether add/edit brand form is visible |
| `editing_brand_id` | `str \| None` | Brand ID being edited |
| `buffer_profiles` | `list \| None` | Cached Buffer API profiles |

## NVIDIA API Details

- **Endpoint:** `POST /chat/completions` (OpenAI-compatible)
- **Model:** `nvidia/llama-3.1-nemotron-nano-vl-8b-v1` (configurable via `NVIDIA_VISION_MODEL`)
- **Vision input:** Image sent as `data:image/jpeg;base64,...` in the `image_url` content block
- **Text-only:** Same endpoint without image content block
- **Retry policy:** 3 attempts, exponential backoff (1 s, 2 s, 4 s); retries on `Timeout` and HTTP 5xx only

## Database Schema

```sql
brands      (id, name, platform, profile_json, posts_json, created_at, updated_at)
captions    (id, brand_id, brand_name, image_name, platform, caption, hashtags,
             score, context, length_pref, scheduled_date, published, created_at)
performance (id, caption_id, likes, comments, reach, saves, recorded_at)
settings    (key, value)
```

`profile_json` and `posts_json` are JSON-encoded Python dicts/lists.
`_hydrate_brand()` deserialises them; errors are caught and logged rather than raised.

## Image Compression Pipeline

Images are compressed before sending to NVIDIA to stay within `NVIDIA_MAX_IMAGE_BYTES` (default ~180 KB):

1. Thumbnail to 1024×1024 (preserves aspect ratio)
2. Save as JPEG at quality 85, 75, 65, 55, 45 — stop when size fits
3. If still too large: thumbnail to 768×768 at quality 40

## Environment Variables

See `config.py` for the full list. Only `NVIDIA_API_KEY` is required.
Copy `.env.example` → `.env` and fill in your key.

## Common Tasks

**Add a new setting:**
Edit `config.py` — add a field to `Settings` dataclass with `os.getenv("VAR", default)`.

**Change the AI model:**
Set `NVIDIA_VISION_MODEL` in `.env` to any NVIDIA NIM vision model name.

**Extend the database schema:**
Edit `init_db()` in `database.py`. For existing deployments, run the ALTER TABLE manually (no migration system).

**Change prompt behaviour:**
All prompts are in `summarizer.py`. `_build_variations_prompt()` constructs the main caption prompt. `_LENGTH_NOTES` and `_HASHTAG_COUNT` are the key constants to adjust.
