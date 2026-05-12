# 🎙️ Voxly — Brand Voice AI Caption Generator

[![CI](https://github.com/KaleshY/voxly/actions/workflows/ci.yml/badge.svg)](https://github.com/KaleshY/voxly/actions/workflows/ci.yml)

> Teach Voxly your brand's voice once. It writes on-brand captions for every image, at scale.

Voxly is a production-ready Streamlit application that combines **multimodal vision AI**, **few-shot brand voice extraction**, and **bulk caption generation** into a single workflow built for social media managers, e-commerce brands, and freelancers who produce content at scale.

---

## ✨ What It Does

| Step | Action |
|------|--------|
| 1 | Paste 3–5 real posts from your brand |
| 2 | AI extracts a structured voice profile (tone, style, emojis, CTAs, audience) |
| 3 | Upload any number of product images |
| 4 | Get 3 scored caption variations per image for Facebook + Instagram |
| 5 | Edit inline, schedule, save to calendar, or publish via Buffer |
| 6 | Export a scheduler-ready CSV with one row per platform per image |

---

## 🧠 AI Architecture

```
User Posts (3-5 examples)
        │
        ▼
 analyze_brand_voice()          ← NVIDIA vision model (text-only call)
        │                          Extracts: tone, style traits, emoji usage,
        │                          sentence length, signature words, CTA style
        ▼
  Brand Voice Profile (JSON)    ← Stored in SQLite, reused for every image
        │
        ▼
generate_caption_variations()   ← NVIDIA vision model (image + text call)
        │                          Input: image bytes + voice profile + example posts
        │                          Output: 3 variations × {FB caption, IG caption,
        │                                  hashtags, score 1-10, score_reason,
        │                                  image_analysis}
        ▼
  Scored Caption Variations     ← Sorted by score, editable in UI
```

**Key AI decisions:**
- **Few-shot prompting** — real brand posts are injected into every caption call, not just the profile. This keeps outputs grounded in actual brand language.
- **Single API call for 3 variations** — all variations + image analysis are generated in one call, minimising latency and cost.
- **Structured JSON output** — model is instructed to return a strict JSON schema; a regex-based fallback parser handles malformed responses gracefully.
- **Exponential backoff retry** — transient API timeouts and 5xx errors are retried up to 3 times before surfacing to the user.

---

## 🔀 Multi-Provider Support

Voxly is model-agnostic. Set `CAPTION_PROVIDER` in `.env` to switch AI backends without touching any code:

| `CAPTION_PROVIDER` | Model | Key variable | Notes |
|--------------------|-------|-------------|-------|
| `nvidia` *(default)* | `nvidia/llama-3.1-nemotron-nano-vl-8b-v1` | `NVIDIA_API_KEY` | Free credits at build.nvidia.com |
| `openai` | `gpt-4o` | `OPENAI_API_KEY` | Best quality; paid |
| `anthropic` | `claude-3-5-sonnet-20241022` | `ANTHROPIC_API_KEY` | Strong reasoning + vision |

Switch in one line:

```env
CAPTION_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
```

The provider abstraction lives entirely in `summarizer.py` — `_call()` dispatches to `_call_openai_compat()` (shared by NVIDIA and OpenAI, same wire format) or `_call_anthropic()` (message format conversion + response normalisation). All public APIs remain identical regardless of provider.

---

## 🏗️ Project Structure

```
vision_agent_enterprise/
├── app.py              # Streamlit UI — 4 tabs: Brands, Generate, Calendar, Analytics
├── summarizer.py       # Core AI logic — brand voice extraction + caption generation
├── database.py         # SQLite data layer — brands, captions, performance, settings
├── analytics.py        # Performance analytics + AI insights
├── scheduler.py        # Buffer API wrapper — publish posts to social channels
├── config.py           # Settings loaded from environment variables
├── .env                # Local secrets (never committed)
├── .env.example        # Template for environment variables
└── .streamlit/
    └── config.toml     # Streamlit theme (dark purple/blue)
```

---

## ⚙️ Setup

### 1. Clone & create environment

```powershell
git clone <repo-url>
cd vision_agent_enterprise
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure environment

```powershell
copy .env.example .env
```

Open `.env` and set your NVIDIA API key:

```env
NVIDIA_API_KEY=nvapi-your-key-here
```

Get a free key at [build.nvidia.com](https://build.nvidia.com/) → select any vision model → "Get API Key".

### 3. Run

```powershell
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## 🔑 Configuration Reference

All settings are read from environment variables (or `.env`). Only the API key for your chosen provider is required.

| Variable | Default | Description |
|----------|---------|-------------|
| `CAPTION_PROVIDER` | `nvidia` | AI backend — `nvidia` \| `openai` \| `anthropic` |
| `NVIDIA_API_KEY` | — | Required when provider = `nvidia` |
| `OPENAI_API_KEY` | — | Required when provider = `openai` |
| `ANTHROPIC_API_KEY` | — | Required when provider = `anthropic` |
| `NVIDIA_API_BASE_URL` | `https://integrate.api.nvidia.com/v1` | NVIDIA API endpoint |
| `NVIDIA_VISION_MODEL` | `nvidia/llama-3.1-nemotron-nano-vl-8b-v1` | NVIDIA vision model |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model name |
| `ANTHROPIC_MODEL` | `claude-3-5-sonnet-20241022` | Anthropic model name |
| `NVIDIA_MAX_IMAGE_BYTES` | `184320` (~180 KB) | Max compressed image size sent to API |
| `REQUEST_TIMEOUT_SECONDS` | `60` | Per-request API timeout |

---

## 📱 Features

### Brand Voice Memory
- Extracts 9 structured attributes from example posts: tone, personality, style traits, emoji usage, preferred emojis, sentence style, signature words, CTA style, audience
- Voice profiles persist in SQLite — teach it once, reuse forever
- Supports multiple brand profiles simultaneously (switch between brands)

### Bulk Caption Generation
- Upload multiple images in one batch
- 3 scored variations per image — pick the best one
- Per-variation score (1–10) with reasoning
- Image auto-analysis included in every response
- Caption length control: Short (<40 words) / Medium (40–80) / Long (80–150)
- Optional context hint per batch: e.g. "Winter jacket, price £89"

### Inline Editing & Scheduling
- Edit any caption in-place before saving
- Date + time picker per image
- Save to built-in content calendar

### Content Calendar
- View all saved posts filtered by brand, platform, or status (Draft / Scheduled / Published)
- Reschedule with a date picker
- Mark as published

### Analytics & AI Insights
- Log likes, comments, reach, saves per published post
- Bar chart of performance by post
- AI recommendations based on top vs. bottom performers (requires ≥3 data points)

### Buffer Integration (optional)
- Add your Buffer access token in the sidebar
- Publish directly to any connected Buffer profile
- Supports scheduled publishing

---

## 🛡️ Security Notes

- **Never commit `.env`** — it is listed in `.gitignore`
- The `NVIDIA_API_KEY` is a paid credential; rotate it immediately if accidentally exposed
- All user data is stored locally in `voxly.db` (also `.gitignore`'d)
- Buffer tokens are stored in the local SQLite database, never sent anywhere except to Buffer's API

---

## 🧪 Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit 1.37+ |
| AI | NVIDIA NIM · OpenAI · Anthropic (switchable via `CAPTION_PROVIDER`) |
| Data | SQLite (via Python `sqlite3`) |
| Image processing | Pillow |
| Publishing | Buffer API v1 |
| Config | `python-dotenv` |
| Data export | `pandas` |
| Tests | pytest + pytest-mock (92 tests) |
| CI | GitHub Actions — lint (ruff) · type check (mypy) · tests |

---

## 🗺️ Data Flow

```
Image Upload (bytes)
    │
    ├─→ _open_image()          Decode + convert to RGB
    │       │
    │       ▼
    │   _compress_image()      Thumbnail 1024×1024, JPEG quality ladder
    │       │                  (85→75→65→55→45 until ≤ NVIDIA_MAX_IMAGE_BYTES)
    │       ▼
    │   Base64 encode
    │       │
    ▼       ▼
_post_vision()                 POST /chat/completions with image_url content block
    │
    ▼
_get_content()                 Extract text from choices[0].message.content
    │
    ▼
_parse_json()                  Try raw text → markdown block → regex fallback
    │
    ▼
_parse_variations()            Validate and normalise variation dicts
    │
    ▼
generate_caption_variations()  Returns {image_analysis, variations[], error}
```
