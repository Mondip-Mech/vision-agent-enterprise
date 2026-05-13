# 🎙️ Voxly — Brand Voice AI Caption Generator

[![CI](https://github.com/Mondip-Mech/vision-agent-enterprise/actions/workflows/ci.yml/badge.svg)](https://github.com/Mondip-Mech/vision-agent-enterprise/actions/workflows/ci.yml)
[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vision-agent-enterprise-sd4tvbstpw5zh6rzt3qeau.streamlit.app/)

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
 analyze_brand_voice()          ← vision model (text-only call)
        │                          Extracts: tone, style traits, emoji usage,
        │                          sentence length, signature words, CTA style
        ▼
  Brand Voice Profile (JSON)    ← Stored in SQLite, reused for every image
        │
        ├─→ embed_posts()        ← sentence-transformers (all-MiniLM-L6-v2)
        │       │                   Embeddings cached in SQLite — never re-encoded
        │       ▼
        │  embeddings_json       ← Persisted per brand; loaded on next Generate
        │
        ▼
generate_caption_variations()   ← vision model (image + text call)
        │   │
        │   ├─ retrieve_relevant_posts()  ← cosine similarity over cached embeddings
        │   │    Picks top-5 posts most relevant to image context; skips re-encoding
        │   │
        │   ├─ _sanitize_text()           ← strips control chars, truncates, removes
        │   │    Applied to every post    ← injection patterns before prompt injection
        │   │
        │   └─ _call_with_retry()         ← provider failover
        │        Tries primary → backup providers on 5xx / Timeout
        │        Fast-fails on 4xx (auth, quota exceeded)
        │
        ▼
  Scored Caption Variations     ← 3 variations × {FB caption, IG caption,
                                   hashtags, score 1-10, score_reason,
                                   image_analysis}
        │
        ▼
 eval_captions.evaluate_batch() ← independent judge model (different provider)
        │                          Avoids self-scoring bias; ranks variations
        ▼
  Re-ranked by unbiased score
```

**Key AI decisions:**
- **Few-shot prompting** — real brand posts are injected into every caption call, not just the profile. This keeps outputs grounded in actual brand language.
- **Semantic RAG retrieval** — when a brand has many posts, dense embeddings (cosine similarity) pick the most contextually relevant ones rather than truncating naively.
- **Embedding caching** — post vectors are computed once on brand save and persisted in SQLite. Generation calls load the cached array — no model re-run per image.
- **Single API call for 3 variations** — all variations + image analysis are generated in one call, minimising latency and cost.
- **Structured JSON output** — model is instructed to return a strict JSON schema; a regex-based fallback parser handles malformed responses gracefully.
- **Independent evaluation** — `eval_captions.py` routes scoring to a *different* provider than the generator, eliminating the self-scoring bias that inflates model self-assessment.

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

### Automatic provider failover

Set API keys for more than one provider and Voxly fails over automatically — no code changes required:

```env
NVIDIA_API_KEY=nvapi-...     # primary
OPENAI_API_KEY=sk-...        # backup if NVIDIA returns 5xx or times out
ANTHROPIC_API_KEY=sk-ant-... # tertiary
```

Failover policy:
- **Timeout / HTTP 5xx** → retry up to 3× with exponential backoff (1 s, 2 s, 4 s), then try next provider
- **HTTP 4xx** → raise immediately; no retry or failover (auth/quota errors are not transient)

The provider abstraction lives in `summarizer.py` — `_call_for_provider()` dispatches to `_call_openai_compat()` (shared by NVIDIA and OpenAI) or `_call_anthropic()`. All public APIs are identical regardless of provider.

---

## 🔍 Semantic RAG Retrieval

When a brand accumulates more example posts than the prompt budget allows, naive truncation (`posts[:5]`) discards potentially the most relevant ones. Voxly uses dense embeddings to retrieve the posts that are semantically closest to the current generation context.

```
Brand posts → embed_posts()         ← all-MiniLM-L6-v2 (22 MB, fully offline)
           → save_brand_embeddings() ← cached in SQLite as embeddings_json

Per image:
  query = brand_name + tone + context_hint
  → retrieve_relevant_posts(query, posts, cached_embeddings=...)
    → cosine similarity (normalised vectors, dot product)
    → top-5 posts by relevance injected into prompt
```

**Install the optional dependency:**
```bash
pip install -r requirements-rag.txt
```
Without it, the system falls back to the first 5 posts — all other functionality is unaffected.

---

## 🛡️ Prompt Injection Sanitization

User-supplied brand posts are injected directly into model prompts. `_sanitize_text()` applies three defences before any user text reaches the model:

1. **Control character stripping** — removes C0/C1 chars (keeps `\n` and `\t`)
2. **Hard truncation** — caps each post at 500 characters
3. **Injection pattern removal** — regex-replaces known prompt-hijacking strings:
   - `ignore previous instructions` / `ignore all above`
   - `you are now` / `new system prompt`
   - LLaMA/Mistral token delimiters: `[INST]`, `[SYS]`, `<|im_start|>`

---

## 🧪 Independent Caption Evaluation

The generation model self-scores its own output (1–10), which suffers from self-serving bias. `eval_captions.py` routes scoring to a *different* configured provider:

```python
from eval_captions import evaluate_batch

results = evaluate_batch(
    captions=["Caption A", "Caption B", "Caption C"],
    brand_profile=brand["profile"],
    image_description="A flat-lay of a cream wool sweater",
)
# Returns list sorted by unbiased score, each with score + reasoning + judge provider
```

**CLI:**
```bash
python eval_captions.py --caption "Just dropped our winter edit ❄️" --brand-name "Zara"
python eval_captions.py --file captions.json --brand-name "Nike" --tone "energetic, bold"
```

Judge selection logic: picks the highest-priority provider *other than* the current generator. Falls back to self-evaluation only if a single key is configured.

---

## 🌊 Streaming AI Insights

Analytics recommendations stream token-by-token via `st.write_stream()` — no waiting for a full response before seeing output.

```
analytics.get_ai_insights_stream(brand_profile)
    → stream_text(prompt)
        → _stream_openai_compat()  or  _stream_anthropic()
            → SSE parsing (iter_lines)
            → ConnectionError if stream yields 0 tokens  ← hardens against silent failures
            → fallback to batch _post_text() in stream_text()
```

---

## 🏗️ Project Structure

```
vision_agent_enterprise/
├── app.py              # Streamlit UI — 4 tabs: Brands, Generate, Calendar, Analytics
├── summarizer.py       # Core AI — brand voice extraction, caption generation,
│                       #   provider dispatch, failover, streaming, sanitization
├── database.py         # SQLite CRUD — brands, captions, performance, settings,
│                       #   embedding cache (embeddings_json)
├── rag.py              # Semantic post retrieval — sentence-transformers embeddings,
│                       #   cosine similarity, embedding caching helpers
├── eval_captions.py    # Independent caption judge — cross-provider scoring,
│                       #   batch evaluation, CLI
├── analytics.py        # Performance analytics + streaming AI insights
├── scheduler.py        # Buffer API wrapper — publish posts to social channels
├── config.py           # Settings loaded from environment variables
├── .env                # Local secrets (never committed)
├── .env.example        # Template for environment variables
├── requirements.txt         # Core dependencies
├── requirements-rag.txt     # Optional: sentence-transformers for semantic RAG
├── requirements-dev.txt     # Test/lint tooling
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
pip install -r requirements-rag.txt   # optional — enables semantic RAG
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
| `OPENAI_API_KEY` | — | Required when provider = `openai`; also used as failover |
| `ANTHROPIC_API_KEY` | — | Required when provider = `anthropic`; also used as failover |
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
- Post embeddings computed and cached on save — zero re-encoding cost on generate

### Bulk Caption Generation
- Upload multiple images in one batch
- 3 scored variations per image — pick the best one
- Per-variation score (1–10) with reasoning
- Image auto-analysis included in every response
- Caption length control: Short (<40 words) / Medium (40–80) / Long (80–150)
- Optional context hint per batch: e.g. "Winter jacket, price £89"
- Semantic RAG: most relevant brand posts selected per image (not just first 5)

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
- AI recommendations streamed token-by-token (requires ≥3 data points)

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
- User-supplied brand posts are sanitized before prompt injection (control chars, length cap, injection pattern removal)

---

## 🧪 Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit 1.37+ |
| AI | NVIDIA NIM · OpenAI · Anthropic (switchable via `CAPTION_PROVIDER`) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (22 MB, offline) |
| Data | SQLite (via Python `sqlite3`) |
| Image processing | Pillow |
| Publishing | Buffer API v1 |
| Config | `python-dotenv` |
| Data export | `pandas` |
| Tests | pytest + pytest-mock (142 tests) |
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
_build_variations_prompt()
    │   │
    │   ├─ _sanitize_text()    Strip control chars, cap 500 chars, remove injections
    │   └─ retrieve_relevant_posts()  RAG: cosine sim over cached embeddings
    │
    ▼
_post_vision_with_retry()      POST /chat/completions with image_url content block
    │   └─ _call_with_retry()  Retry + provider failover (5xx/Timeout → next provider)
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
