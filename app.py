from __future__ import annotations

import io
from datetime import date, datetime, timezone

import pandas as pd
import streamlit as st

import analytics as analytics_store
import database as db
import scheduler as buf
from summarizer import analyze_brand_voice, generate_caption_variations

# ── Bootstrap ─────────────────────────────────────────────────────────────────
db.init_db()

st.set_page_config(
    page_title="Voxly",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  #MainMenu, footer { visibility: hidden; }
  .block-container { padding-top: 1.5rem; max-width: 860px; }

  .vx-logo {
    font-size: 2.4rem; font-weight: 900; letter-spacing: -1px;
    background: linear-gradient(135deg, #7C3AED 0%, #2563EB 60%, #06B6D4 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  }
  .vx-tagline { color: #64748B; font-size: 0.88rem; }

  .step-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: #1E1B4B; color: #A5B4FC; border: 1px solid #3730A3;
    border-radius: 20px; padding: 0.25rem 0.85rem;
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 0.6rem;
  }

  .brand-card {
    background: #0F172A; border: 1px solid #1E293B;
    border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 0.5rem;
  }
  .brand-card-active {
    background: linear-gradient(135deg, #0F172A, #1E1B4B); border-color: #4F46E5;
  }
  .brand-name { font-weight: 800; font-size: 1rem; color: #E2E8F0; }
  .brand-meta { font-size: 0.78rem; color: #64748B; margin-top: 0.15rem; }

  .pill { display: inline-block; padding: 0.1rem 0.55rem; border-radius: 20px; font-size: 0.7rem; font-weight: 700; }
  .pill-ig   { background: #4A1D5F; color: #E879F9; }
  .pill-fb   { background: #1E3A5F; color: #60A5FA; }
  .pill-both { background: #1E3A5F; color: #A5B4FC; }
  .pill-active { background: #052e16; color: #34d399; margin-left: 0.3rem; }

  .chip-row { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.45rem; }
  .chip-blue { background: #1E3A5F; color: #60A5FA; padding: 0.15rem 0.5rem; border-radius: 20px; font-size: 0.72rem; }

  .voice-card {
    background: linear-gradient(135deg, #0F172A, #1E1B4B);
    border: 1px solid #3730A3; border-radius: 14px; padding: 1.1rem 1.4rem; margin-bottom: 0.9rem;
  }
  .voice-card-title { font-weight: 800; font-size: 0.95rem; color: #A5B4FC; margin-bottom: 0.5rem; }
  .voice-meta { font-size: 0.8rem; color: #94A3B8; margin: 0.18rem 0; }
  .voice-meta b { color: #C4B5FD; }

  .score-hi  { color: #34D399; font-weight: 800; }
  .score-mid { color: #FBBF24; font-weight: 800; }
  .score-lo  { color: #F87171; font-weight: 800; }

  .section-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #7C3AED; margin-bottom: 0.3rem;
  }
  .char-hint { font-size: 0.68rem; color: #475569; text-align: right; margin-top: -0.35rem; }
  .result-name { font-weight: 700; font-size: 0.95rem; color: #E2E8F0; margin-bottom: 0.6rem; }
  .analysis-box {
    background: #0F172A; border-left: 3px solid #4F46E5;
    padding: 0.5rem 0.8rem; border-radius: 0 8px 8px 0; margin-bottom: 0.75rem;
    font-size: 0.82rem; color: #94A3B8; font-style: italic;
  }
  .hashtag-row { display: flex; flex-wrap: wrap; gap: 0.3rem; margin: 0.35rem 0; }
  .hashtag { background: #2D1B69; color: #A78BFA; padding: 0.18rem 0.6rem; border-radius: 20px; font-size: 0.78rem; }

  .cal-card {
    background: #0F172A; border: 1px solid #1E293B; border-radius: 10px;
    padding: 0.8rem 1rem; margin-bottom: 0.5rem;
  }
  .status-draft     { color: #94A3B8; }
  .status-scheduled { color: #60A5FA; }
  .status-published { color: #34D399; }

  .stat-box {
    background: #0F172A; border: 1px solid #1E293B; border-radius: 10px;
    padding: 0.9rem; text-align: center;
  }
  .stat-val   { font-size: 1.6rem; font-weight: 900; color: #A5B4FC; }
  .stat-label { font-size: 0.72rem; color: #64748B; margin-top: 0.2rem; }

  .empty-state { text-align: center; color: #475569; padding: 1.5rem 0; font-size: 0.88rem; }
  .insight-box {
    background: linear-gradient(135deg, #0F172A, #1E1B4B);
    border: 1px solid #3730A3; border-radius: 12px; padding: 1.1rem 1.3rem;
    font-size: 0.87rem; color: #C4B5FD; line-height: 1.7;
  }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("active_brand", None), ("results", None), ("show_brand_form", False),
              ("editing_brand_id", None), ("buffer_profiles", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

_PLATFORM_PILL = {
    "instagram": '<span class="pill pill-ig">📸 Instagram</span>',
    "facebook":  '<span class="pill pill-fb">📘 Facebook</span>',
    "both":      '<span class="pill pill-both">📱 Both</span>',
}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 0.5rem 0 0.25rem;">
  <div class="vx-logo">🎙️ Voxly</div>
  <div class="vx-tagline">Your brand's voice. Every image. At scale.</div>
</div>
""", unsafe_allow_html=True)
st.divider()

# ── Sidebar — Buffer integration ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("**Buffer Integration**")
    st.caption(
        "Paste your Buffer access token to publish directly. "
        "Get it at [buffer.com/developers/apps](https://buffer.com/developers/apps)."
    )
    saved_token = db.get_setting("buffer_token")
    buf_token_input = st.text_input(
        "Access Token", value=saved_token, type="password",
        label_visibility="collapsed", placeholder="Paste Buffer access token…",
    )
    if st.button("Save Token", use_container_width=True):
        db.set_setting("buffer_token", buf_token_input.strip())
        st.session_state.buffer_profiles = None
        st.success("Saved!")

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_brands, tab_gen, tab_cal, tab_analytics = st.tabs([
    "🎙️ My Brands",
    "✨ Generate",
    "📅 Calendar",
    "📊 Analytics",
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — MY BRANDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_brands:
    all_brands = db.load_brands()
    active     = st.session_state.active_brand

    if not all_brands:
        st.markdown('<div class="empty-state">No brands yet. Create your first brand below ↓</div>',
                    unsafe_allow_html=True)
    else:
        for b in all_brands:
            is_active = active and active["id"] == b["id"]
            card_cls  = "brand-card brand-card-active" if is_active else "brand-card"
            pill      = _PLATFORM_PILL.get(b["platform"], "")
            a_pill    = '<span class="pill pill-active">✓ Active</span>' if is_active else ""
            traits_h  = "".join(
                f'<span class="chip-blue">{t}</span>'
                for t in (b["profile"].get("style_traits") or [])[:3]
            )

            st.markdown(f"""
            <div class="{card_cls}">
              <div class="brand-name">{b["name"]} {pill} {a_pill}</div>
              <div class="brand-meta">{b["profile"].get("tone","")} · {b["profile"].get("personality","")}</div>
              <div class="chip-row">{traits_h}</div>
            </div>
            """, unsafe_allow_html=True)

            ca, cb, cc = st.columns([2, 1, 1])
            with ca:
                if not is_active:
                    if st.button("✓ Use This Brand", key=f"use_{b['id']}", use_container_width=True):
                        st.session_state.active_brand = b
                        st.session_state.results = None
                        st.rerun()
                else:
                    st.button("✓ Active", key=f"act_{b['id']}", disabled=True, use_container_width=True)
            with cb:
                if st.button("✏️ Edit", key=f"edit_{b['id']}", use_container_width=True):
                    st.session_state.show_brand_form = True
                    st.session_state.editing_brand_id = b["id"]
                    st.rerun()
            with cc:
                if st.button("🗑️ Delete", key=f"del_{b['id']}", use_container_width=True):
                    db.delete_brand(b["id"])
                    if active and active["id"] == b["id"]:
                        st.session_state.active_brand = None
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("＋ Add New Brand", use_container_width=True):
        st.session_state.show_brand_form = True
        st.session_state.editing_brand_id = None
        st.rerun()

    # ── Add / Edit form ────────────────────────────────────────────────────
    if st.session_state.show_brand_form:
        st.divider()
        editing_id    = st.session_state.editing_brand_id
        editing_brand = db.get_brand(editing_id) if editing_id else None
        form_title    = f"✏️ Edit — {editing_brand['name']}" if editing_brand else "＋ Add New Brand"
        st.markdown(f'<div class="step-badge">{form_title}</div>', unsafe_allow_html=True)

        with st.container(border=True):
            f1, f2 = st.columns([2, 1])
            with f1:
                brand_name = st.text_input(
                    "Brand / Account Name",
                    value=editing_brand["name"] if editing_brand else "",
                    placeholder="e.g. Zara, Nike, My Coffee Shop",
                )
            with f2:
                p_opts    = ["both", "instagram", "facebook"]
                p_default = p_opts.index(editing_brand["platform"]) if editing_brand else 0
                platform  = st.selectbox(
                    "Platform", p_opts, index=p_default,
                    format_func={"both": "📱 Both", "instagram": "📸 Instagram", "facebook": "📘 Facebook"}.get,
                )

            st.markdown("**Paste 3–5 real posts** *(separate each with `---`)*")
            default_posts = "---".join(editing_brand["example_posts"]) if editing_brand else ""
            posts_raw = st.text_area(
                "posts_input", value=default_posts, height=180,
                placeholder="Your first post here...\n---\nYour second post...\n---\nYour third post...",
                label_visibility="collapsed",
            )

            btn_label = "🔄 Update Brand Voice" if editing_brand else "🎙️ Learn My Brand Voice"
            col_btn, col_cancel = st.columns([3, 1])
            with col_cancel:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.show_brand_form = False
                    st.session_state.editing_brand_id = None
                    st.rerun()
            with col_btn:
                if st.button(btn_label, type="primary", use_container_width=True):
                    posts = [p.strip() for p in posts_raw.split("---") if p.strip()]
                    if not brand_name.strip():
                        st.warning("Enter a brand name.")
                    elif len(posts) < 2:
                        st.warning("Add at least 2 posts separated by `---`.")
                    else:
                        with st.spinner(f"Analyzing **{brand_name}**'s voice…"):
                            try:
                                profile = analyze_brand_voice(brand_name.strip(), platform, posts)
                                if editing_brand:
                                    saved = db.update_brand(editing_id, brand_name.strip(), platform, profile, posts)
                                else:
                                    saved = db.save_brand(brand_name.strip(), platform, profile, posts)
                                st.session_state.active_brand = saved
                                st.session_state.show_brand_form = False
                                st.session_state.editing_brand_id = None
                                st.session_state.results = None
                                st.success(f"✅ {'Updated' if editing_brand else 'Saved'}: **{brand_name}**")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Failed: {exc}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — GENERATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_gen:
    active = st.session_state.active_brand
    if not active:
        st.info("Select or create a brand in the **My Brands** tab first.")
        st.stop()

    # Active brand voice card
    p = active["profile"]
    traits_html = "".join(f'<span class="chip-blue">{t}</span>' for t in (p.get("style_traits") or []))
    st.markdown(f"""
    <div class="voice-card">
      <div class="voice-card-title">🎙️ {active["name"]} {_PLATFORM_PILL.get(active["platform"],"")}</div>
      <div class="voice-meta"><b>Tone:</b> {p.get("tone","—")} &nbsp;|&nbsp; <b>Sentences:</b> {p.get("sentence_style","—")}</div>
      <div class="voice-meta"><b>Emojis:</b> {p.get("emoji_usage","—")} {" ".join(p.get("preferred_emojis") or [])}</div>
      <div class="voice-meta"><b>Audience:</b> {p.get("audience","—")}</div>
      <div class="chip-row">{traits_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # Controls
    ctrl1, ctrl2 = st.columns([3, 1])
    with ctrl1:
        image_context = st.text_input(
            "Context (optional)",
            placeholder="e.g. New winter jacket, price £89 · Limited stock",
            help="Injected into every caption prompt for more specific output.",
        )
    with ctrl2:
        length_opts = {"short": "⚡ Short", "medium": "✦ Medium", "long": "📖 Long"}
        length = st.selectbox("Length", list(length_opts.keys()), index=1,
                              format_func=length_opts.get)

    # Upload
    uploaded = st.file_uploader(
        "Upload images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )

    if uploaded:
        st.caption(f"📂 {len(uploaded)} image{'s' if len(uploaded) > 1 else ''} ready")
        for row_start in range(0, len(uploaded), 5):
            row = uploaded[row_start: row_start + 5]
            cols = st.columns(len(row))
            for col, f in zip(cols, row):
                col.image(f.getvalue(), caption=f.name[:16], use_container_width=True)

    btn_text = (
        f"✨ Generate Captions ({len(uploaded)} image{'s' if len(uploaded) != 1 else ''})"
        if uploaded else "✨ Generate Captions"
    )

    if st.button(btn_text, type="primary", use_container_width=True, disabled=not uploaded):
        results: list[dict] = []
        bar    = st.progress(0, text="Starting…")
        status = st.empty()
        total  = len(uploaded)

        for i, f in enumerate(uploaded):
            status.markdown(f"⚙️ **{f.name}** — {i + 1} of {total}")
            result = generate_caption_variations(
                f.getvalue(),
                brand_profile=active["profile"],
                example_posts=active["example_posts"],
                context=image_context,
                num_variations=3,
                length=length,
            )
            result["filename"] = f.name
            results.append(result)
            bar.progress((i + 1) / total, text=f"✅ {i + 1} / {total} done")

        st.session_state.results = results
        status.success(f"Done! Captions ready for {total} image{'s' if total > 1 else ''}.")

    # Results
    if st.session_state.results:
        st.divider()

        for r in st.session_state.results:
            fname = r["filename"]
            with st.container(border=True):
                st.markdown(f'<div class="result-name">📷 {fname}</div>', unsafe_allow_html=True)

                if r.get("error"):
                    st.error(f"Failed: {r['error']}")
                    continue

                if r.get("image_analysis"):
                    st.markdown(
                        f'<div class="analysis-box">🔍 {r["image_analysis"]}</div>',
                        unsafe_allow_html=True,
                    )

                variations = r.get("variations", [])
                if not variations:
                    st.warning("No captions generated.")
                    continue

                # Variation selector
                var_labels = []
                for vi, v in enumerate(variations):
                    sc = v.get("score", 0)
                    stars = "★" * round(sc / 2) + "☆" * (5 - round(sc / 2))
                    var_labels.append(f"Variation {vi + 1}  {stars}  {sc}/10")

                sel_idx = st.radio(
                    "Choose variation",
                    range(len(variations)),
                    format_func=lambda i: var_labels[i],
                    horizontal=True,
                    key=f"var_{fname}",
                )

                sel    = variations[sel_idx]
                sc     = sel.get("score", 0)
                sc_cls = "score-hi" if sc >= 8 else "score-mid" if sc >= 6 else "score-lo"
                st.markdown(
                    f'<span class="{sc_cls}">{sc}/10</span>'
                    f' &nbsp;<span style="color:#64748B;font-size:0.8rem">{sel.get("score_reason","")}</span>',
                    unsafe_allow_html=True,
                )

                platform = active["platform"]
                fb_text = ig_text = ht_text = ""

                if platform in ("facebook", "both") and sel.get("facebook_caption"):
                    st.markdown('<div class="section-label">📘 Facebook Caption</div>', unsafe_allow_html=True)
                    fb_text = st.text_area(
                        "fb", value=sel["facebook_caption"],
                        key=f"fb_{fname}_{sel_idx}", height=110, label_visibility="collapsed",
                    )
                    st.markdown(f'<div class="char-hint">{len(fb_text):,} chars</div>', unsafe_allow_html=True)

                if platform in ("instagram", "both") and sel.get("instagram_caption"):
                    st.markdown('<div class="section-label">📸 Instagram Caption</div>', unsafe_allow_html=True)
                    ig_text = st.text_area(
                        "ig", value=sel["instagram_caption"],
                        key=f"ig_{fname}_{sel_idx}", height=110, label_visibility="collapsed",
                    )
                    st.markdown(f'<div class="char-hint">{len(ig_text):,} / 2,200 chars</div>', unsafe_allow_html=True)

                ht_default = " ".join(sel.get("hashtags") or [])
                st.markdown('<div class="section-label"># Hashtags</div>', unsafe_allow_html=True)
                ht_text = st.text_area(
                    "ht", value=ht_default,
                    key=f"ht_{fname}_{sel_idx}", height=60, label_visibility="collapsed",
                )

                # Schedule + Save
                sc1, sc2 = st.columns(2)
                with sc1:
                    sched_date = st.date_input(
                        "📅 Schedule date", value=None, key=f"date_{fname}", min_value=date.today(),
                    )
                with sc2:
                    sched_time = st.time_input("🕐 Time", value=None, key=f"time_{fname}")

                action_cols = st.columns([1, 1, 1])
                with action_cols[0]:
                    if st.button("💾 Save to Calendar", key=f"save_{fname}", use_container_width=True):
                        scheduled_iso: str | None = None
                        if sched_date and sched_time:
                            scheduled_iso = datetime.combine(
                                sched_date, sched_time, tzinfo=timezone.utc
                            ).isoformat()

                        if platform in ("facebook", "both") and fb_text:
                            db.save_caption(
                                brand_id=active["id"], brand_name=active["name"],
                                image_name=fname, platform="Facebook",
                                caption=fb_text, hashtags=ht_text,
                                score=float(sc), context=image_context, length_pref=length,
                                scheduled_date=scheduled_iso,
                            )
                        if platform in ("instagram", "both") and ig_text:
                            db.save_caption(
                                brand_id=active["id"], brand_name=active["name"],
                                image_name=fname, platform="Instagram",
                                caption=ig_text, hashtags=ht_text,
                                score=float(sc), context=image_context, length_pref=length,
                                scheduled_date=scheduled_iso,
                            )
                        st.success("Saved to Calendar ✓")

                with action_cols[1]:
                    buf_token = db.get_setting("buffer_token")
                    if buf_token:
                        if st.button("🚀 Post to Buffer", key=f"buf_{fname}", use_container_width=True):
                            full_post = f"{fb_text or ig_text}\n\n{ht_text}".strip()
                            try:
                                profiles = st.session_state.buffer_profiles
                                if not profiles:
                                    profiles = buf.get_profiles(buf_token)
                                    st.session_state.buffer_profiles = profiles
                                if not profiles:
                                    st.warning("No Buffer profiles found.")
                                else:
                                    siso = None
                                    if sched_date and sched_time:
                                        siso = datetime.combine(
                                            sched_date, sched_time, tzinfo=timezone.utc
                                        ).isoformat()
                                    buf.publish_post(buf_token, profiles[0]["id"], full_post, siso)
                                    st.success("Posted to Buffer ✓")
                            except Exception as exc:
                                st.error(f"Buffer error: {exc}")
                    else:
                        st.button("🚀 Buffer (add token →)", key=f"buf_{fname}",
                                  use_container_width=True, disabled=True,
                                  help="Add your Buffer access token in the sidebar.")

        # CSV export
        st.divider()
        brand_slug = active["name"].lower().replace(" ", "_")
        platform   = active["platform"]
        csv_rows: list[dict] = []

        for r in st.session_state.results:
            if r.get("error") or not r.get("variations"):
                continue
            fname = r["filename"]
            si    = st.session_state.get(f"var_{fname}", 0)
            sel   = r["variations"][si] if si < len(r["variations"]) else r["variations"][0]
            ht    = st.session_state.get(f"ht_{fname}_{si}", " ".join(sel.get("hashtags") or []))
            fb    = st.session_state.get(f"fb_{fname}_{si}", sel.get("facebook_caption") or "")
            ig    = st.session_state.get(f"ig_{fname}_{si}", sel.get("instagram_caption") or "")
            sd    = st.session_state.get(f"date_{fname}")
            st_   = st.session_state.get(f"time_{fname}")
            sched_str = f"{sd} {st_}" if sd and st_ else (str(sd) if sd else "")

            if platform in ("facebook", "both") and fb:
                csv_rows.append({
                    "Brand": active["name"], "Image": fname, "Platform": "Facebook",
                    "Caption": fb, "Hashtags": ht,
                    "Full Post": f"{fb}\n\n{ht}".strip(),
                    "Scheduled Date": sched_str, "Score": sel.get("score", ""),
                    "Image Analysis": r.get("image_analysis", ""),
                })
            if platform in ("instagram", "both") and ig:
                csv_rows.append({
                    "Brand": active["name"], "Image": fname, "Platform": "Instagram",
                    "Caption": ig, "Hashtags": ht,
                    "Full Post": f"{ig}\n\n{ht}".strip(),
                    "Scheduled Date": sched_str, "Score": sel.get("score", ""),
                    "Image Analysis": r.get("image_analysis", ""),
                })

        if csv_rows:
            df      = pd.DataFrame(csv_rows)
            buf_csv = io.StringIO()
            df.to_csv(buf_csv, index=False)
            st.download_button(
                "⬇️ Download Scheduler-Ready CSV",
                data=buf_csv.getvalue(),
                file_name=f"voxly_{brand_slug}_captions.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption("One row per platform per image — paste directly into Buffer, Hootsuite, or Later.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — CALENDAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_cal:
    all_captions = db.load_captions()

    if not all_captions:
        st.markdown(
            '<div class="empty-state">No posts saved yet. Generate captions and click "Save to Calendar".</div>',
            unsafe_allow_html=True,
        )
    else:
        fc1, fc2, fc3 = st.columns(3)
        all_brands_cal = db.load_brands()
        with fc1:
            filter_brand = st.selectbox(
                "Brand", ["All brands"] + [b["name"] for b in all_brands_cal], key="cal_fb"
            )
        with fc2:
            filter_plat = st.selectbox("Platform", ["All", "Facebook", "Instagram"], key="cal_fp")
        with fc3:
            filter_status = st.selectbox(
                "Status", ["All", "Draft", "Scheduled", "Published"], key="cal_fs"
            )

        filtered = all_captions
        if filter_brand != "All brands":
            filtered = [c for c in filtered if c["brand_name"] == filter_brand]
        if filter_plat != "All":
            filtered = [c for c in filtered if c["platform"] == filter_plat]
        if filter_status == "Draft":
            filtered = [c for c in filtered if not c["scheduled_date"] and not c["published"]]
        elif filter_status == "Scheduled":
            filtered = [c for c in filtered if c["scheduled_date"] and not c["published"]]
        elif filter_status == "Published":
            filtered = [c for c in filtered if c["published"]]

        st.caption(f"{len(filtered)} post{'s' if len(filtered) != 1 else ''}")
        st.markdown("<br>", unsafe_allow_html=True)

        for c in filtered:
            if c["published"]:
                status_html = '<span class="status-published">✅ Published</span>'
            elif c["scheduled_date"]:
                status_html = f'<span class="status-scheduled">🕐 {c["scheduled_date"][:16]}</span>'
            else:
                status_html = '<span class="status-draft">📝 Draft</span>'

            pill    = _PLATFORM_PILL.get(c["platform"].lower(), "")
            preview = c["caption"][:100].replace("\n", " ")

            st.markdown(f"""
            <div class="cal-card">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.3rem;">
                <span style="font-weight:700;font-size:0.88rem;color:#E2E8F0">📷 {c["image_name"]}</span>
                <span>{pill} {status_html}</span>
              </div>
              <div style="font-size:0.8rem;color:#94A3B8">{c["brand_name"]}</div>
              <div style="font-size:0.82rem;color:#CBD5E1;margin-top:0.35rem">{preview}…</div>
            </div>
            """, unsafe_allow_html=True)

            ac1, ac2, ac3 = st.columns([2, 1, 1])
            with ac1:
                if not c["published"]:
                    new_date = st.date_input(
                        "Reschedule",
                        value=date.fromisoformat(c["scheduled_date"][:10]) if c["scheduled_date"] else None,
                        key=f"cal_date_{c['id']}",
                        label_visibility="collapsed",
                    )
                    if new_date and str(new_date) != (c["scheduled_date"] or "")[:10]:
                        db.update_caption(c["id"], scheduled_date=str(new_date))
                        st.rerun()
            with ac2:
                if not c["published"]:
                    if st.button("✅ Mark Published", key=f"pub_{c['id']}", use_container_width=True):
                        db.update_caption(c["id"], published=1)
                        st.rerun()
            with ac3:
                if st.button("🗑️ Delete", key=f"caldel_{c['id']}", use_container_width=True):
                    db.delete_caption(c["id"])
                    st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — ANALYTICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_analytics:
    perf_rows = db.load_performance()
    stats     = analytics_store.summary_stats(perf_rows)

    ms1, ms2, ms3 = st.columns(3)
    with ms1:
        st.markdown(
            f'<div class="stat-box"><div class="stat-val">{stats["total_posts"]}</div>'
            f'<div class="stat-label">Posts tracked</div></div>', unsafe_allow_html=True
        )
    with ms2:
        st.markdown(
            f'<div class="stat-box"><div class="stat-val">{stats["avg_likes"]}</div>'
            f'<div class="stat-label">Avg likes</div></div>', unsafe_allow_html=True
        )
    with ms3:
        st.markdown(
            f'<div class="stat-box"><div class="stat-val">{stats["avg_reach"]}</div>'
            f'<div class="stat-label">Avg reach</div></div>', unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Add performance data
    published_caps = db.load_captions(published=True)
    with st.expander("➕ Add Performance Data", expanded=not perf_rows):
        if not published_caps:
            st.info("Mark posts as Published in the Calendar tab to add performance data here.")
        else:
            cap_options = {
                f"{c['image_name']} — {c['platform']} ({c['brand_name']})": c["id"]
                for c in published_caps if not db.has_performance(c["id"])
            }
            if not cap_options:
                st.success("Performance data already recorded for all published posts.")
            else:
                sel_label   = st.selectbox("Select post", list(cap_options.keys()))
                sel_cap_id  = cap_options[sel_label]
                pc1, pc2, pc3, pc4 = st.columns(4)
                likes    = pc1.number_input("Likes",    min_value=0, step=1)
                comments = pc2.number_input("Comments", min_value=0, step=1)
                reach    = pc3.number_input("Reach",    min_value=0, step=1)
                saves    = pc4.number_input("Saves",    min_value=0, step=1)
                if st.button("Save Performance Data", type="primary"):
                    db.save_performance(sel_cap_id, int(likes), int(comments), int(reach), int(saves))
                    st.success("Saved!")
                    st.rerun()

    # Chart
    if perf_rows:
        chart_df = pd.DataFrame(perf_rows)[["image_name", "platform", "likes", "reach", "saves"]].copy()
        chart_df["label"] = chart_df["image_name"].str[:14] + " · " + chart_df["platform"]
        st.markdown("**Performance by Post**")
        st.bar_chart(chart_df.set_index("label")[["likes", "reach", "saves"]])

    # AI Insights
    st.divider()
    st.markdown("**🤖 AI Insights**")

    active_for_insights = st.session_state.active_brand
    if not active_for_insights:
        st.info("Select an active brand in My Brands to get AI insights.")
    elif len(perf_rows) < 3:
        st.info("Add performance data for at least 3 posts to unlock AI insights.")
    else:
        if st.button("✨ Get AI Recommendations", type="primary"):
            with st.spinner("Analyzing your top and bottom performers…"):
                try:
                    insight_text = analytics_store.get_ai_insights(active_for_insights["profile"])
                    st.session_state["ai_insights"] = insight_text
                except Exception as exc:
                    st.error(f"Failed: {exc}")

        if st.session_state.get("ai_insights"):
            st.markdown(
                f'<div class="insight-box">'
                f'{st.session_state["ai_insights"].replace(chr(10), "<br>")}'
                f'</div>',
                unsafe_allow_html=True,
            )
