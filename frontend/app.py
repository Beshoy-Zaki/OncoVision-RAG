import hashlib
import io
import os
import pymupdf
import streamlit as st
from PIL import Image
from api_client import api_client

st.set_page_config(
    page_title="OncoVision-RAG | Neuro-Oncology Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Adaptive Dark/Light Mode Clinical Styling
st.markdown(
    """
<style>
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 12px;
        color: #F8FAFC;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15);
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        color: #CBD5E1;
        margin-bottom: 0.9rem;
    }
    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
    }
    .med-badge {
        background-color: rgba(255, 255, 255, 0.12);
        color: #E0F2FE;
        padding: 0.22rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid rgba(186, 230, 253, 0.25);
    }
    .vision-card {
        background-color: #F8FAFC !important;
        border: 1px solid #CBD5E1 !important;
        border-left: 5px solid #2563EB !important;
        padding: 12px 16px !important;
        border-radius: 8px !important;
        margin: 10px 0 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06) !important;
    }
    .vision-header {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }
    .vision-card-title {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }
    .vision-card-tag {
        font-weight: 800 !important;
        font-size: 0.88rem !important;
        padding: 3px 10px !important;
        border-radius: 6px !important;
        letter-spacing: 0.5px !important;
    }
    .vision-card-conf {
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }
    .doc-page-header {
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-left: 4px solid #1E3A8A !important;
        padding: 10px 14px !important;
        border-radius: 6px !important;
        margin-top: 10px !important;
        margin-bottom: 10px !important;
    }
    .passage-box {
        background-color: #F8FAFC;
        color: #1E293B;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 14px;
        font-size: 0.9rem;
        line-height: 1.55;
    }
</style>
""",
    unsafe_allow_html=True,
)

DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw_docs"))
if not os.path.exists(DOCS_DIR):
    DOCS_DIR = os.path.abspath(os.path.join("data", "raw_docs"))


@st.cache_data(ttl=15, show_spinner=False)
def get_cached_health():
    return api_client.check_health()


@st.cache_data(show_spinner=False, max_entries=64)
def render_pdf_page_with_bbox(
    source: str, page_num: int, chunk_text: str, bbox: list | None = None, dpi: int = 130
) -> bytes | None:
    """Renders the PDF page with a translucent highlight and bounding box around the cited block."""
    pdf_path = os.path.join(DOCS_DIR, source)
    if not os.path.exists(pdf_path):
        return None
    try:
        with pymupdf.open(pdf_path) as doc:
            p_idx = max(0, min(page_num - 1, len(doc) - 1))
            page = doc.load_page(p_idx)

            union_bbox = None
            if bbox and len(bbox) == 4:
                union_bbox = pymupdf.Rect(*bbox)
            else:
                body = chunk_text
                if body.startswith("[Document:") and "]" in body:
                    body = body.split("]", 1)[-1].strip()

                words = body.split()
                matched_rects = []
                for i in range(0, len(words), 5):
                    phrase = " ".join(words[i : i + 5])
                    if len(phrase) >= 8:
                        r = page.search_for(phrase)
                        if r:
                            matched_rects.extend(r)

                if matched_rects:
                    union_bbox = matched_rects[0]
                    for r in matched_rects[1:]:
                        union_bbox |= r

            if union_bbox:
                pad = 5
                padded = pymupdf.Rect(
                    max(0, union_bbox.x0 - pad),
                    max(0, union_bbox.y0 - pad),
                    min(page.rect.width, union_bbox.x1 + pad),
                    min(page.rect.height, union_bbox.y1 + pad),
                )
                # Subtle translucent highlight + sharp crimson border
                page.draw_rect(
                    padded,
                    color=(0.85, 0.1, 0.1),
                    fill=(1.0, 0.92, 0.2),
                    fill_opacity=0.18,
                    width=2.2,
                )

                badge_w = min(92, padded.width)
                badge_h = 13
                badge_y0 = padded.y0 - badge_h if padded.y0 >= badge_h else padded.y0
                badge_rect = pymupdf.Rect(
                    padded.x0, badge_y0, padded.x0 + badge_w, badge_y0 + badge_h
                )
                page.draw_rect(badge_rect, color=(0.85, 0.1, 0.1), fill=(0.85, 0.1, 0.1))
                page.insert_text(
                    (badge_rect.x0 + 4, badge_rect.y1 - 3.5),
                    "EXACT EVIDENCE",
                    fontsize=6.5,
                    color=(1, 1, 1),
                )

            pix = page.get_pixmap(dpi=dpi)
            return pix.tobytes("png")
    except Exception:
        return None


def render_vision_card(vf: dict) -> str:
    if not vf:
        return ""
    pred_cls = vf.get("detected_class", "unknown").upper()
    conf = vf.get("confidence", 0.0)
    probs = vf.get("probabilities", {})

    if pred_cls == "NOTUMOR":
        bg_tag, txt_tag, border_tag = "#ECFDF5", "#047857", "#A7F3D0"
    else:
        bg_tag, txt_tag, border_tag = "#FEF2F2", "#DC2626", "#FECACA"

    prob_pills = ""
    if probs:
        pills = [
            f"<span style='font-size:0.75rem; color:#475569; background:#E2E8F0; padding:2px 7px; border-radius:4px;'>"
            f"<b>{k.capitalize()}</b>: {v*100:.1f}%</span>"
            for k, v in sorted(probs.items(), key=lambda x: x[1], reverse=True)
        ]
        prob_pills = f"<div style='display:flex; gap:6px; flex-wrap:wrap; margin-top:8px;'>{''.join(pills)}</div>"

    return f"""
    <div class="vision-card">
        <div class="vision-header">
            <span class="vision-card-title">👁️ Computer Vision Finding:</span>
            <span class="vision-card-tag" style="background-color: {bg_tag}; color: {txt_tag}; border: 1px solid {border_tag};">{pred_cls}</span>
            <span class="vision-card-conf">({conf*100:.1f}% confidence)</span>
        </div>
        {prob_pills}
    </div>
    """


def render_web_grounding(web_sources: list):
    st.markdown(
        """
    <div style="background-color: #F0FDF4; border-left: 4px solid #10B981; padding: 8px 12px; border-radius: 6px; margin-top: 6px; margin-bottom: 8px;">
        <span style="color: #065F46; font-weight: 700; font-size: 0.88rem;">🌐 Grounded via Real-Time Google Search</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if web_sources:
        with st.expander(f"🔗 Verified Web Sources ({len(web_sources)})", expanded=True):
            for w_idx, ws in enumerate(web_sources, 1):
                title = ws.get("title", "Web Reference")
                url = ws.get("url", "#")
                st.markdown(f"**{w_idx}.** [{title}]({url})")


def render_document_grounding(chunks: list):
    """Renders side-by-side PDF evidence bounding boxes and clean extracted text."""
    if not chunks:
        return
    with st.expander(f"📍 Document Grounding: Verified Textbook & Guideline Pages ({len(chunks)})", expanded=True):
        for idx, c in enumerate(chunks, 1):
            src = c.get("source", "Unknown Document")
            page_num = c.get("page", 1)
            raw_text = c.get("text", "")

            body_text = raw_text
            if "[Document:" in body_text and "]" in body_text:
                body_text = body_text.split("]", 1)[-1].strip()

            hierarchy = c.get("hierarchy", "")
            rerank_score = c.get("rerank_score")
            score_badge = (
                f"<span style='background-color: #EFF6FF; color: #1D4ED8; font-size: 0.76rem; "
                f"font-weight: 700; padding: 2px 7px; border-radius: 4px; margin-left: 8px; "
                f"border: 1px solid #BFDBFE;'>🎯 Relevance: {rerank_score*100:.1f}%</span>"
                if rerank_score is not None
                else ""
            )
            hierarchy_html = (
                f"<div style='font-size: 0.8rem; color: #475569; margin-top: 4px; font-weight: 500;'>"
                f"📂 <i>{hierarchy}</i></div>"
                if hierarchy
                else ""
            )

            st.markdown(
                f"""
            <div class="doc-page-header">
                <div>
                    <span style="color: #1E3A8A; font-weight: 700; font-size: 0.95rem;">📄 Source {idx}:</span>
                    <span style="color: #0F172A; font-weight: 600; font-size: 0.95rem;">{src}</span> —
                    <span style="color: #475569; font-size: 0.9rem;">Page {page_num}</span>
                    {score_badge}
                </div>
                {hierarchy_html}
            </div>
            """,
                unsafe_allow_html=True,
            )

            col_pdf, col_txt = st.columns([3, 2], gap="medium")
            with col_pdf:
                chunk_bbox = c.get("bbox")
                page_png = render_pdf_page_with_bbox(src, page_num, body_text, bbox=chunk_bbox)
                if page_png:
                    st.image(
                        page_png,
                        caption=f"{src} (Page {page_num}) — Highlighted Evidence Block",
                        use_container_width=True,
                    )
                else:
                    st.caption(f"Original PDF page preview unavailable for {src} (Page {page_num}).")

            with col_txt:
                st.markdown("**Extracted Clinical Passage:**")
                st.markdown(f'<div class="passage-box">{body_text}</div>', unsafe_allow_html=True)


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_analyzed_image_hash" not in st.session_state:
    st.session_state.last_analyzed_image_hash = None
if "latest_vision_finding" not in st.session_state:
    st.session_state.latest_vision_finding = None

# Sidebar
with st.sidebar:
    st.markdown("### 🧠 OncoVision Control")

    col_status, col_refresh = st.columns([4, 1])
    with col_refresh:
        if st.button("🔄", help="Refresh backend health status"):
            get_cached_health.clear()

    health = get_cached_health()
    if health:
        st.success(
            f"🟢 **Backend Online**\n\n"
            f"• Indexed Chunks: `{health.get('total_chunks', 0):,}`\n"
            f"• Hybrid BM25: `{'Active' if health.get('bm25_loaded') else 'Off'}`\n"
            f"• YOLO ONNX: `{'Loaded' if health.get('yolo_model_loaded') else 'Missing'}`"
        )
    else:
        st.error(f"🔴 **Backend Offline**\n\nEnsure FastAPI is running on `{api_client.base_url}`.")

    st.divider()

    st.subheader("🖼️ Patient MRI Study")
    st.caption("Upload an axial T1/T2 Brain MRI scan for automated YOLO classification & grounded retrieval.")

    uploaded_file = st.file_uploader(
        "Upload Brain MRI (.jpg, .png)", type=["jpg", "jpeg", "png"]
    )

    uploaded_image_bytes = None
    attach_mri_to_query = False

    if uploaded_file is not None:
        uploaded_image_bytes = uploaded_file.getvalue()
        img_hash = hashlib.md5(uploaded_image_bytes).hexdigest()
        is_new_upload = img_hash != st.session_state.last_analyzed_image_hash

        image = Image.open(io.BytesIO(uploaded_image_bytes))
        st.image(image, caption=f"Study: {uploaded_file.name}", use_container_width=True)

        # Automatically enable MRI attachment on a fresh upload; let clinician toggle for follow-ups
        attach_mri_to_query = st.toggle(
            "Include MRI scan in next query",
            value=is_new_upload,
            help="When enabled, the MRI scan is analyzed by YOLO and fused into the next clinical query.",
        )

        # Display latest YOLO probabilities if this image was already analyzed
        if not is_new_upload and st.session_state.latest_vision_finding:
            vf = st.session_state.latest_vision_finding
            st.caption(f"**Latest Finding:** `{vf['detected_class'].upper()}` ({vf['confidence']*100:.1f}%)")
            for cls_name, prob in vf.get("probabilities", {}).items():
                st.progress(min(1.0, max(0.0, float(prob))), text=f"{cls_name.capitalize()}: {prob*100:.1f}%")
    else:
        st.session_state.last_analyzed_image_hash = None
        st.session_state.latest_vision_finding = None

    st.divider()

    st.subheader("💡 Clinical Quick Queries")
    sample_queries = [
        "What does Simpson Grade I resection require?",
        "What is the Stupp protocol for Glioblastoma?",
        "When is 'watch-and-scan' indicated for Meningiomas?",
        "What is first-line therapy for Prolactinomas?",
        "Analyze this MRI scan and summarize standard management.",
    ]
    for idx, sq in enumerate(sample_queries):
        if st.button(sq, key=f"preset_btn_{idx}", use_container_width=True):
            st.session_state["user_input_preset"] = sq

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Hero Banner
st.markdown(
    """
<div class="hero-banner">
    <div class="hero-title">🧠 OncoVision-RAG: Neuro-Oncology Clinical Assistant</div>
    <div class="hero-subtitle">Grounded clinical decision support powered by <b>Osborn's Brain</b>, <b>EANO & WHO CNS5 Guidelines</b>, and <b>YOLOv8 ONNX</b>.</div>
    <div class="badge-row">
        <span class="med-badge">📚 1,893 Pages Indexed</span>
        <span class="med-badge">⚡ Hybrid Dense + BM25 (RRF)</span>
        <span class="med-badge">🎯 FlashRank Cross-Encoder</span>
        <span class="med-badge">👁️ YOLOv8 ONNX Vision</span>
        <span class="med-badge">📍 Native PDF Bounding Boxes</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("image"):
            st.image(msg["image"], width=200, caption="Attached MRI Study")
        if msg.get("vision_finding"):
            st.markdown(render_vision_card(msg["vision_finding"]), unsafe_allow_html=True)
        if msg.get("low_relevance_warning"):
            st.warning(msg["low_relevance_warning"])
        st.markdown(msg["content"])
        if msg.get("is_web_search"):
            render_web_grounding(msg.get("web_sources", []))
        if msg.get("chunks"):
            render_document_grounding(msg["chunks"])

# Handle User Input
preset_val = st.session_state.pop("user_input_preset", "")
prompt = st.chat_input("Ask a neuro-oncology question or discuss the uploaded MRI study...")
user_question = prompt or preset_val

if user_question:
    # Determine if we should send the MRI image bytes on this turn
    send_image_bytes = uploaded_image_bytes if (uploaded_image_bytes and attach_mri_to_query) else None

    # Build prior history payload before appending current user turn
    prior_history = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-4:]
    ]

    user_entry = {"role": "user", "content": user_question}
    if send_image_bytes:
        user_entry["image"] = send_image_bytes
    st.session_state.messages.append(user_entry)

    with st.chat_message("user"):
        if send_image_bytes:
            st.image(send_image_bytes, width=200, caption="Attached MRI Study")
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Synthesizing evidence across 1,893 pages of Neuro-Oncology Guidelines..."):
            filename = uploaded_file.name if uploaded_file else "scan.jpg"
            response = api_client.query(
                question=user_question,
                image_bytes=send_image_bytes,
                filename=filename,
                chat_history=prior_history,
            )

        if response and not response.get("error"):
            answer = response.get("answer", "")
            sources = response.get("sources", [])
            chunks = response.get("chunks", [])
            vision_finding = response.get("vision_finding")
            is_web_search = response.get("is_web_search", False)
            low_relevance_warning = response.get("low_relevance_warning")
            web_sources = response.get("web_sources", [])

            if send_image_bytes and vision_finding:
                st.session_state.last_analyzed_image_hash = hashlib.md5(send_image_bytes).hexdigest()
                st.session_state.latest_vision_finding = vision_finding

            if vision_finding:
                st.markdown(render_vision_card(vision_finding), unsafe_allow_html=True)

            if low_relevance_warning:
                st.warning(low_relevance_warning)

            st.markdown(answer)

            if is_web_search:
                render_web_grounding(web_sources)

            if chunks:
                render_document_grounding(chunks)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "chunks": chunks,
                    "vision_finding": vision_finding,
                    "is_web_search": is_web_search,
                    "low_relevance_warning": low_relevance_warning,
                    "web_sources": web_sources,
                }
            )
        else:
            err_msg = (
                response.get("detail", "An unexpected error occurred.")
                if response
                else "Backend did not respond."
            )
            st.error(f"⚠️ **Request Failed**: {err_msg}")
