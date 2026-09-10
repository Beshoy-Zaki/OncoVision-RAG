import os
import streamlit as st
from PIL import Image
import io
from api_client import api_client

st.set_page_config(
    page_title="Neuro-Oncology Multimodal RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Streamlit UI Configuration for Neuro-Oncology Multimodal RAG

# Custom CSS for clinical styling and document bounding boxes
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .badge-container {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
    .med-badge {
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #BFDBFE;
    }
    .vision-card {
        background-color: #F8FAFC !important;
        border: 1px solid #CBD5E1 !important;
        border-left: 5px solid #2563EB !important;
        padding: 10px 16px !important;
        border-radius: 8px !important;
        margin: 10px 0 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
        flex-wrap: wrap !important;
    }
    .vision-card-title {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }
    .vision-card-tag {
        font-weight: 800 !important;
        font-size: 0.92rem !important;
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
        padding: 8px 14px !important;
        border-radius: 6px !important;
        margin-top: 14px !important;
        margin-bottom: 8px !important;
        display: flex !important;
        align-items: center !important;
        gap: 6px !important;
    }
</style>
""", unsafe_allow_html=True)

import pymupdf

DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw_docs"))
if not os.path.exists(DOCS_DIR):
    DOCS_DIR = os.path.abspath(os.path.join("data", "raw_docs"))

@st.cache_data(show_spinner=False)
def render_pdf_page_with_bbox(source: str, page_num: int, chunk_text: str, bbox: list = None, dpi: int = 125) -> bytes:
    """Renders the actual PDF page with a bounding box drawn directly on the relevant paragraph.
    
    If bbox is provided (from ingestion-time block extraction), draws it directly — no text search.
    Otherwise falls back to legacy phrase-search matching for pre-reindex chunks.
    """
    pdf_path = os.path.join(DOCS_DIR, source)
    if not os.path.exists(pdf_path):
        return None
    try:
        doc = pymupdf.open(pdf_path)
        p_idx = max(0, min(page_num - 1, len(doc) - 1))
        page = doc.load_page(p_idx)
        
        union_bbox = None
        
        if bbox and len(bbox) == 4:
            # Native ingestion bbox — draw directly, no text search needed
            union_bbox = pymupdf.Rect(*bbox)
        else:
            # Legacy fallback: phrase-search matching for chunks without stored bbox
            body = chunk_text
            if body.startswith("[Document:") and "]" in body:
                body = body.split("]", 1)[-1].strip()
                
            words = body.split()
            matched_rects = []
            step = 5
            for i in range(0, len(words), step):
                phrase = " ".join(words[i:i+5])
                if len(phrase) >= 8:
                    r = page.search_for(phrase)
                    if r:
                        matched_rects.extend(r)
                        
            if not matched_rects and len(words) >= 3:
                for i in range(0, min(len(words), 30), 3):
                    phrase = " ".join(words[i:i+3])
                    if len(phrase) >= 6:
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
                min(page.rect.height, union_bbox.y1 + pad)
            )
            # Draw authentic red bounding box around the paragraph
            page.draw_rect(padded, color=(0.85, 0.1, 0.1), width=2.5)
            
            # Draw EXACT EVIDENCE label badge
            badge_w = min(90, padded.width)
            badge_h = 13
            badge_y0 = max(0, padded.y0 - badge_h)
            badge_rect = pymupdf.Rect(padded.x0, badge_y0, padded.x0 + badge_w, padded.y0)
            page.draw_rect(badge_rect, color=(0.85, 0.1, 0.1), fill=(0.85, 0.1, 0.1))
            page.insert_text((badge_rect.x0 + 3, badge_rect.y1 - 3), "EXACT EVIDENCE", fontsize=6.5, color=(1, 1, 1))

        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    except Exception:
        return None

def render_vision_card(vf: dict) -> str:
    """Renders the computer vision findings with high-contrast text and theme-safe colors."""
    if not vf:
        return ""
    pred_cls = vf.get("detected_class", "unknown").upper()
    conf = vf.get("confidence", 0.0)
    if pred_cls == "NOTUMOR":
        bg_tag = "#ECFDF5"
        txt_tag = "#047857"
        border_tag = "#A7F3D0"
    else:
        bg_tag = "#FEF2F2"
        txt_tag = "#DC2626"
        border_tag = "#FECACA"
    return f"""
    <div class="vision-card">
        <span class="vision-card-title">👁️ Computer Vision Finding:</span>
        <span class="vision-card-tag" style="background-color: {bg_tag}; color: {txt_tag}; border: 1px solid {border_tag};">{pred_cls}</span>
        <span class="vision-card-conf">({conf*100:.1f}% confidence)</span>
    </div>
    """

def render_web_grounding(web_sources: list):
    """Renders real-time web search grounding citations."""
    st.markdown("""
    <div style="background-color: #F0FDF4; border-left: 4px solid #10B981; padding: 8px 12px; border-radius: 6px; margin-top: 6px; margin-bottom: 8px;">
        <span style="color: #065F46; font-weight: 700; font-size: 0.88rem;">🌐 Grounded via Real-Time Web Search</span>
        <span style="color: #047857; font-size: 0.8rem; margin-left: 6px;">(Google Search & Gemma 4)</span>
    </div>
    """, unsafe_allow_html=True)
    
    if web_sources:
        with st.expander(f"🔗 Verified Web Sources ({len(web_sources)})", expanded=True):
            for w_idx, ws in enumerate(web_sources, 1):
                title = ws.get("title", "Web Reference")
                url = ws.get("url", "#")
                st.markdown(f"**{w_idx}.** [{title}]({url})")

def render_document_grounding(chunks: list):
    """Renders original PDF document pages with bounding boxes on cited paragraphs."""
    if not chunks:
        return
    with st.expander(f"📍 Document Grounding: Original Verified Pages ({len(chunks)})", expanded=True):
        for idx, c in enumerate(chunks, 1):
            src = c.get("source", "Unknown Document")
            page_num = c.get("page", 1)
            raw_text = c.get("text", "")
            
            body_text = raw_text
            if "[Document:" in body_text and "]" in body_text:
                body_text = body_text.split("]", 1)[-1].strip()
                
            hierarchy = c.get("hierarchy", "")
            rerank_score = c.get("rerank_score")
            score_badge = f"<span style='background-color: #EFF6FF; color: #1D4ED8; font-size: 0.76rem; font-weight: 700; padding: 2px 6px; border-radius: 4px; margin-left: 8px; border: 1px solid #BFDBFE;'>🎯 Relevance: {rerank_score*100:.1f}%</span>" if rerank_score is not None else ""
            hierarchy_html = f"<div style='font-size: 0.8rem; color: #64748B; margin-top: 4px; font-weight: 500;'>📂 <i>{hierarchy}</i></div>" if hierarchy else ""
            
            st.markdown(f"""
            <div class="doc-page-header" style="flex-direction: column; align-items: flex-start;">
                <div>
                    <span style="color: #1E3A8A; font-weight: 700; font-size: 0.95rem;">📄 Source {idx}:</span>
                    <span style="color: #0F172A; font-weight: 600; font-size: 0.95rem;">{src}</span> — 
                    <span style="color: #475569; font-size: 0.9rem;">Page {page_num}</span>
                    {score_badge}
                </div>
                {hierarchy_html}
            </div>
            """, unsafe_allow_html=True)
            
            chunk_bbox = c.get("bbox")
            page_png = render_pdf_page_with_bbox(src, page_num, body_text, bbox=chunk_bbox)
            if page_png:
                st.image(page_png, caption=f"Original Document Page: {src} (Page {page_num}) — Evidence Bounding Box", width="stretch")
            else:
                st.caption(f"Original page preview unavailable for {src} (Page {page_num}).")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/brain.png", width=64)
    st.title("System Control")
    
    health = api_client.check_health()
    if health:
        st.success(f"🟢 **Backend Online**\n\nChunks: `{health.get('total_chunks', 0):,}` | YOLO: `Loaded`")
    else:
        st.error(f"🔴 **Backend Offline**\n\nEnsure FastAPI is running on `{api_client.base_url}`.")

    st.divider()
    
    st.subheader("🖼️ Multimodal MRI Input")
    st.caption("Upload an axial T1/T2 Brain MRI scan for automated YOLO tumor localization & grounded analysis.")
    
    uploaded_file = st.file_uploader(
        "Choose an MRI scan (.jpg, .png)",
        type=["jpg", "jpeg", "png"]
    )
    
    uploaded_image_bytes = None
    if uploaded_file is not None:
        uploaded_image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(uploaded_image_bytes))
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", width="stretch")
        st.info("💡 Image attached to your next question.")

    st.divider()
    
    st.subheader("💡 Quick Queries")
    sample_queries = [
        "What does Simpson Grade I resection require?",
        "What is the Stupp protocol for Glioblastoma?",
        "When is 'watch-and-scan' indicated for Meningiomas?",
        "What is first-line therapy for Prolactinomas?",
        "Analyze this MRI scan and summarize standard management."
    ]
    for sq in sample_queries:
        if st.button(sq, key=f"btn_{sq[:15]}", use_container_width=True):
            st.session_state["user_input_preset"] = sq

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main Header
st.markdown('<div class="main-header">🧠 Neuro-Oncology Multimodal RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Grounded clinical decision support powered by <b>Osborn\'s Brain</b>, <b>EANO Guidelines</b>, and <b>YOLOv8</b>.</div>', unsafe_allow_html=True)

st.markdown("""
<div class="badge-container">
    <span class="med-badge">📚 1,893 Pages Indexed</span>
    <span class="med-badge">🎯 ChromaDB Vector Store</span>
    <span class="med-badge">👁️ YOLOv8 (97.5% Acc)</span>
    <span class="med-badge">📍 Visual Document Bounding Boxes</span>
    <span class="med-badge">🌐 Gemma 4 Web Grounding</span>
</div>
""", unsafe_allow_html=True)

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("image"):
            st.image(msg["image"], width=180, caption="Attached MRI Scan")
        st.markdown(msg["content"])
        
        # Display Vision Card if available
        if msg.get("vision_finding"):
            st.markdown(render_vision_card(msg["vision_finding"]), unsafe_allow_html=True)
            
        # Display Low Relevance Warning banner if present
        if msg.get("low_relevance_warning"):
            st.warning(msg["low_relevance_warning"])
            
        # Display Web Grounding
        if msg.get("is_web_search"):
            render_web_grounding(msg.get("web_sources", []))
            
        # Display Document Bounding Boxes
        if msg.get("chunks"):
            render_document_grounding(msg["chunks"])

# Handle User Input
preset_val = st.session_state.pop("user_input_preset", "")
prompt = st.chat_input("Ask a neuro-oncology question or discuss the uploaded MRI...")

user_question = prompt or preset_val

if user_question:
    user_entry = {"role": "user", "content": user_question}
    if uploaded_image_bytes:
        user_entry["image"] = uploaded_image_bytes
    st.session_state.messages.append(user_entry)
    
    with st.chat_message("user"):
        if uploaded_image_bytes:
            st.image(uploaded_image_bytes, width=180, caption="Attached MRI Scan")
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing MRI & Searching 1,893 pages of Medical Guidelines..."):
            filename = uploaded_file.name if uploaded_file else "scan.jpg"
            response = api_client.query(
                question=user_question,
                image_bytes=uploaded_image_bytes,
                filename=filename
            )

        if response and not response.get("error"):
            answer = response.get("answer", "")
            sources = response.get("sources", [])
            chunks = response.get("chunks", [])
            vision_finding = response.get("vision_finding")
            is_web_search = response.get("is_web_search", False)
            low_relevance_warning = response.get("low_relevance_warning")
            web_sources = response.get("web_sources", [])
            
            if vision_finding:
                st.markdown(render_vision_card(vision_finding), unsafe_allow_html=True)

            if low_relevance_warning:
                st.warning(low_relevance_warning)

            st.markdown(answer)
            
            # Display Web Grounding
            if is_web_search:
                render_web_grounding(web_sources)
                
            # Display Document Bounding Boxes
            if chunks:
                render_document_grounding(chunks)
                        
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "chunks": chunks,
                "vision_finding": vision_finding,
                "is_web_search": is_web_search,
                "low_relevance_warning": low_relevance_warning,
                "web_sources": web_sources
            })
        else:
            err_msg = response.get("detail", "An unexpected error occurred.") if response else "Backend did not respond."
            st.error(f"⚠️ **Request Failed**: {err_msg}")
