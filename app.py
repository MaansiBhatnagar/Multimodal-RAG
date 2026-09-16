"""
app.py
=======
Streamlit front-end for the multimodal Insurance Document Navigator.

Three tabs:
  1. Chat        - upload any document, ask questions (optionally with an
                    attached image), see cited answers + a live "pipeline
                    trace" panel showing exactly what each RAG stage did.
  2. Sources     - browse every chunk that was retrieved for the last
                    answer, with rerank scores, page numbers, and type.
  3. Evaluation  - run the LLM-as-judge evaluator over a few Q&A pairs to
                    score faithfulness/relevance live, on this document.

Although branded as an "Insurance Document Navigator", the pipeline is
fully generic - any PDF/DOCX/TXT works.
"""

import os
import tempfile

import streamlit as st
from PIL import Image

from src.pipeline import RAGPipeline
from src.evaluation.evaluator import Evaluator

# ---------------------------------------------------------------------------
# Page config + styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Insurance Document Navigator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --brand-navy: #0F2C4C;
    --brand-teal: #1B998B;
    --brand-bg: #F7F9FB;
}
.stApp { background-color: var(--brand-bg); }

/* Header banner */
.hero {
    background: linear-gradient(135deg, var(--brand-navy) 0%, #17406b 100%);
    padding: 1.6rem 2rem;
    border-radius: 14px;
    color: white;
    margin-bottom: 1.2rem;
}
.hero h1 { margin: 0; font-size: 1.7rem; }
.hero p { margin: 0.35rem 0 0 0; opacity: 0.85; font-size: 0.95rem; }

/* Source cards */
.source-card {
    background: white;
    border-left: 4px solid var(--brand-teal);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.source-tag {
    display: inline-block;
    background: var(--brand-teal);
    color: white;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
    margin-right: 6px;
}
.confidence-badge-high {
    background: #E4F7EE; color: #1B7A4C; padding: 3px 10px;
    border-radius: 20px; font-size: 0.78rem; font-weight: 600;
}
.confidence-badge-low {
    background: #FDECEC; color: #B02A2A; padding: 3px 10px;
    border-radius: 20px; font-size: 0.78rem; font-weight: 600;
}
.trace-step {
    border-left: 3px solid #C8D3DE;
    padding-left: 0.8rem;
    margin-bottom: 0.7rem;
}
.trace-stage { font-weight: 700; color: var(--brand-navy); font-size: 0.85rem; }
.trace-detail { color: #445; font-size: 0.82rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached resources (loaded once per session, not on every Streamlit rerun)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline() -> RAGPipeline:
    return RAGPipeline()


@st.cache_resource(show_spinner=False)
def load_evaluator(_pipeline: RAGPipeline) -> Evaluator:
    return Evaluator(_pipeline.llm)


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, sources?, trace?}
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "last_trace" not in st.session_state:
    st.session_state.last_trace = []

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>📄 Insurance Document Navigator</h1>
        <p>Upload any policy document (or any PDF/DOCX/TXT really) and ask questions in plain English —
        with citations back to the exact page, table, or diagram. Multimodal: attach a photo too.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: upload + settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("📤 Upload a document")
    uploaded_file = st.file_uploader(
        "PDF, DOCX, or TXT — any document works, not just insurance policies.",
        type=["pdf", "docx", "txt", "md"],
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        if st.session_state.doc_name != uploaded_file.name:
            with st.status("Running ingestion pipeline...", expanded=True) as status:
                pipeline = load_pipeline()

                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name

                progress_bar = st.progress(0.0)

                def progress_cb(stage, pct):
                    progress_bar.progress(pct)
                    st.write(f"• {stage}")

                collection_name = pipeline.ingest(tmp_path, file_bytes, progress_cb=progress_cb)
                os.unlink(tmp_path)

                st.session_state.collection_name = collection_name
                st.session_state.doc_name = uploaded_file.name
                st.session_state.messages = []
                status.update(label=f"✅ Indexed '{uploaded_file.name}'", state="complete")

    if st.session_state.doc_name:
        st.success(f"Active document:\n**{st.session_state.doc_name}**")
    else:
        st.info("No document uploaded yet.")

    st.divider()
    st.subheader("🧠 Pipeline")
    st.caption(
        "Semantic chunking → hybrid (dense+sparse) retrieval → query "
        "transformation (HyDE / RAG-Fusion / Step-back, auto-routed) → "
        "cross-encoder reranking → grounded generation (Gemini 1.5 Flash)."
    )
    show_trace = st.checkbox("Show pipeline trace after each answer", value=True)

    st.divider()
    st.caption("Built as a portfolio project — multimodal RAG pipeline, 100% free-tier stack.")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_chat, tab_sources, tab_eval = st.tabs(["💬 Chat", "🔎 Retrieved Sources", "📊 Evaluation"])

# ===========================================================================
# TAB 1: CHAT
# ===========================================================================
with tab_chat:
    if not st.session_state.collection_name:
        st.warning("👈 Upload a document in the sidebar to get started.")
    else:
        # Render conversation history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("confident") is not None:
                    badge_class = "confidence-badge-high" if msg["confident"] else "confidence-badge-low"
                    badge_text = "High confidence" if msg["confident"] else "Low confidence — check original doc"
                    st.markdown(f'<span class="{badge_class}">{badge_text}</span>', unsafe_allow_html=True)

        # Optional image attachment for the NEXT question (multimodal query)
        attached_image_file = st.file_uploader(
            "📎 Attach a photo with your question (optional) — e.g. damage photo, bill, ID card",
            type=["jpg", "jpeg", "png"],
            key=f"img_upload_{len(st.session_state.messages)}",
        )
        attached_image = None
        if attached_image_file is not None:
            attached_image = Image.open(attached_image_file).convert("RGB")
            st.image(attached_image, width=180, caption="Attached image")

        question = st.chat_input("Ask a question about your document...")

        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking through the document..."):
                    pipeline = load_pipeline()
                    result = pipeline.query(
                        question=question,
                        collection_name=st.session_state.collection_name,
                        image=attached_image,
                    )
                st.markdown(result.answer)
                badge_class = "confidence-badge-high" if result.is_confident else "confidence-badge-low"
                badge_text = "High confidence" if result.is_confident else "Low confidence — check original doc"
                st.markdown(f'<span class="{badge_class}">{badge_text}</span>', unsafe_allow_html=True)

                if show_trace and result.trace:
                    with st.expander("🔬 See how the pipeline produced this answer"):
                        for step in result.trace:
                            st.markdown(
                                f'<div class="trace-step">'
                                f'<div class="trace-stage">{step["stage"]}</div>'
                                f'<div class="trace-detail">{step["detail"]}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

            st.session_state.messages.append(
                {"role": "assistant", "content": result.answer, "confident": result.is_confident}
            )
            st.session_state.last_sources = result.sources
            st.session_state.last_trace = result.trace
            st.rerun()

# ===========================================================================
# TAB 2: RETRIEVED SOURCES
# ===========================================================================
with tab_sources:
    st.subheader("Sources used in the last answer")
    if not st.session_state.last_sources:
        st.caption("Ask a question in the Chat tab to see the retrieved, reranked source chunks here.")
    else:
        for s in st.session_state.last_sources:
            st.markdown(
                f"""
                <div class="source-card">
                    <span class="source-tag">{s['tag']}</span>
                    <b>Page {s['page']}</b> — {s.get('section') or 'Untitled section'}
                    &nbsp;·&nbsp; <i>{s['type']}</i>
                    &nbsp;·&nbsp; relevance score: <b>{s['score']}</b>
                    <p style="margin-top:0.5rem; font-size:0.88rem; color:#333;">{s['text'][:500]}{'…' if len(s['text'])>500 else ''}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ===========================================================================
# TAB 3: EVALUATION
# ===========================================================================
with tab_eval:
    st.subheader("📊 Evaluate retrieval + generation quality")
    st.caption(
        "Runs an LLM-as-judge evaluation (faithfulness + relevance, each scored 1-5) "
        "over a set of test questions against the currently indexed document."
    )

    if not st.session_state.collection_name:
        st.warning("Upload a document first.")
    else:
        default_qs = "What is the deductible?\nWhat is not covered under this policy?\nHow do I file a claim?"
        eval_questions_raw = st.text_area(
            "Test questions (one per line):", value=default_qs, height=100
        )

        if st.button("▶️ Run evaluation"):
            pipeline = load_pipeline()
            evaluator = load_evaluator(pipeline)
            questions = [q.strip() for q in eval_questions_raw.split("\n") if q.strip()]

            results_table = []
            progress = st.progress(0.0)
            for i, q in enumerate(questions):
                result = pipeline.query(q, st.session_state.collection_name)
                context_text = "\n\n".join(s["text"] for s in result.sources)
                eval_result = evaluator.judge(q, context_text, result.answer)
                results_table.append(
                    {
                        "Question": q,
                        "Faithfulness (1-5)": eval_result.faithfulness,
                        "Relevance (1-5)": eval_result.relevance,
                        "Retrieval confident": "✅" if result.is_confident else "⚠️",
                        "Judge reasoning": eval_result.reason,
                    }
                )
                progress.progress((i + 1) / len(questions))

            st.dataframe(results_table, use_container_width=True)

            avg_faith = sum(r["Faithfulness (1-5)"] for r in results_table) / len(results_table)
            avg_rel = sum(r["Relevance (1-5)"] for r in results_table) / len(results_table)
            col1, col2 = st.columns(2)
            col1.metric("Avg. Faithfulness", f"{avg_faith:.2f} / 5")
            col2.metric("Avg. Relevance", f"{avg_rel:.2f} / 5")
