"""
config.py
=========
Single source of truth for every tunable constant in the pipeline.

Why centralize this?
Every RAG stage (chunking, retrieval, reranking, generation) has knobs that
you will want to tweak while demoing / tuning quality. Keeping them all in
one file means you never have to hunt through modules to change a threshold.
"""

import os
from dotenv import load_dotenv

# Load variables from a local .env file (ignored if not present, e.g. on
# Streamlit Cloud where secrets are injected via st.secrets instead).
load_dotenv()


def get_secret(key: str, default: str = "") -> str:
    """
    Fetch a secret from environment variables first (local dev via .env),
    falling back to Streamlit's secrets manager when deployed on
    Streamlit Community Cloud (which injects st.secrets, not env vars).
    """
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
GROQ_API_KEY = get_secret("GROQ_API_KEY")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Gemini 1.5 Flash is natively multimodal (text + image in one call) and has
# a generous free tier - used for vision captioning AND final answer
# generation.
GEMINI_MODEL = "gemini-3.1-flash-lite"

# Groq hosts Llama 3.x at very high throughput on a free tier - used as a
# fast fallback if Gemini is rate-limited, and for lightweight query
# transformation calls (HyDE / multi-query) to save Gemini quota.
GROQ_MODEL = "openai/gpt-oss-20b"

# Local, free, no-API embedding model. 384-dim, small enough to run on CPU
# comfortably, strong performance on retrieval benchmarks for its size.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Local, free cross-encoder used purely for reranking (NOT for the initial
# retrieval pass - cross-encoders are too slow to run over an entire corpus,
# so we only use them on the top-N candidates coming out of hybrid search).
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
# Semantic chunker groups sentences until the semantic "distance" between
# consecutive sentences spikes above this percentile - i.e. it breaks at
# genuine topic shifts rather than at a fixed token count.
SEMANTIC_BREAKPOINT_PERCENTILE = 90

# Safety bounds so semantic chunking never produces a chunk that's too tiny
# (useless context) or too huge (dilutes retrieval precision / blows the
# context window).
MIN_CHUNK_CHARS = 200
MAX_CHUNK_CHARS = 1800

# Sliding overlap (characters) applied when we fall back to the simple
# recursive splitter (used for table blocks, which shouldn't be split
# semantically since rows are positionally meaningful).
CHUNK_OVERLAP = 150

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
# How many candidates each retrieval arm (dense / sparse) pulls before fusion.
DENSE_TOP_K = 15
SPARSE_TOP_K = 15

# Weight given to dense (semantic) vs sparse (keyword/BM25) scores during
# Reciprocal Rank Fusion. 0.5 = equal weight. Insurance docs have a lot of
# exact-match terms (policy numbers, clause codes, dollar figures) where
# keyword search shines, so we lean slightly sparse.
HYBRID_ALPHA = 0.45  # weight on dense score; (1 - alpha) goes to sparse

# How many fused candidates get passed into the cross-encoder reranker.
RERANK_CANDIDATE_COUNT = 12

# How many reranked chunks finally get sent to the LLM as context.
FINAL_TOP_K = 5

# Retrieval Quality Control: if the top reranked chunk scores below this,
# we treat the corpus as "not containing a good answer" and tell the user
# instead of letting the LLM hallucinate from weak context.
MIN_RERANK_SCORE = -2.0  # cross-encoder ms-marco scores are unbounded logits;
                          # empirically, scores below ~-2 are poor matches.

# Number of query variants generated for RAG-Fusion.
RAG_FUSION_NUM_QUERIES = 4

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
GENERATION_TEMPERATURE = 0.2  # low temperature - we want grounded, factual
                               # answers, not creative ones, for a document
                               # Q&A tool.
MAX_OUTPUT_TOKENS = 1024

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------
DATA_DIR = "data"
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
