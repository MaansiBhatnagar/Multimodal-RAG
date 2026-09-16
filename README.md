# 📄 Insurance Document Navigator — Multimodal Advanced RAG Pipeline

A full advanced-RAG pipeline (not just "embed + retrieve + generate") built to answer
questions about **any uploaded document** (PDF / DOCX / TXT) — text, tables, scanned
pages, embedded diagrams, and user-uploaded photos all handled. Branded around an
insurance use case, but fully document-agnostic.

100% free-tier stack — no paid APIs, no paid hosting required.

---

## 🧠 Pipeline Architecture

Every stage lives in its own module so each part can be explained, demoed, or swapped
independently.

| # | Stage | Module | What it does |
|---|-------|--------|---------------|
| 1 | **Ingestion / Parsing** | `src/ingestion/loader.py` | Extracts text, tables (as markdown), embedded images, and flags scanned/image-only pages — per element, not flattened into one blob |
| 1b| **Multimodal normalization** | `src/ingestion/image_processor.py` | Converts every image/scanned page into rich text via Gemini vision captioning (OCR fallback via Tesseract) so it becomes searchable |
| 2 | **Chunking** | `src/chunking/semantic_chunker.py` | Embedding-based **semantic breakpoint** chunking — splits at real topic shifts, not fixed character counts. Tables/images stay atomic. |
| 3 | **Metadata enrichment** | `src/indexing/metadata_enricher.py` | Adds section titles, keyword tags, dollar-figure flags, page numbers to every chunk |
| 4 | **Indexing** | `src/indexing/vector_store.py` | Dual index: **Chroma** (dense/semantic) + **BM25** (sparse/keyword) |
| 5 | **Query transformation** | `src/retrieval/query_transform.py` | Auto-routes each question to **HyDE**, **RAG-Fusion** (multi-query), or **Step-back prompting** based on query characteristics |
| 6 | **Hybrid retrieval** | `src/retrieval/hybrid_retriever.py` | Runs dense + sparse search across all query variants, fuses with **Reciprocal Rank Fusion** |
| 7 | **Reranking** | `src/retrieval/reranker.py` | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) reranks the fused candidate set |
| 8 | **Retrieval quality control** | `src/retrieval/reranker.py` | Confidence gate — if top rerank score is too low, the app tells the user honestly instead of hallucinating |
| 9 | **Prompt construction** | `src/generation/prompt_builder.py` | Builds a citation-tagged, grounded prompt (`[S1]`, `[S2]`…) |
| 10| **Generation** | `src/generation/llm.py` | Gemini 1.5 Flash — natively multimodal, so an attached user photo goes into the *same* generation call |
| 11| **Evaluation** | `src/evaluation/evaluator.py` | LLM-as-judge scoring: faithfulness + answer relevance, live in the app |

`src/pipeline.py` orchestrates all of the above and returns a full **trace** of every
step, which the UI displays so anyone (e.g. a recruiter) can see exactly how an answer
was produced — this is the single best "show, don't tell" feature for a portfolio demo.

### Multimodality, explained
- **Ingestion side:** images/diagrams/tables embedded in the document, and fully
  scanned pages, are all converted to rich text descriptions via Gemini vision (with
  a local OCR fallback), then flow through the *same* chunking/indexing/retrieval
  pipeline as regular text. This means a table or diagram is just as retrievable as
  a paragraph.
- **Query side:** a user can attach a photo (e.g. a photo of car damage) alongside
  their text question. The photo is captioned, folded into the retrieval query
  *and* passed directly into the Gemini generation call, so the model reasons over
  the image and the retrieved policy text together.

---

## 🗂 Project Structure

```
insurance-rag/
├── app.py                      # Streamlit UI (3 tabs: Chat, Sources, Evaluation)
├── config.py                   # All tunable constants in one place
├── requirements.txt
├── .env.example
├── src/
│   ├── ingestion/               # parsing + multimodal normalization
│   ├── chunking/                # semantic chunker
│   ├── indexing/                # embeddings, metadata, vector store
│   ├── retrieval/                # query transform, hybrid retrieval, reranker
│   ├── generation/               # prompt builder, LLM client
│   ├── evaluation/               # LLM-as-judge evaluator
│   └── pipeline.py               # orchestrator
└── utils/logger.py
```

---

## 🚀 Local setup

```bash
git clone <your-repo-url>
cd insurance-rag

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Tesseract OCR binary (not a python package) - needed for the OCR fallback
# macOS:   brew install tesseract
# Ubuntu:  sudo apt-get install tesseract-ocr
# Windows: https://github.com/UB-Mannheim/tesseract/wiki

cp .env.example .env
# edit .env and paste in your free GEMINI_API_KEY and GROQ_API_KEY

streamlit run app.py
```

Get your free keys:
- Gemini: https://aistudio.google.com/app/apikey
- Groq: https://console.groq.com/keys

---

## ☁️ Deploying for free (Streamlit Community Cloud)

1. Push this project to a public (or private) GitHub repo.
2. Go to https://share.streamlit.io → "New app" → connect your repo → set main file
   to `app.py`.
3. In **App Settings → Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "your_key_here"
   GROQ_API_KEY = "your_key_here"
   ```
   (`config.py` reads from `st.secrets` automatically when env vars aren't present.)
4. Add a `packages.txt` file (already included) so Streamlit Cloud installs the
   Tesseract OCR system binary.
5. Deploy — you'll get a shareable `*.streamlit.app` URL.

**Note on persistence:** Streamlit Cloud's filesystem is ephemeral — the Chroma index
resets on app restart/redeploy. That's fine for a portfolio demo (users upload their
own doc each session). If you want indexes to persist across restarts, wire up
Supabase Storage to sync the `data/chroma_db` folder — left as a clearly-scoped
extension since it's not needed for the core demo.

---

## 🎯 Talking points for interviews / portfolio write-ups

- Explain **why** hybrid retrieval beats pure dense search (exact dollar figures /
  policy numbers vs. paraphrased meaning).
- Explain the **query-transform router**: why HyDE for vague queries but RAG-Fusion
  for well-formed ones — and show the trace panel proving it happens live.
- Explain the **retrieval quality control gate** — a concrete example of designing
  against hallucination, not just prompting "don't make things up."
- Walk through the **evaluation tab** — this shows you think about RAG as a system
  that needs measurement, not just a demo that "looks like it works."
