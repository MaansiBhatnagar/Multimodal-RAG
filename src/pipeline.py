"""
src/pipeline.py
=================
Orchestrates every stage into two public entry points:

  RAGPipeline.ingest(file_path)      -> builds/updates the index for a doc
  RAGPipeline.query(question, image) -> runs the full retrieval+generation
                                         flow and returns a rich result
                                         object (answer + sources + full
                                         trace of what each stage did, for
                                         UI transparency)

Kept deliberately thin - each stage's actual logic lives in its own module
(see the imports below); this file just wires them together in order and
records a trace so the Streamlit UI can show "how the answer was produced"
as a portfolio-worthy transparency feature.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
from PIL import Image

from src.ingestion.loader import DocumentLoader
from src.ingestion.image_processor import ImageProcessor
from src.chunking.semantic_chunker import SemanticChunker
from src.indexing.embeddings import EmbeddingModel
from src.indexing.metadata_enricher import enrich
from src.indexing.vector_store import VectorStore
from src.retrieval.query_transform import QueryTransformer
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.generation.prompt_builder import build_prompt
from src.generation.llm import LLMClient
from utils.logger import get_logger, TraceCollector

logger = get_logger("pipeline")


@dataclass
class QueryResult:
    answer: str
    is_confident: bool
    sources: list[dict] = field(default_factory=list)   # [{page, section, text, score}]
    trace: list[dict] = field(default_factory=list)      # step-by-step pipeline trace


class RAGPipeline:
    """
    One instance = one document's worth of index + all the shared, heavy
    models (embedder, reranker, LLM client). Cache this at the Streamlit
    layer with st.cache_resource so models load exactly once per session.
    """

    def __init__(self):
        logger.info("Initializing RAG pipeline components (loading models)...")
        self.llm = LLMClient()
        self.embedder = EmbeddingModel()
        self.reranker = Reranker()
        self.query_transformer = QueryTransformer(self.llm)
        self.loader = DocumentLoader()
        self.image_processor = ImageProcessor(self.llm)
        self.chunker = SemanticChunker(self.embedder)

        self._stores: dict[str, VectorStore] = {}   # collection_name -> VectorStore
        logger.info("Pipeline ready.")

    # ------------------------------------------------------------------
    # INGESTION
    # ------------------------------------------------------------------
    def collection_name_for(self, file_path: str, file_bytes: bytes) -> str:
        """Deterministic collection name so re-uploading the same file
        reuses its existing index instead of duplicating it."""
        digest = hashlib.md5(file_bytes).hexdigest()[:12]
        return f"doc_{digest}"

    def ingest(self, file_path: str, file_bytes: bytes, progress_cb=None) -> str:
        """Runs the full ingestion -> chunking -> enrichment -> indexing
        pipeline for one uploaded file. Returns the collection_name so the
        caller can route subsequent queries to it.
        `progress_cb(stage: str, pct: float)` is an optional callback for
        driving a Streamlit progress bar."""

        def report(stage, pct):
            logger.info(f"[{pct*100:.0f}%] {stage}")
            if progress_cb:
                progress_cb(stage, pct)

        collection_name = self.collection_name_for(file_path, file_bytes)
        store = VectorStore(collection_name, self.embedder)

        if store.count() > 0:
            report("Document already indexed - reusing existing index.", 1.0)
            self._stores[collection_name] = store
            return collection_name

        report("Parsing document (text, tables, images)...", 0.15)
        elements = self.loader.load(file_path)

        report("Processing images (vision captioning / OCR)...", 0.35)
        elements = self.image_processor.process(elements)

        report("Semantic chunking...", 0.55)
        chunks = self.chunker.chunk_elements(elements)

        report("Enriching chunk metadata...", 0.70)
        enriched = enrich(chunks)

        report("Building hybrid index (dense + sparse)...", 0.90)
        store.add_chunks(enriched)

        self._stores[collection_name] = store
        report(f"Done - indexed {len(enriched)} chunks.", 1.0)
        return collection_name

    # ------------------------------------------------------------------
    # QUERY
    # ------------------------------------------------------------------
    def query(
        self,
        question: str,
        collection_name: str,
        image: Image.Image | None = None,
    ) -> QueryResult:
        trace = TraceCollector()

        store = self._stores.get(collection_name)
        if store is None:
            store = VectorStore(collection_name, self.embedder)
            self._stores[collection_name] = store

        if store.count() == 0:
            return QueryResult(
                answer="This document hasn't been indexed yet - please upload it first.",
                is_confident=False,
            )

        # Multimodal query handling: if the user attached an image, caption
        # it and fold that description into the retrieval query and prompt,
        # so e.g. a photo of car damage can retrieve the relevant collision
        # clause even if the user's text question is minimal.
        image_description = None
        effective_question = question
        if image is not None:
            image_description = self.llm.caption_image(
                image, context_hint="A user is asking an insurance question and attached this photo."
            )
            trace.add("Multimodal input", f"Captioned uploaded image: \"{image_description[:150]}...\"")
            effective_question = f"{question}\n(Attached image shows: {image_description})"

        # Stage 5: query transformation
        transform_result = self.query_transformer.transform(effective_question)
        trace.add(
            f"Query transformation ({transform_result.strategy})",
            transform_result.detail + f" Queries used: {transform_result.search_queries}",
        )

        # Stage 6: hybrid retrieval
        retriever = HybridRetriever(store)
        candidates = retriever.retrieve(transform_result)
        trace.add(
            "Hybrid retrieval",
            f"Dense + sparse search across {len(transform_result.search_queries)} quer(y/ies), "
            f"fused via Reciprocal Rank Fusion -> {len(candidates)} candidates for reranking.",
        )

        if not candidates:
            trace.add("Result", "No candidates retrieved at all - the index may be empty.")
            return QueryResult(
                answer="I couldn't find anything relevant to that question in this document.",
                is_confident=False,
                trace=trace.as_list(),
            )

        # Stage 7 + 8: rerank + quality control
        top_chunks, is_confident = self.reranker.rerank(question, candidates)
        trace.add(
            "Reranking (cross-encoder) + quality control",
            f"Reranked top {len(candidates)} candidates -> kept top {len(top_chunks)}. "
            f"Top relevance score: {top_chunks[0].rerank_score:.2f} | "
            f"Confidence gate: {'PASSED' if is_confident else 'FAILED (low confidence)'}",
        )

        # Stage 9: prompt construction
        prompt = build_prompt(question, top_chunks, is_confident, image_description)
        trace.add("Prompt construction", f"Built grounded prompt with {len(top_chunks)} cited excerpts.")

        # Stage 10 (generation)
        answer = self.llm.generate_answer(prompt, image=image)
        trace.add("Generation", "Gemini 1.5 Flash generated the final grounded answer.")

        sources = [
            {
                "tag": f"S{i+1}",
                "page": c.metadata.get("page_number"),
                "section": c.metadata.get("section_title"),
                "type": c.metadata.get("element_type"),
                "text": c.text,
                "score": round(c.rerank_score, 3),
            }
            for i, c in enumerate(top_chunks)
        ]

        return QueryResult(answer=answer, is_confident=is_confident, sources=sources, trace=trace.as_list())
