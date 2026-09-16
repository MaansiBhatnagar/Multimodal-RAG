"""
src/chunking/semantic_chunker.py
==================================
Stage 2: CHUNKING.

Why not just split every 500 characters?
Fixed-size chunking routinely slices a sentence (or a policy clause) in
half, splitting its meaning across two chunks that then embed poorly and
retrieve poorly. Semantic chunking instead asks: "where do consecutive
sentences actually change topic?" and only cuts there.

Algorithm (embedding-based breakpoint detection, similar to the approach
popularized by Greg Kamradt / LlamaIndex's SemanticSplitter):

  1. Split text into sentences.
  2. Embed every sentence (reusing the same embedding model as the index -
     see indexing/embeddings.py - so distances are meaningful).
  3. Compute cosine distance between each pair of *consecutive* sentences.
  4. Anything above the Nth percentile of all distances in the document is
     treated as a topic-shift "breakpoint".
  5. Group sentences between breakpoints into a chunk.
  6. Enforce MIN/MAX_CHUNK_CHARS as safety rails (merge tiny chunks
     forward, hard-split runaway chunks) so retrieval granularity stays
     sane regardless of how choppy or run-on the source prose is.

Tables and image captions are NOT run through this - they're structurally
atomic (a table's rows only make sense together), so they pass through as
a single chunk each, tagged with their own metadata (see chunking.chunk()).
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import nltk

from src.ingestion.loader import RawElement
import config
from utils.logger import get_logger

logger = get_logger("chunking.semantic")

# Ensure the sentence tokenizer model is available. Downloaded once and
# cached; safe to call repeatedly.
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


@dataclass
class Chunk:
    text: str
    element_type: str      # "text" | "table" | "image" | "scanned_page"
    page_number: int
    source_file: str
    chunk_index: int = 0
    extra: dict = field(default_factory=dict)


class SemanticChunker:
    def __init__(self, embedder):
        """
        `embedder` is an EmbeddingModel instance (src/indexing/embeddings.py).
        Injected rather than imported directly so we load the sentence-
        transformer model exactly once per app session and share it across
        chunking + indexing + query embedding.
        """
        self.embedder = embedder

    def chunk_elements(self, elements: list[RawElement]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for el in elements:
            if el.element_type == "text":
                chunks.extend(self._semantic_split(el))
            else:
                # Tables / image captions / scanned-page captions: keep as
                # one atomic chunk (further split only if truly oversized).
                chunks.extend(self._atomic_or_hard_split(el))

        for i, c in enumerate(chunks):
            c.chunk_index = i

        logger.info(f"Chunked {len(elements)} elements into {len(chunks)} chunks.")
        return chunks

    # ------------------------------------------------------------------
    def _semantic_split(self, el: RawElement) -> list[Chunk]:
        sentences = nltk.sent_tokenize(el.content)
        if len(sentences) <= 1:
            return self._atomic_or_hard_split(el)

        embeddings = self.embedder.encode(sentences)
        distances = self._consecutive_distances(embeddings)

        if not distances:
            return self._atomic_or_hard_split(el)

        threshold = np.percentile(distances, config.SEMANTIC_BREAKPOINT_PERCENTILE)
        breakpoints = {i for i, d in enumerate(distances) if d > threshold}

        # Group sentences into segments split at breakpoints.
        segments: list[list[str]] = []
        current: list[str] = [sentences[0]]
        for i in range(1, len(sentences)):
            if (i - 1) in breakpoints:
                segments.append(current)
                current = [sentences[i]]
            else:
                current.append(sentences[i])
        segments.append(current)

        # Merge undersized segments forward, hard-split oversized ones.
        merged_texts = self._enforce_size_bounds(segments)

        return [
            Chunk(text=t, element_type="text", page_number=el.page_number, source_file=el.source_file)
            for t in merged_texts
            if t.strip()
        ]

    @staticmethod
    def _consecutive_distances(embeddings: np.ndarray) -> list[float]:
        distances = []
        for i in range(len(embeddings) - 1):
            a, b = embeddings[i], embeddings[i + 1]
            cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            distances.append(1 - cos_sim)  # cosine distance
        return distances

    @staticmethod
    def _enforce_size_bounds(segments: list[list[str]]) -> list[str]:
        texts = [" ".join(seg) for seg in segments]

        # Merge forward any segment that's below MIN_CHUNK_CHARS.
        merged: list[str] = []
        buffer = ""
        for t in texts:
            buffer = f"{buffer} {t}".strip() if buffer else t
            if len(buffer) >= config.MIN_CHUNK_CHARS:
                merged.append(buffer)
                buffer = ""
        if buffer:
            if merged:
                merged[-1] = f"{merged[-1]} {buffer}"
            else:
                merged.append(buffer)

        # Hard-split anything still over MAX_CHUNK_CHARS with a sliding
        # window, so one giant paragraph can't blow the context window.
        final: list[str] = []
        for t in merged:
            if len(t) <= config.MAX_CHUNK_CHARS:
                final.append(t)
                continue
            start = 0
            while start < len(t):
                end = start + config.MAX_CHUNK_CHARS
                final.append(t[start:end])
                start = end - config.CHUNK_OVERLAP
        return final

    def _atomic_or_hard_split(self, el: RawElement) -> list[Chunk]:
        """Used for tables/images (keep whole) and degenerate text blocks
        (single sentence, or too short to semantically split)."""
        text = el.content
        if len(text) <= config.MAX_CHUNK_CHARS:
            return [Chunk(text, el.element_type, el.page_number, el.source_file, extra=dict(el.extra))]

        # Oversized table/caption - split with overlap rather than semantic
        # breakpoints (structure, not topic, governs these).
        out = []
        start = 0
        while start < len(text):
            end = start + config.MAX_CHUNK_CHARS
            out.append(Chunk(text[start:end], el.element_type, el.page_number, el.source_file, extra=dict(el.extra)))
            start = end - config.CHUNK_OVERLAP
        return out
