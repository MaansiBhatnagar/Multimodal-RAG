"""
src/retrieval/hybrid_retriever.py
===================================
Stage 6: HYBRID RETRIEVAL.

Runs dense (semantic) + sparse (BM25 keyword) search, potentially across
MULTIPLE query variants (from query_transform.py's RAG-Fusion output), and
merges everything with Reciprocal Rank Fusion (RRF).

Why RRF instead of just averaging raw scores?
Dense cosine similarity and BM25 scores live on completely different,
unnormalized scales - averaging them directly is meaningless. RRF instead
only looks at each document's RANK within each result list
(score = sum of 1/(k + rank) across lists), which is scale-invariant and a
well-established, simple way to fuse heterogeneous ranked lists (used in
the original RAG-Fusion paper and standard IR literature).
"""

from __future__ import annotations
from collections import defaultdict

from src.indexing.vector_store import VectorStore
from src.retrieval.query_transform import TransformResult
import config
from utils.logger import get_logger

logger = get_logger("retrieval.hybrid")

RRF_K = 60  # standard smoothing constant from the RRF literature


class HybridRetriever:
    def __init__(self, store: VectorStore):
        self.store = store

    def retrieve(self, transform_result: TransformResult) -> list[dict]:
        """Runs dense + sparse search for every query variant, fuses all
        result lists with weighted RRF, and returns a deduplicated,
        fused-score-sorted candidate list (length up to
        RERANK_CANDIDATE_COUNT) ready for the reranker."""

        fused_scores: dict[str, float] = defaultdict(float)
        chunk_lookup: dict[str, dict] = {}

        for query in transform_result.search_queries:
            dense_results = self.store.dense_search(query, top_k=config.DENSE_TOP_K)
            sparse_results = self.store.sparse_search(query, top_k=config.SPARSE_TOP_K)

            self._accumulate_rrf(dense_results, fused_scores, chunk_lookup, weight=config.HYBRID_ALPHA)
            self._accumulate_rrf(sparse_results, fused_scores, chunk_lookup, weight=1 - config.HYBRID_ALPHA)

        ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)
        candidates = [
            {**chunk_lookup[cid], "fusion_score": fused_scores[cid]}
            for cid in ranked_ids[: config.RERANK_CANDIDATE_COUNT]
        ]

        logger.info(
            f"Hybrid retrieval: {len(transform_result.search_queries)} quer(y/ies) -> "
            f"{len(fused_scores)} unique candidates -> top {len(candidates)} passed to reranker."
        )
        return candidates

    @staticmethod
    def _accumulate_rrf(results: list[dict], fused_scores: dict, chunk_lookup: dict, weight: float):
        for rank, r in enumerate(results):
            cid = r["id"]
            fused_scores[cid] += weight * (1.0 / (RRF_K + rank + 1))
            chunk_lookup[cid] = r
