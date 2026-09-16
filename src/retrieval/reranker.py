"""
src/retrieval/reranker.py
===========================
Stage 7: RERANKING + Stage 8: RETRIEVAL QUALITY CONTROL.

Why rerank at all if hybrid retrieval already ranked things?
Dense/sparse retrieval both score a query against a chunk INDEPENDENTLY -
they never actually let the query and the chunk "attend" to each other. A
cross-encoder does: it feeds the (query, chunk) pair together through a
transformer and outputs a single relevance logit, which is far more
accurate but also far more expensive - so we only ever run it over the
small candidate set (RERANK_CANDIDATE_COUNT, e.g. 12) coming out of hybrid
fusion, never the whole corpus.

Retrieval Quality Control:
After reranking, if even the BEST candidate scores below a threshold, that
is a strong signal the document simply doesn't contain a good answer to
this question. Rather than forcing the LLM to generate an answer from weak
context (a major hallucination risk), we surface this explicitly so the UI
can tell the user "I couldn't find this in your document" instead of
fabricating a confident-sounding but ungrounded answer.
"""

from __future__ import annotations
from dataclasses import dataclass
from sentence_transformers import CrossEncoder

import config
from utils.logger import get_logger

logger = get_logger("retrieval.reranker")


@dataclass
class RerankedChunk:
    id: str
    text: str
    metadata: dict
    rerank_score: float


class Reranker:
    def __init__(self):
        logger.info(f"Loading cross-encoder reranker: {config.RERANKER_MODEL}")
        self.model = CrossEncoder(config.RERANKER_MODEL)

    def rerank(self, query: str, candidates: list[dict]) -> tuple[list[RerankedChunk], bool]:
        """
        Returns (top_chunks, is_confident).
        `is_confident` is False when the top score is below
        config.MIN_RERANK_SCORE - this is the retrieval-quality-control gate
        the generation stage checks before answering.
        """
        if not candidates:
            return [], False

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        scored = [
            RerankedChunk(id=c["id"], text=c["text"], metadata=c["metadata"], rerank_score=float(s))
            for c, s in zip(candidates, scores)
        ]
        scored.sort(key=lambda x: x.rerank_score, reverse=True)
        top = scored[: config.FINAL_TOP_K]

        is_confident = bool(top) and top[0].rerank_score >= config.MIN_RERANK_SCORE
        logger.info(
            f"Reranked {len(candidates)} candidates. "
            f"Top score: {top[0].rerank_score:.3f} | confident={is_confident}"
        )
        return top, is_confident
