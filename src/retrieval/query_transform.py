"""
src/retrieval/query_transform.py
==================================
Stage 5: QUERY TRANSLATION / RECONSTRUCTION.

A raw user question is often a poor retrieval query - too short, too
conversational, or phrased differently than the document's own language.
This module implements three complementary techniques and a lightweight
router that picks the best one per query (so the pipeline stays efficient
instead of always paying for all three):

  1. HyDE (Hypothetical Document Embeddings): ask the LLM to write a
     hypothetical *answer* to the question, then embed and search with
     THAT instead of the raw question. Works well when the question is
     vague/short, because a hypothetical answer uses document-like
     language ("Collision coverage applies when...") that matches chunk
     phrasing far better than a question does.

  2. RAG-Fusion (multi-query + Reciprocal Rank Fusion): ask the LLM to
     generate several differently-worded versions of the question, run
     retrieval for each, then fuse the ranked lists. Works well for
     ambiguous or broad questions, since different phrasings surface
     different relevant chunks that a single query would miss.

  3. Step-back prompting: ask the LLM to reformulate a specific question
     into a more general "step back" question first (e.g. "does my plan
     cover a chipped windshield" -> "what does my plan's glass/windshield
     coverage include") - helpful when the user's specific phrasing is
     narrower than how the document actually organizes the topic.

Router heuristic: we classify the query cheaply (no extra LLM call) by
surface features, defaulting to RAG-Fusion (the most broadly robust
technique) unless the query looks vague (-> HyDE) or looks like a narrow
specific-instance question (-> step-back).
"""

from __future__ import annotations
import re
from dataclasses import dataclass

from src.generation.llm import LLMClient
from utils.logger import get_logger
import config

logger = get_logger("retrieval.query_transform")


@dataclass
class TransformResult:
    strategy: str
    search_queries: list[str]     # one or more queries to actually run retrieval with
    detail: str                   # human-readable trace note for the UI


class QueryTransformer:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    # ------------------------------------------------------------------
    def transform(self, question: str) -> TransformResult:
        strategy = self._route(question)
        if strategy == "hyde":
            return self._hyde(question)
        elif strategy == "step_back":
            return self._step_back(question)
        else:
            return self._rag_fusion(question)

    # ------------------------------------------------------------------
    def _route(self, question: str) -> str:
        word_count = len(question.split())
        q_lower = question.lower()

        # Very short / vague questions -> HyDE (needs a richer hypothetical
        # answer to anchor retrieval).
        if word_count <= 5:
            return "hyde"

        # Narrow "does X specific thing" questions naming a concrete
        # instance -> step-back to the general category first.
        specific_markers = ["this", "my ", "chipped", "dented", "specific", "particular"]
        if any(m in q_lower for m in specific_markers) and word_count <= 14:
            return "step_back"

        # Default: RAG-Fusion, the most broadly robust technique for
        # normal, well-formed questions.
        return "rag_fusion"

    # ------------------------------------------------------------------
    def _hyde(self, question: str) -> TransformResult:
        prompt = (
            "You are an expert insurance analyst. Write a short, confident "
            "hypothetical passage (3-4 sentences) that WOULD answer the "
            "following question, as if it were taken directly from an "
            "insurance policy document. Do not hedge or say you're unsure - "
            "write it in the authoritative style of policy document text.\n\n"
            f"Question: {question}\n\nHypothetical policy passage:"
        )
        hypothetical = self.llm.fast_text(prompt, temperature=0.4, max_tokens=180)
        logger.info(f"[HyDE] generated hypothetical passage for: '{question}'")
        return TransformResult(
            strategy="HyDE",
            search_queries=[hypothetical if hypothetical else question],
            detail=f"Generated a hypothetical answer passage and searched using it "
                    f"instead of the raw question (better lexical match to policy language).",
        )

    # ------------------------------------------------------------------
    def _rag_fusion(self, question: str) -> TransformResult:
        prompt = (
            f"Generate {config.RAG_FUSION_NUM_QUERIES} different search queries "
            "that all aim to retrieve information relevant to answering the "
            "question below. Vary the phrasing, terminology, and specificity "
            "(e.g. one more literal, one using likely document jargon, one "
            "broader, one narrower). Return ONLY the queries, one per line, "
            "no numbering, no extra commentary.\n\n"
            f"Question: {question}"
        )
        raw = self.llm.fast_text(prompt, temperature=0.5, max_tokens=200)
        queries = [q.strip("-•* \t") for q in raw.split("\n") if q.strip()]
        queries = queries[: config.RAG_FUSION_NUM_QUERIES] or [question]
        if question not in queries:
            queries.append(question)  # always keep the original as a safety net

        logger.info(f"[RAG-Fusion] generated {len(queries)} query variants.")
        return TransformResult(
            strategy="RAG-Fusion",
            search_queries=queries,
            detail=f"Generated {len(queries)} reworded variants of your question and "
                    f"fused their retrieval results (Reciprocal Rank Fusion) for broader recall.",
        )

    # ------------------------------------------------------------------
    def _step_back(self, question: str) -> TransformResult:
        prompt = (
            "Reformulate the following specific question into a more general "
            "'step-back' question about the broader category or policy "
            "section it falls under. Return ONLY the reformulated question.\n\n"
            f"Specific question: {question}\n\nStep-back question:"
        )
        step_back_q = self.llm.fast_text(prompt, temperature=0.3, max_tokens=60)
        logger.info(f"[Step-back] '{question}' -> '{step_back_q}'")
        # Search with BOTH the general step-back question and the original,
        # specific question - the general one surfaces the relevant policy
        # section, the specific one catches any exact-match detail.
        queries = [step_back_q, question] if step_back_q else [question]
        return TransformResult(
            strategy="Step-back",
            search_queries=queries,
            detail=f"Reformulated your specific question into a broader question "
                    f"('{step_back_q}') to first locate the right policy section, "
                    f"then searched both.",
        )
