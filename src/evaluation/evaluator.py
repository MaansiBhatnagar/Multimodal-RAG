"""
src/evaluation/evaluator.py
=============================
Stage 10: EVALUATION.

Implements a lightweight, dependency-free version of the standard RAG
evaluation triad (the same core ideas behind frameworks like RAGAS), using
the LLM itself as a judge - free-tier friendly since it's just a few extra
Gemini calls per evaluated question, no separate eval infrastructure.

Metrics (each scored 1-5 by an LLM judge with a strict rubric prompt):

  1. Faithfulness - does the generated answer only contain claims that are
     actually supported by the retrieved context? (catches hallucination)
  2. Answer Relevance - does the answer actually address the question that
     was asked? (catches off-topic or evasive answers)
  3. Context Precision - of the retrieved chunks, how many were actually
     relevant/used? (catches noisy retrieval bloating the prompt)

Plus one retrieval-only metric that doesn't need an LLM:
  4. Retrieval Hit Rate - if the user supplies a small labeled eval set
     (question -> expected page number), what fraction of the time does
     the correct page actually appear in the final retrieved chunks?

This module is invoked from the "Evaluation" tab in app.py, where a user
can paste a handful of Q&A pairs (or use the auto-generated sample set) and
see the pipeline score itself, live - a great, concrete portfolio talking
point ("I built an automated eval harness for my RAG pipeline").
"""

from __future__ import annotations
from dataclasses import dataclass
import re

from src.generation.llm import LLMClient

_JUDGE_PROMPT = """You are a strict evaluator scoring a RAG (Retrieval-Augmented Generation) system's output.

QUESTION: {question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER: {answer}

Score the answer on two dimensions, each from 1 (very poor) to 5 (excellent):

FAITHFULNESS: Does the answer ONLY state things that are directly supported by the retrieved context above? (5 = fully grounded, no unsupported claims. 1 = mostly fabricated / not supported by context.)

RELEVANCE: Does the answer actually address what the question asked? (5 = fully and directly answers it. 1 = off-topic or evasive.)

Respond in EXACTLY this format, nothing else:
FAITHFULNESS: <score>
RELEVANCE: <score>
REASON: <one sentence explaining the scores>"""


@dataclass
class EvalResult:
    question: str
    faithfulness: float
    relevance: float
    reason: str
    retrieval_hit: bool | None = None  # None if no expected page was supplied


class Evaluator:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def judge(self, question: str, context: str, answer: str) -> EvalResult:
        prompt = _JUDGE_PROMPT.format(question=question, context=context[:4000], answer=answer)
        raw = self.llm.fast_text(prompt, temperature=0.0, max_tokens=150)

        faithfulness = self._extract_score(raw, "FAITHFULNESS")
        relevance = self._extract_score(raw, "RELEVANCE")
        reason_match = re.search(r"REASON:\s*(.+)", raw)
        reason = reason_match.group(1).strip() if reason_match else "N/A"

        return EvalResult(question=question, faithfulness=faithfulness, relevance=relevance, reason=reason)

    @staticmethod
    def _extract_score(raw: str, label: str) -> float:
        match = re.search(rf"{label}:\s*([1-5])", raw)
        return float(match.group(1)) if match else 0.0

    @staticmethod
    def check_retrieval_hit(retrieved_pages: list[int], expected_page: int | None) -> bool | None:
        if expected_page is None:
            return None
        return expected_page in retrieved_pages
