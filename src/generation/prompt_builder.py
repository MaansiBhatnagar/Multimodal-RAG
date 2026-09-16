"""
src/generation/prompt_builder.py
==================================
Stage 9: PROMPT CONSTRUCTION.

Builds the final prompt sent to Gemini for answer generation. Key design
choices:

  - Each retrieved chunk is labeled with a citation tag [S1], [S2]... and
    its page number/section, and the model is instructed to cite sources
    inline using those tags. This is what lets the UI show "grounded in
    page 12" next to the answer - the whole point of using RAG over a
    document instead of a generic chatbot.
  - The model is explicitly told to say so if the context doesn't contain
    the answer, rather than filling gaps from general knowledge - critical
    for an insurance tool, where a wrong but confident answer is worse
    than no answer.
  - If the retrieval-quality-control gate (reranker.py) flagged low
    confidence, we still show the best-effort chunks but add an explicit
    instruction to be extra conservative / hedge clearly.
"""

from __future__ import annotations
from src.retrieval.reranker import RerankedChunk


SYSTEM_INSTRUCTIONS = """You are an insurance policy assistant. You answer questions using ONLY the provided document excerpts below - never your own general knowledge about insurance, since policy terms vary enormously between providers and plans.

Rules:
1. Answer only from the excerpts provided. If they don't contain enough information to answer, say so plainly - do not guess or generalize from typical insurance knowledge.
2. Cite the excerpt(s) you used with their tags, e.g. "...covered under your policy [S2]."
3. If dollar amounts, percentages, or numeric limits are involved, quote them exactly as written in the excerpts - do not round or approximate.
4. Keep answers concise and directly useful - lead with the answer, then explain.
5. If an image was attached to the question, use its description as part of the context for what the user is asking about."""


def build_prompt(
    question: str,
    chunks: list[RerankedChunk],
    is_confident: bool,
    image_description: str | None = None,
) -> str:
    excerpt_blocks = []
    for i, c in enumerate(chunks, start=1):
        loc = f"page {c.metadata.get('page_number', '?')}"
        section = c.metadata.get("section_title", "")
        header = f"[S{i}] ({loc}{', ' + section if section else ''})"
        excerpt_blocks.append(f"{header}\n{c.text}")

    excerpts_text = "\n\n".join(excerpt_blocks) if excerpt_blocks else "(no relevant excerpts found)"

    confidence_note = (
        ""
        if is_confident
        else "\n\nNOTE: Retrieval confidence for this question was LOW - the excerpts "
             "below may not fully answer it. Be explicit about uncertainty and clearly "
             "state if the document doesn't seem to cover this."
    )

    image_note = (
        f"\n\nThe user also attached an image. Here is what it shows: {image_description}"
        if image_description else ""
    )

    prompt = f"""{SYSTEM_INSTRUCTIONS}

DOCUMENT EXCERPTS:
{excerpts_text}
{confidence_note}
USER QUESTION: {question}{image_note}

ANSWER:"""
    return prompt
