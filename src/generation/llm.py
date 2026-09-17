"""
src/generation/llm.py
======================
Thin wrapper around the two free-tier LLM providers used in this project:

  - Gemini 3.1 Flash-Lite: natively multimodal (accepts images directly), used
    for (a) captioning images/diagrams/tables extracted during ingestion,
    and (b) final grounded answer generation.
  - Groq (Llama 3.1 8B Instant): extremely fast text-only inference, used
    for cheap/fast query-transformation calls (HyDE, multi-query rewriting)
    so we don't burn Gemini quota on auxiliary steps, and as a fallback if
    Gemini is rate-limited.

Keeping both providers behind one interface (`LLMClient`) means the rest of
the pipeline never needs to know which provider is actually answering.
"""

from __future__ import annotations
import time
from typing import Optional

import google.generativeai as genai
from groq import Groq
from PIL import Image

import config
from utils.logger import get_logger

logger = get_logger("llm")


class LLMClient:
    def __init__(self):
        if config.GEMINI_API_KEY:
            genai.configure(api_key=config.GEMINI_API_KEY)
            self._gemini = genai.GenerativeModel(config.GEMINI_MODEL)
        else:
            self._gemini = None
            logger.warning("GEMINI_API_KEY not set - vision + generation will fail.")

        self._groq = Groq(api_key=config.GROQ_API_KEY) if config.GROQ_API_KEY else None
        if not self._groq:
            logger.warning("GROQ_API_KEY not set - fast text calls will fall back to Gemini.")

    # ------------------------------------------------------------------
    # Text-only generation (fast path via Groq, used for query transforms)
    # ------------------------------------------------------------------
    def fast_text(self, prompt: str, temperature: float = 0.3, max_tokens: int = 300) -> str:
        """
        Cheap/fast text completion. Prefers Groq (very low latency, good
        for the 2-4 auxiliary LLM calls query transformation needs per
        user question). Falls back to Gemini if Groq isn't configured.
        """
        if self._groq:
            try:
                resp = self._groq.chat.completions.create(
                    model=config.GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"Groq call failed ({e}), falling back to Gemini.")

        return self._gemini_text(prompt, temperature, max_tokens)

    # ------------------------------------------------------------------
    # Grounded final-answer generation (always Gemini - handles citations
    # and can also take an optional user-uploaded image in the same call)
    # ------------------------------------------------------------------
    def generate_answer(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        temperature: float = config.GENERATION_TEMPERATURE,
    ) -> str:
        if not self._gemini:
            return "⚠️ GEMINI_API_KEY is not configured - cannot generate an answer."

        contents = [prompt] if image is None else [prompt, image]
        return self._call_gemini(contents, temperature)

    # ------------------------------------------------------------------
    # Vision captioning (ingestion time - describing diagrams/tables/photos
    # extracted from the uploaded document, and user-uploaded claim photos)
    # ------------------------------------------------------------------
    def caption_image(self, image: Image.Image, context_hint: str = "") -> str:
        if not self._gemini:
            return ""
        prompt = (
            "You are helping build a searchable index of a document. "
            "Describe this image factually and specifically so someone "
            "could find it via a text search later. If it is a table, "
            "transcribe it as a markdown table. If it is a chart/diagram, "
            "describe every labeled value and relationship. If it is a "
            "photo (e.g. of damage, a receipt, an ID), describe exactly "
            "what is visible - do not speculate beyond what's shown.\n"
            f"{('Context: ' + context_hint) if context_hint else ''}"
        )
        return self._call_gemini([prompt, image], temperature=0.1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _gemini_text(self, prompt: str, temperature: float, max_tokens: int) -> str:
        if not self._gemini:
            return ""
        return self._call_gemini(
            [prompt], temperature, max_output_tokens=max_tokens
        )

    def _call_gemini(
        self,
        contents: list,
        temperature: float,
        max_output_tokens: int = config.MAX_OUTPUT_TOKENS,
        retries: int = 2,
    ) -> str:
        gen_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        for attempt in range(retries + 1):
            try:
                logger.info(f"Sending request to Gemini (attempt {attempt+1}, timeout=30s)...")
                resp = self._gemini.generate_content(
                    contents,
                    generation_config=gen_config,
                    request_options={"timeout": 30},  # hard cap - a hung
                    # request with no timeout is worse than a failed one we
                    # can retry, since it blocks the whole Streamlit app
                    # with zero visible feedback.
                )
                logger.info("Gemini response received.")

                # A response can come back "successfully" (no exception) but
                # with EMPTY content if Gemini's safety/recitation filter
                # blocked it - this happens most often when the retrieved
                # context contains large verbatim excerpts of well-known
                # text (e.g. famous papers/books Gemini was itself trained
                # on) and it declines to reproduce them closely. Detect this
                # explicitly instead of silently returning "".
                if not resp.candidates:
                    reason = getattr(resp.prompt_feedback, "block_reason", "unknown")
                    logger.warning(f"Gemini returned no candidates. Prompt feedback: {reason}")
                    return (
                        "⚠️ The model declined to answer this one (no response candidates - "
                        f"reason: {reason}). Try rephrasing the question."
                    )

                candidate = resp.candidates[0]
                finish_reason = getattr(candidate, "finish_reason", None)
                text = (resp.text or "").strip() if candidate.content.parts else ""

                if not text:
                    logger.warning(f"Gemini returned empty text. finish_reason={finish_reason}")
                    if str(finish_reason) in ("2", "RECITATION"):
                        return (
                            "⚠️ The model blocked this answer because the retrieved passages "
                            "closely match well-known published text (a 'recitation' safety "
                            "filter) - this can happen with famous papers/books. Try asking "
                            "the question in a way that asks for an explanation/summary "
                            "rather than exact wording, e.g. 'summarize how...' instead of "
                            "'what does it say about...'."
                        )
                    elif str(finish_reason) in ("3", "SAFETY"):
                        return "⚠️ The model's safety filter blocked this response. Try rephrasing the question."
                    else:
                        # Empty for some other reason (e.g. MAX_TOKENS with no
                        # content yet) - worth a retry.
                        raise ValueError(f"Empty response, finish_reason={finish_reason}")

                return text
            except Exception as e:
                logger.warning(f"Gemini call failed (attempt {attempt+1}): {e}")
                time.sleep(1.5 * (attempt + 1))
        return "⚠️ Generation failed after retries - the API may be rate-limited. Try again shortly."


# Singleton - Streamlit reruns the script often, so avoid re-initializing
# the SDK clients on every rerun by caching this at the app layer (see app.py).
def get_llm_client() -> LLMClient:
    return LLMClient()


