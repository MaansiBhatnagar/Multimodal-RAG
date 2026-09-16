"""
src/ingestion/image_processor.py
==================================
Stage 1b of the pipeline: MULTIMODAL NORMALIZATION.

This is the core of "multimodal RAG" on the ingestion side. Every
`RawElement` of type "image" or "scanned_page" carries a PIL image but no
text yet. We need text to embed and index it (our vector store indexes
text embeddings - see the design note in indexing/embeddings.py for why we
chose the text-unification approach over raw CLIP image embeddings).

Two strategies, chosen automatically per element:

  1. Vision captioning (Gemini 1.5 Flash) - default. Produces a rich,
     descriptive caption. For tables/diagrams it transcribes structure;
     for photos it describes content. This is far more useful for
     retrieval than raw OCR because it captures *meaning*, not just
     characters on the page.

  2. OCR (Tesseract) - free, local, offline fallback used only if the
     Gemini call fails (e.g. rate limit) or is unavailable, so ingestion
     never hard-fails just because a vision API hiccupped.
"""

from __future__ import annotations
from PIL import Image
import pytesseract

from src.ingestion.loader import RawElement
from src.generation.llm import LLMClient
from utils.logger import get_logger

logger = get_logger("ingestion.image_processor")


class ImageProcessor:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def process(self, elements: list[RawElement]) -> list[RawElement]:
        """Mutates image/scanned_page elements in place, filling in
        `.content` with a caption or OCR text so downstream chunking can
        treat every element as text uniformly."""
        for el in elements:
            if el.element_type not in ("image", "scanned_page"):
                continue
            el.content = self._describe(el.image, is_full_page=(el.element_type == "scanned_page"))
        return elements

    def _describe(self, image: Image.Image, is_full_page: bool) -> str:
        hint = (
            "This is a full scanned page from an insurance policy document."
            if is_full_page
            else "This is an image embedded inside an insurance policy document "
                 "(could be a table, chart, diagram, logo, or photo)."
        )
        try:
            caption = self.llm.caption_image(image, context_hint=hint)
            if caption:
                return caption
        except Exception as e:
            logger.warning(f"Vision captioning failed, falling back to OCR: {e}")

        # Fallback: local OCR, free and offline.
        try:
            ocr_text = pytesseract.image_to_string(image).strip()
            if ocr_text:
                return f"[OCR extracted text]\n{ocr_text}"
        except Exception as e:
            logger.warning(f"OCR fallback also failed: {e}")

        return "[Image could not be processed - no caption or OCR text available]"
