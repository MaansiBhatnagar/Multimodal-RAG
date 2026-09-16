"""
src/indexing/embeddings.py
============================
Wraps a local, free sentence-transformer model for dense embeddings.

Design note - why text-unified multimodal embeddings instead of CLIP:
CLIP gives you a shared image/text embedding space, but its text encoder is
trained on short captions (~77 tokens) and performs noticeably worse than a
dedicated text embedding model on long-form document retrieval (the exact
task we need for policy clauses). Since every image/table in this pipeline
is *already* converted to a rich text description at ingestion time (see
ingestion/image_processor.py), we get multimodal retrieval "for free" by
embedding everything - text, table markdown, and image captions - with one
strong text embedding model. This also means a user's uploaded photo (also
captioned by Gemini before retrieval) lands in the exact same embedding
space as the document content, so image-to-document matching just works
without needing a second index. This is the "Option A" tradeoff explained
to the user: simpler pipeline, one embedding space, still genuinely
multimodal at the input/output level.
"""

from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer

import config
from utils.logger import get_logger

logger = get_logger("indexing.embeddings")


class EmbeddingModel:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Batch-encode a list of strings into L2-normalized embeddings
        (normalization lets us use dot product as cosine similarity, which
        Chroma and our manual similarity calcs both rely on)."""
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0].tolist()
