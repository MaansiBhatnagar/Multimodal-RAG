"""
src/indexing/metadata_enricher.py
===================================
Stage 3: METADATA ENRICHMENT.

Raw chunks only carry page number and source file. Retrieval quality (and
citation quality shown to the user) improves a lot if every chunk also
carries:

  - a unique chunk_id (for dedup / update / delete)
  - a short auto-generated section_title, so the UI can show "Section:
    Collision Coverage - page 12" instead of a bare page number
  - light keyword tags extracted heuristically (dollar figures, section-
    like headers, the word "deductible"/"exclusion"/"claim" etc.) - these
    feed the sparse/BM25 side of hybrid retrieval and let us do metadata
    filtering later (e.g. "only search tables") if extended
  - a cheap heuristic "is_high_value" flag for chunks that look like they
    contain concrete coverage numbers - useful for prioritizing during
    quality control

This stage is deliberately rule-based (regex/heuristics), not another LLM
call - running an LLM over every chunk of every uploaded document would be
slow and burn free-tier quota fast. Cheap heuristics get 90% of the value.
"""

from __future__ import annotations
import re
import hashlib

from src.chunking.semantic_chunker import Chunk
from utils.logger import get_logger

logger = get_logger("indexing.metadata")

_DOLLAR_RE = re.compile(r"\$[\d,]+(?:\.\d{2})?")
_HEADER_RE = re.compile(r"^([A-Z][A-Za-z0-9 /&\-]{3,60})$", re.MULTILINE)
_KEYWORDS = [
    "deductible", "premium", "exclusion", "coverage", "claim", "copay",
    "co-insurance", "liability", "policyholder", "beneficiary", "rider",
    "endorsement", "lapse", "grace period", "renewal", "subrogation",
]


def enrich(chunks: list[Chunk]) -> list[dict]:
    """Returns a list of plain dicts (one per chunk) ready to hand to the
    vector store: {id, text, metadata}. Using plain dicts (not the Chunk
    dataclass) at this boundary keeps indexing.py decoupled from the
    chunking module's internal types."""
    enriched = []
    for c in chunks:
        chunk_id = _make_id(c)
        section_title = _guess_section_title(c.text)
        found_keywords = [kw for kw in _KEYWORDS if kw in c.text.lower()]
        has_dollar_figures = bool(_DOLLAR_RE.search(c.text))

        metadata = {
            "chunk_id": chunk_id,
            "source_file": c.source_file,
            "page_number": c.page_number,
            "element_type": c.element_type,
            "section_title": section_title,
            "keywords": ", ".join(found_keywords),
            "has_dollar_figures": has_dollar_figures,
            "char_count": len(c.text),
        }
        enriched.append({"id": chunk_id, "text": c.text, "metadata": metadata})

    logger.info(f"Enriched {len(enriched)} chunks with metadata.")
    return enriched


def _make_id(c: Chunk) -> str:
    raw = f"{c.source_file}-{c.page_number}-{c.chunk_index}-{c.text[:50]}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _guess_section_title(text: str) -> str:
    """Heuristic: an ALL-CAPS-ish short standalone line near the start of a
    chunk is very often a section heading in insurance docs (e.g.
    'SECTION 4: COLLISION COVERAGE'). Falls back to the first few words."""
    match = _HEADER_RE.search(text[:300])
    if match:
        candidate = match.group(1).strip()
        # avoid matching a normal capitalized sentence start
        if candidate.isupper() or len(candidate.split()) <= 6:
            return candidate
    first_line = text.strip().split("\n")[0]
    words = first_line.split()
    return " ".join(words[:8]) + ("…" if len(words) > 8 else "")
