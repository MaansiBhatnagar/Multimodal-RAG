"""
utils/logger.py
================
Tiny shared logger so every module reports progress consistently. Streamlit
reruns the whole script on every interaction, so we also expose a helper to
mirror log lines into a Streamlit UI element (e.g. a "pipeline trace" panel)
which is great for a portfolio demo - recruiters can literally watch the
RAG pipeline think.
"""

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


class TraceCollector:
    """
    Collects human-readable step descriptions during a single query so the
    UI can render "what the pipeline did" transparently (query rewrites,
    retrieval counts, rerank scores, etc). This is purely for
    observability/demo value, not used for control flow.
    """

    def __init__(self):
        self.steps = []

    def add(self, stage: str, detail: str):
        self.steps.append({"stage": stage, "detail": detail})

    def as_list(self):
        return self.steps

    def reset(self):
        self.steps = []
