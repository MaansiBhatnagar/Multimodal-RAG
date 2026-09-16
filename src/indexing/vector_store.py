"""
src/indexing/vector_store.py
==============================
Stage 4: INDEXING.

Maintains TWO parallel indexes over the same chunk set, because dense and
sparse retrieval fail in complementary ways:

  - Dense (Chroma, cosine similarity over sentence-transformer embeddings):
    great at "meaning" matches - e.g. a query about "car accident" retrieves
    a chunk about "collision damage" even with zero shared words. Weak on
    exact identifiers - a policy number, a specific dollar figure, or rare
    jargon can get diluted in the embedding.

  - Sparse (BM25, classic term-frequency keyword ranking): great at exact
    term/number matches. Weak on paraphrase / synonym queries.

Both are combined at query time via Reciprocal Rank Fusion in
retrieval/hybrid_retriever.py. This file just owns building + querying each
index independently.

Each uploaded document gets its own Chroma collection (named by a hash of
the filename+size) so multiple users/documents don't bleed into each
other's retrieval results in a shared deployment.
"""

from __future__ import annotations
import pickle
import os
from typing import Optional

import chromadb
from rank_bm25 import BM25Okapi

import config
from utils.logger import get_logger

logger = get_logger("indexing.vector_store")


class VectorStore:
    def __init__(self, collection_name: str, embedder):
        self.embedder = embedder
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

        # BM25 index lives in memory (rebuilt from the doc's chunk store on
        # load) + persisted to disk via pickle so it survives a Streamlit
        # session/app restart alongside the Chroma collection.
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_corpus_ids: list[str] = []
        self._bm25_path = os.path.join(config.CHROMA_DIR, f"{collection_name}_bm25.pkl")
        self._load_bm25_if_exists()

    # ------------------------------------------------------------------
    def add_chunks(self, enriched_chunks: list[dict]):
        """enriched_chunks: list of {id, text, metadata} from
        metadata_enricher.enrich(). Adds to both the dense (Chroma) and
        sparse (BM25) indexes."""
        if not enriched_chunks:
            return

        ids = [c["id"] for c in enriched_chunks]
        texts = [c["text"] for c in enriched_chunks]
        metadatas = [c["metadata"] for c in enriched_chunks]

        embeddings = self.embedder.encode(texts).tolist()
        self._collection.upsert(
            ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas
        )

        self._rebuild_bm25(ids, texts)
        logger.info(f"Indexed {len(ids)} chunks into collection '{self.collection_name}'.")

    def _rebuild_bm25(self, new_ids: list[str], new_texts: list[str]):
        # Pull everything currently in Chroma so BM25 always reflects the
        # full corpus for this document, not just the latest batch.
        all_data = self._collection.get(include=["documents"])
        all_ids = all_data["ids"]
        all_docs = all_data["documents"]

        tokenized = [doc.lower().split() for doc in all_docs]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_corpus_ids = all_ids
        self._save_bm25()

    def _save_bm25(self):
        with open(self._bm25_path, "wb") as f:
            pickle.dump({"corpus_ids": self._bm25_corpus_ids, "bm25": self._bm25}, f)

    def _load_bm25_if_exists(self):
        if os.path.exists(self._bm25_path):
            with open(self._bm25_path, "rb") as f:
                data = pickle.load(f)
                self._bm25 = data["bm25"]
                self._bm25_corpus_ids = data["corpus_ids"]

    # ------------------------------------------------------------------
    def dense_search(self, query: str, top_k: int = config.DENSE_TOP_K) -> list[dict]:
        query_embedding = self.embedder.encode([query])[0].tolist()
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, max(self._collection.count(), 1)),
            include=["documents", "metadatas", "distances"],
        )
        out = []
        if not results["ids"] or not results["ids"][0]:
            return out
        for i, cid in enumerate(results["ids"][0]):
            out.append({
                "id": cid,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                # Chroma returns cosine *distance*; convert to similarity.
                "score": 1 - results["distances"][0][i],
            })
        return out

    def sparse_search(self, query: str, top_k: int = config.SPARSE_TOP_K) -> list[dict]:
        if self._bm25 is None or not self._bm25_corpus_ids:
            return []
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        ranked = sorted(zip(self._bm25_corpus_ids, scores), key=lambda x: x[1], reverse=True)[:top_k]

        # Fetch text/metadata for the ranked ids from Chroma.
        ids = [r[0] for r in ranked if r[1] > 0]
        if not ids:
            return []
        fetched = self._collection.get(ids=ids, include=["documents", "metadatas"])
        id_to_data = {
            fid: (fetched["documents"][i], fetched["metadatas"][i])
            for i, fid in enumerate(fetched["ids"])
        }
        out = []
        for cid, score in ranked:
            if cid not in id_to_data or score <= 0:
                continue
            text, meta = id_to_data[cid]
            out.append({"id": cid, "text": text, "metadata": meta, "score": float(score)})
        return out

    def count(self) -> int:
        return self._collection.count()

    def delete_collection(self):
        self._client.delete_collection(self.collection_name)
        if os.path.exists(self._bm25_path):
            os.remove(self._bm25_path)
