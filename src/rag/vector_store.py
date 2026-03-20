"""
Vector Store using ChromaDB (in-memory, HF Spaces compatible)
Stores and retrieves chunks from both documents via semantic search.
"""
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import hashlib

from .chunker import Chunk


_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # fast, small, works great


class VectorStore:
    """
    Wraps ChromaDB with a SentenceTransformer embedding function.
    Collection name: 'doc_chunks' — shared for both documents.
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self._model = SentenceTransformer(_EMBED_MODEL_NAME)

        if persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            self._client = chromadb.EphemeralClient()

        self._collection = self._client.get_or_create_collection(
            name="doc_chunks",
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Embed and upsert chunks into the collection."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self._model.encode(texts, batch_size=32, show_progress_bar=False).tolist()

        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "doc_id": c.doc_id,
                "chunk_index": c.chunk_index,
                "section": c.section,
                "page": c.page,
                **{k: str(v) for k, v in c.metadata.items()},
            }
            for c in chunks
        ]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def clear(self) -> None:
        """Remove all chunks (useful for re-ingestion)."""
        self._client.delete_collection("doc_chunks")
        self._collection = self._client.get_or_create_collection(
            name="doc_chunks",
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        n_results: int = 5,
        doc_filter: Optional[str] = None,   # "doc1" | "doc2" | None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over stored chunks.
        Returns list of dicts with keys: text, doc_id, section, score.
        """
        query_embedding = self._model.encode([query]).tolist()

        where = {"doc_id": doc_filter} if doc_filter else None

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, self._collection.count() or 1),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text": text,
                "doc_id": meta.get("doc_id"),
                "section": meta.get("section", ""),
                "chunk_index": meta.get("chunk_index", -1),
                "score": round(1 - dist, 4),   # cosine similarity
            })

        return hits

    def count(self) -> int:
        return self._collection.count()

    def get_all_chunks_for_doc(self, doc_id: str) -> List[Dict[str, Any]]:
        """Retrieve all stored chunks for a given document."""
        results = self._collection.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas"],
        )
        items = []
        for text, meta in zip(results["documents"], results["metadatas"]):
            items.append({"text": text, **meta})
        # Sort by chunk_index
        items.sort(key=lambda x: int(x.get("chunk_index", 0)))
        return items