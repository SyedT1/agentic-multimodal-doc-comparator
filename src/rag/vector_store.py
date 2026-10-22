"""Isolated Chroma collections plus BM25S, combined with reciprocal rank fusion."""
import uuid
import config
import bm25s
import chromadb
import numpy as np
from chromadb.config import Settings
from utils.model_runtime import get_text_model


class VectorStore:
    def __init__(self, persist_dir=None, embedding_model=None):
        self._model = embedding_model or get_text_model()
        settings = Settings(anonymized_telemetry=False)
        self._client = (chromadb.PersistentClient(path=persist_dir, settings=settings) if persist_dir
                        else chromadb.EphemeralClient(settings=settings))
        self.collection_name = "comparison_" + uuid.uuid4().hex
        self._collection = self._client.create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"})
        self.chunks = []
        self.embeddings = np.empty((0, 0))
        self._lexical = None

    def add_chunks(self, chunks):
        if not chunks:
            return
        vectors = np.asarray(self._model.encode([c.text for c in chunks], normalize_embeddings=True,
                                               show_progress_bar=False, batch_size=32))
        self._collection.upsert(ids=[c.chunk_id for c in chunks], embeddings=vectors.tolist(),
            documents=[c.text for c in chunks], metadatas=[{
                "doc_id": c.doc_id, "chunk_index": c.chunk_index, "page": c.page,
                "section": c.section, "filename": c.metadata.get("filename", ""),
                "kind": c.metadata.get("kind", "text"),
            } for c in chunks])
        self.chunks.extend(chunks)
        self.embeddings = vectors if not self.embeddings.size else np.vstack((self.embeddings, vectors))
        tokens = bm25s.tokenize([c.text for c in self.chunks], stopwords="en", show_progress=False)
        self._lexical = bm25s.BM25()
        self._lexical.index(tokens, show_progress=False)

    def _hit(self, chunk, score, **extra):
        return {"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "text": chunk.text,
                "chunk_index": chunk.chunk_index, "page": chunk.page, "section": chunk.section,
                "filename": chunk.metadata.get("filename", ""), "score": float(score), **extra}

    def search(self, query, n_results=5, doc_filter=None):
        if not self.chunks or n_results < 1:
            return []
        count = sum(c.doc_id == doc_filter for c in self.chunks) if doc_filter else len(self.chunks)
        if not count:
            return []
        k = min(max(n_results * 3, 12), count)
        query_embedding = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False).tolist()
        dense = self._collection.query(query_embeddings=query_embedding, n_results=k,
            where={"doc_id": doc_filter} if doc_filter else None, include=["distances"])
        chunk_map = {c.chunk_id: c for c in self.chunks}
        scores, relevance = {}, {}
        for rank, (cid, distance) in enumerate(zip(dense["ids"][0], dense["distances"][0])):
            scores[cid] = 1 / (60 + rank + 1)
            relevance[cid] = max(0.0, min(1.0, 1 - distance))
        tokenized_query = bm25s.tokenize([query], stopwords="en", show_progress=False)
        indices, lexical_scores = self._lexical.retrieve(tokenized_query, k=len(self.chunks), show_progress=False)
        rank = 0
        for idx, value in zip(indices[0], lexical_scores[0]):
            chunk = self.chunks[int(idx)]
            if value <= 0 or (doc_filter and chunk.doc_id != doc_filter):
                continue
            cid = chunk.chunk_id
            scores[cid] = scores.get(cid, 0) + 1 / (60 + rank + 1)
            rank += 1
            if rank >= k:
                break
        ranked = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:n_results]
        return [self._hit(chunk_map[cid], relevance.get(cid, 0), fusion_score=scores[cid]) for cid in ranked]

    def count(self):
        return len(self.chunks)

    def get_all_chunks_for_doc(self, doc_id):
        return [self._hit(c, 1) for c in self.chunks if c.doc_id == doc_id]

    def clear(self):
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})
        self.chunks = []
        self.embeddings = np.empty((0, 0))
        self._lexical = None

    def close(self):
        self._client.delete_collection(self.collection_name)
