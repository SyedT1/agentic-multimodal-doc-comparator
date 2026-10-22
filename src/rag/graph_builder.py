"""Bounded document relationships; not an LLM-extracted entity knowledge base."""
import heapq
import re
import networkx as nx
import numpy as np
from utils.model_runtime import get_text_model


def _extract_noun_phrases(text):
    return set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b|\b[A-Z]{2,}\b", text))


class GraphBuilder:
    def __init__(self):
        self.graph = nx.Graph()
        self._chunk_map = {}

    def build(self, doc1_chunks, doc2_chunks, embeddings=None):
        self.graph.clear()
        chunks = doc1_chunks + doc2_chunks
        self._chunk_map = {c.chunk_id: c for c in chunks}
        for c in chunks:
            self.graph.add_node(c.chunk_id, doc_id=c.doc_id, page=c.page, section=c.section,
                                text=c.text[:160], filename=c.metadata.get("filename", ""))
        for document in (doc1_chunks, doc2_chunks):
            for left, right in zip(document, document[1:]):
                self.graph.add_edge(left.chunk_id, right.chunk_id, relation="sequential", weight=0.75)
        if doc1_chunks and doc2_chunks:
            if embeddings is None:
                embeddings = get_text_model().encode([c.text for c in chunks], normalize_embeddings=True,
                                                     show_progress_bar=False)
            matrix = np.asarray(embeddings[:len(doc1_chunks)]) @ np.asarray(embeddings[len(doc1_chunks):]).T
            for i, chunk in enumerate(doc1_chunks):
                for j in np.argsort(matrix[i])[-3:]:
                    similarity = float(matrix[i, j])
                    if similarity >= 0.55:
                        self.graph.add_edge(chunk.chunk_id, doc2_chunks[j].chunk_id,
                                            relation="semantic", weight=min(1.0, similarity))
        entities = {}
        for c in chunks:
            for entity in _extract_noun_phrases(c.text):
                entities.setdefault(entity, []).append(c)
        for matches in entities.values():
            for left, right in zip(matches[:16], matches[1:17]):
                if left.doc_id != right.doc_id and not self.graph.has_edge(left.chunk_id, right.chunk_id):
                    self.graph.add_edge(left.chunk_id, right.chunk_id, relation="shared_term", weight=0.5)
        return self.graph

    def retrieve(self, query, seed_chunks, hops=2, max_nodes=10):
        queue, visited, results = [], set(), []
        for seed in seed_chunks:
            cid = seed.get("chunk_id") or f"{seed['doc_id']}_chunk_{seed['chunk_index']}"
            heapq.heappush(queue, (-max(seed.get("score", 0), 0.01), cid, hops))
        while queue and len(results) < max_nodes:
            negative, cid, remaining = heapq.heappop(queue)
            if cid in visited or cid not in self._chunk_map:
                continue
            visited.add(cid)
            c = self._chunk_map[cid]
            results.append({"chunk_id": cid, "doc_id": c.doc_id, "text": c.text, "page": c.page,
                            "section": c.section, "filename": c.metadata.get("filename", ""),
                            "relevance": -negative, "score": -negative})
            if remaining:
                neighbors = sorted(self.graph[cid].items(), key=lambda pair: pair[1]["weight"], reverse=True)[:4]
                for neighbor, data in neighbors:
                    if neighbor not in visited:
                        heapq.heappush(queue, (negative * data["weight"], neighbor, remaining - 1))
        return results

    def get_stats(self):
        relations = {}
        for _, _, data in self.graph.edges(data=True):
            relation = data["relation"]
            relations[relation] = relations.get(relation, 0) + 1
        return {"nodes": self.graph.number_of_nodes(), "edges": self.graph.number_of_edges(), "edge_types": relations}
