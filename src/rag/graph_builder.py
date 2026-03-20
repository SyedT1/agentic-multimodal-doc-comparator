"""
Graph RAG — Knowledge Graph Builder
Builds a NetworkX graph where:
  - Nodes  = chunks (from doc1 & doc2)
  - Edges  = relationships between chunks:
      * sequential   : consecutive chunks in same document
      * same_section : chunks sharing the same heading/section
      * cross_similar: high cosine similarity between doc1 chunk & doc2 chunk
      * entity_link  : chunks sharing important noun phrases (entities)
"""
import re
import networkx as nx
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .chunker import Chunk


_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_CROSS_SIM_THRESHOLD = 0.55   # min similarity to create a cross-doc edge
_ENTITY_MIN_LEN = 4           # min characters for an entity term


def _extract_noun_phrases(text: str) -> set:
    """
    Lightweight noun phrase extraction via regex patterns.
    No spacy dependency — works in constrained environments.
    """
    # Capitalised multi-word phrases and key technical terms
    patterns = [
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',   # "Neural Network", "New York"
        r'\b[A-Z]{2,}\b',                          # acronyms: "RAG", "LLM"
        r'\b\w{5,}\b',                             # any long word (catch technical terms)
    ]
    entities = set()
    for pat in patterns:
        found = re.findall(pat, text)
        entities.update(f.strip().lower() for f in found if len(f) >= _ENTITY_MIN_LEN)
    # Remove very common stopwords
    stopwords = {'which', 'these', 'those', 'their', 'there', 'where', 'about',
                 'would', 'could', 'should', 'other', 'being', 'using', 'having'}
    return entities - stopwords


class GraphBuilder:
    """
    Builds and queries a knowledge graph from doc chunks.
    """

    def __init__(self):
        self._model = SentenceTransformer(_EMBED_MODEL_NAME)
        self.graph: nx.Graph = nx.Graph()
        self._chunk_map: Dict[str, Chunk] = {}   # chunk_id -> Chunk

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, doc1_chunks: List[Chunk], doc2_chunks: List[Chunk]) -> nx.Graph:
        """
        Full graph construction pipeline.
        Returns the built NetworkX graph.
        """
        self.graph = nx.Graph()
        self._chunk_map = {}

        all_chunks = doc1_chunks + doc2_chunks

        # 1. Add nodes
        for chunk in all_chunks:
            self._chunk_map[chunk.chunk_id] = chunk
            self.graph.add_node(
                chunk.chunk_id,
                text=chunk.text[:200],   # store snippet
                doc_id=chunk.doc_id,
                section=chunk.section,
                chunk_index=chunk.chunk_index,
                entities=list(_extract_noun_phrases(chunk.text)),
            )

        # 2. Sequential edges (within same doc)
        self._add_sequential_edges(doc1_chunks)
        self._add_sequential_edges(doc2_chunks)

        # 3. Same-section edges
        self._add_section_edges(all_chunks)

        # 4. Cross-document similarity edges
        self._add_cross_similarity_edges(doc1_chunks, doc2_chunks)

        # 5. Entity co-occurrence edges
        self._add_entity_edges(all_chunks)

        return self.graph

    def _add_sequential_edges(self, chunks: List[Chunk]) -> None:
        sorted_chunks = sorted(chunks, key=lambda c: c.chunk_index)
        for i in range(len(sorted_chunks) - 1):
            a, b = sorted_chunks[i], sorted_chunks[i + 1]
            self.graph.add_edge(
                a.chunk_id, b.chunk_id,
                relation="sequential",
                weight=0.9,
            )

    def _add_section_edges(self, chunks: List[Chunk]) -> None:
        section_map: Dict[str, List[str]] = {}
        for chunk in chunks:
            key = f"{chunk.doc_id}::{chunk.section}"
            section_map.setdefault(key, []).append(chunk.chunk_id)

        for ids in section_map.values():
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    if not self.graph.has_edge(ids[i], ids[j]):
                        self.graph.add_edge(
                            ids[i], ids[j],
                            relation="same_section",
                            weight=0.6,
                        )

    def _add_cross_similarity_edges(
        self, doc1_chunks: List[Chunk], doc2_chunks: List[Chunk]
    ) -> None:
        if not doc1_chunks or not doc2_chunks:
            return

        texts1 = [c.text for c in doc1_chunks]
        texts2 = [c.text for c in doc2_chunks]

        emb1 = self._model.encode(texts1, batch_size=32, show_progress_bar=False)
        emb2 = self._model.encode(texts2, batch_size=32, show_progress_bar=False)

        sim_matrix = cosine_similarity(emb1, emb2)

        for i, c1 in enumerate(doc1_chunks):
            for j, c2 in enumerate(doc2_chunks):
                sim = float(sim_matrix[i, j])
                if sim >= _CROSS_SIM_THRESHOLD:
                    self.graph.add_edge(
                        c1.chunk_id, c2.chunk_id,
                        relation="cross_similar",
                        weight=round(sim, 4),
                        similarity=round(sim, 4),
                    )

    def _add_entity_edges(self, chunks: List[Chunk]) -> None:
        entity_to_chunks: Dict[str, List[str]] = {}
        for chunk in chunks:
            entities = _extract_noun_phrases(chunk.text)
            for ent in entities:
                entity_to_chunks.setdefault(ent, []).append(chunk.chunk_id)

        for ent, ids in entity_to_chunks.items():
            if len(ids) < 2:
                continue
            # Only connect cross-doc pairs to avoid too many same-doc entity edges
            doc_ids = {self._chunk_map[cid].doc_id: cid for cid in ids}
            if len(doc_ids) >= 2:
                cids = list(doc_ids.values())
                for i in range(len(cids)):
                    for j in range(i + 1, len(cids)):
                        if not self.graph.has_edge(cids[i], cids[j]):
                            self.graph.add_edge(
                                cids[i], cids[j],
                                relation="entity_link",
                                entity=ent,
                                weight=0.5,
                            )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        seed_chunks: List[Dict[str, Any]],   # from VectorStore.search()
        hops: int = 2,
        max_nodes: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Graph-aware retrieval:
        1. Start from seed chunk nodes (vector search results)
        2. Expand via BFS up to `hops` hops, prioritising high-weight edges
        3. Return unique chunks from both docs, ranked by relevance
        """
        visited = set()
        result_nodes = []

        seed_ids = [
            f"{s['doc_id']}_chunk_{s['chunk_index']}"
            for s in seed_chunks
            if s.get('chunk_index') is not None
        ]

        # BFS queue: (node_id, remaining_hops, accumulated_weight)
        queue = [(nid, hops, 1.0) for nid in seed_ids if nid in self.graph]

        while queue and len(result_nodes) < max_nodes:
            node_id, remaining, acc_weight = queue.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)

            chunk = self._chunk_map.get(node_id)
            if chunk:
                result_nodes.append({
                    "chunk_id": node_id,
                    "text": chunk.text,
                    "doc_id": chunk.doc_id,
                    "section": chunk.section,
                    "relevance": round(acc_weight, 4),
                })

            if remaining > 0:
                neighbors = sorted(
                    self.graph[node_id].items(),
                    key=lambda x: x[1].get("weight", 0),
                    reverse=True,
                )
                for neighbor_id, edge_data in neighbors[:4]:   # top-4 neighbours
                    if neighbor_id not in visited:
                        queue.append((
                            neighbor_id,
                            remaining - 1,
                            acc_weight * edge_data.get("weight", 0.5),
                        ))

        # Sort by relevance
        result_nodes.sort(key=lambda x: x["relevance"], reverse=True)
        return result_nodes[:max_nodes]

    def get_stats(self) -> Dict[str, Any]:
        edge_types = {}
        for _, _, data in self.graph.edges(data=True):
            rel = data.get("relation", "unknown")
            edge_types[rel] = edge_types.get(rel, 0) + 1
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "edge_types": edge_types,
        }