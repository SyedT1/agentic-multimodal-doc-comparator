"""
RAG Pipeline — wires everything together.
Used by the Streamlit chat tab.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .chunker import Chunk, chunk_document
from .vector_store import VectorStore
from .graph_builder import GraphBuilder
from .groq_chat import GroqGraphChat


@dataclass
class PipelineState:
    """Holds the built RAG state after ingestion."""
    doc1_chunks: List[Chunk] = field(default_factory=list)
    doc2_chunks: List[Chunk] = field(default_factory=list)
    vector_store: Optional[VectorStore] = None
    graph_builder: Optional[GraphBuilder] = None
    is_ready: bool = False
    stats: Dict[str, Any] = field(default_factory=dict)


class GraphRAGPipeline:
    """
    End-to-end Graph RAG pipeline.

    Usage:
        pipeline = GraphRAGPipeline(groq_api_key="...")
        state = pipeline.ingest(raw_doc1, raw_doc2)
        answer = pipeline.query("What does doc1 say about climate?", state)
    """

    def __init__(
        self,
        groq_api_key: str,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        top_k_vector: int = 5,
        graph_hops: int = 2,
        graph_max_nodes: int = 10,
    ):
        self.groq_api_key = groq_api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_vector = top_k_vector
        self.graph_hops = graph_hops
        self.graph_max_nodes = graph_max_nodes

        self._chat: Optional[GroqGraphChat] = None

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, raw_doc1, raw_doc2) -> PipelineState:
        """
        Process both documents: chunk → embed → store → build graph.
        Returns a PipelineState that should be stored in st.session_state.
        """
        state = PipelineState()

        # 1. Chunk
        state.doc1_chunks = chunk_document(
            raw_doc1, "doc1", self.chunk_size, self.chunk_overlap
        )
        state.doc2_chunks = chunk_document(
            raw_doc2, "doc2", self.chunk_size, self.chunk_overlap
        )

        # 2. Vector store
        state.vector_store = VectorStore()
        state.vector_store.add_chunks(state.doc1_chunks)
        state.vector_store.add_chunks(state.doc2_chunks)

        # 3. Knowledge graph
        state.graph_builder = GraphBuilder()
        state.graph_builder.build(state.doc1_chunks, state.doc2_chunks)

        # 4. Stats
        graph_stats = state.graph_builder.get_stats()
        state.stats = {
            "doc1_chunks": len(state.doc1_chunks),
            "doc2_chunks": len(state.doc2_chunks),
            "total_vectors": state.vector_store.count(),
            **graph_stats,
        }
        state.is_ready = True

        # 5. Fresh chat session
        self._chat = GroqGraphChat(api_key=self.groq_api_key)

        return state

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        user_query: str,
        state: PipelineState,
        stream: bool = True,
    ):
        """
        Retrieve relevant context via vector + graph search,
        then pass to Groq for generation.
        """
        if not state.is_ready:
            raise RuntimeError("Pipeline not ready. Call ingest() first.")

        # Step 1: Vector search (both docs)
        seed_chunks = state.vector_store.search(
            user_query, n_results=self.top_k_vector
        )

        # Step 2: Graph expansion
        retrieved_nodes = state.graph_builder.retrieve(
            query=user_query,
            seed_chunks=seed_chunks,
            hops=self.graph_hops,
            max_nodes=self.graph_max_nodes,
        )

        # Fallback: if graph expansion returned nothing, use raw vector results
        if not retrieved_nodes:
            retrieved_nodes = [
                {
                    "chunk_id": f"{s['doc_id']}_chunk_{s['chunk_index']}",
                    "text": s["text"],
                    "doc_id": s["doc_id"],
                    "section": s.get("section", ""),
                    "relevance": s["score"],
                }
                for s in seed_chunks
            ]

        # Step 3: Generate answer via Groq
        return self._chat.chat(
            user_query=user_query,
            retrieved_nodes=retrieved_nodes,
            stream=stream,
        )

    def reset_chat(self) -> None:
        """Clear conversation history (keep the indexed data)."""
        if self._chat:
            self._chat.reset()