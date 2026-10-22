"""Bounded LangGraph agent: plan -> hybrid retrieval -> graph expansion -> grade -> retry/answer."""
from dataclasses import dataclass, field
from typing import Any, TypedDict
import config
import numpy as np
import re
from langgraph.graph import StateGraph, START, END
from .chunker import chunk_document
from .vector_store import VectorStore
from .graph_builder import GraphBuilder
from .groq_chat import GroqGraphChat, source_id
from utils.model_runtime import get_reranker

@dataclass
class PipelineState:
    doc1_chunks: list = field(default_factory=list)
    doc2_chunks: list = field(default_factory=list)
    vector_store: Any = None
    graph_builder: Any = None
    is_ready: bool = False
    stats: dict = field(default_factory=dict)

class QueryState(TypedDict, total=False):
    question: str
    query: str
    scope: str
    attempts: int
    sources: list
    sufficient: bool
    trace: list

class GraphRAGPipeline:
    def __init__(self, groq_api_key="", chunk_size=220, chunk_overlap=40, top_k_vector=5,
                 graph_hops=1, graph_max_nodes=12, rerank=True, chat_client=None):
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk size")
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap
        self.top_k_vector, self.graph_hops, self.graph_max_nodes = top_k_vector, graph_hops, graph_max_nodes
        self.rerank = rerank
        self._chat = chat_client or (GroqGraphChat(groq_api_key) if groq_api_key else None)
        self.last_sources, self.last_trace = [], []

    def ingest(self, raw_doc1, raw_doc2):
        state = PipelineState()
        state.doc1_chunks = chunk_document(raw_doc1, "doc1", self.chunk_size, self.chunk_overlap)
        state.doc2_chunks = chunk_document(raw_doc2, "doc2", self.chunk_size, self.chunk_overlap)
        if not state.doc1_chunks or not state.doc2_chunks:
            raise ValueError("Each document needs readable text or tables before building a chat index.")
        try:
            state.vector_store = VectorStore()
            state.vector_store.add_chunks(state.doc1_chunks + state.doc2_chunks)
            state.graph_builder = GraphBuilder()
            state.graph_builder.build(state.doc1_chunks, state.doc2_chunks, state.vector_store.embeddings)
        except Exception:
            if state.vector_store:
                state.vector_store.close()
            raise
        state.stats = {"doc1_chunks": len(state.doc1_chunks), "doc2_chunks": len(state.doc2_chunks),
                       "total_vectors": state.vector_store.count(), **state.graph_builder.get_stats()}
        state.is_ready = True
        return state

    def _retrieve(self, context, query, scope, breadth):
        docs = ("doc1", "doc2") if scope == "both" else (scope,)
        hits = []
        for doc in docs:
            hits.extend(context.vector_store.search(query, n_results=self.top_k_vector * breadth, doc_filter=doc))
        expanded = context.graph_builder.retrieve(query, hits, hops=min(self.graph_hops + breadth - 1, 3),
                                                   max_nodes=self.graph_max_nodes * breadth)
        by_id = {hit["chunk_id"]: hit for hit in expanded if hit["doc_id"] in docs}
        by_id.update({hit["chunk_id"]: hit for hit in hits})
        return list(by_id.values())

    def retrieve(self, user_query, context):
        if not context.is_ready:
            raise ValueError("Build the document index first.")
        if not user_query.strip():
            raise ValueError("Enter a question.")
        reranker = get_reranker() if self.rerank else None

        def plan(state):
            try:
                result = self._chat.plan(state["question"]) if self._chat else {"query": state["question"], "scope": "both"}
                trace = [f"Search question: {result['query']}"]
            except Exception:
                result = {"query": state["question"], "scope": "both"}
                trace = ["Planner unavailable; searched the original question"]
            return {**result, "trace": trace, "attempts": 0}

        def search(state):
            sources = self._retrieve(context, state["query"], state["scope"], state["attempts"] + 1)
            return {"sources": sources, "trace": state["trace"] + [f"Hybrid search + graph: {len(sources)} passages"]}

        def grade(state):
            sources = state["sources"]
            if sources and reranker:
                import torch
                values = reranker.predict([(state["query"], item["text"]) for item in sources],
                                          activation_fn=torch.nn.Sigmoid(), show_progress_bar=False)
                for source, score in zip(sources, np.asarray(values).reshape(-1)):
                    source["rerank_score"] = float(score)
            key = "rerank_score" if reranker else "score"
            sources.sort(key=lambda item: item.get(key, 0), reverse=True)
            selected = []
            if state["scope"] == "both":
                for doc in ("doc1", "doc2"):
                    selected.extend([s for s in sources if s["doc_id"] == doc][:max(1, self.top_k_vector // 2)])
            else:
                selected = sources[:self.top_k_vector]
            # Cross-encoder outputs rank passages; they are not calibrated probabilities.
            # Also recognize explicit query terms in semantically relevant evidence.
            stopwords = {"what", "which", "when", "where", "compare", "changed", "changes", "both",
                         "document", "documents", "project", "about", "does", "this", "that", "with", "from", "have", "were",
                         "the", "and", "for", "are", "how", "much", "between"}
            terms = set(re.findall(r"\b[a-z0-9]{3,}\b", state["query"].lower())) - stopwords
            supported = any(s.get("score", 0) >= 0.15 and
                len(terms & set(re.findall(r"\b[a-z0-9]{3,}\b", s["text"].lower()))) >= max(1,len(terms)*0.25)
                for s in selected)
            sufficient = bool(selected) and (supported or max(s.get(key, 0) for s in selected) >= (0.08 if reranker else 0.2))
            if selected and self._chat and hasattr(self._chat,"grade"):
                try:
                    decision = self._chat.grade(state["question"],selected)
                    if isinstance(decision,bool):
                        sufficient = decision
                except Exception:
                    pass  # Keep the local evidence check if the provider is unavailable.
            return {"sources": selected, "sufficient": sufficient,
                    "trace": state["trace"] + [f"Evidence check: {'ready' if sufficient else 'weak'} ({len(selected)} sources)"]}

        def rewrite(state):
            try:
                result = self._chat.plan(state["question"], retry=True) if self._chat else {"query": state["question"]}
                query = result.get("query", state["query"])
            except Exception:
                query = state["question"]
            return {"query": query, "attempts": state["attempts"] + 1,
                    "trace": state["trace"] + ["Broadened search once because evidence was weak"]}

        graph = StateGraph(QueryState)
        graph.add_node("plan", plan)
        graph.add_node("retrieve", search)
        graph.add_node("grade", grade)
        graph.add_node("rewrite", rewrite)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "retrieve")
        graph.add_edge("retrieve", "grade")
        graph.add_conditional_edges("grade", lambda state: "done" if state["sufficient"] or state["attempts"] >= 1 else "retry",
                                    {"done": END, "retry": "rewrite"})
        graph.add_edge("rewrite", "retrieve")
        result = graph.compile().invoke({"question": user_query, "trace": []}, {"recursion_limit": 12})
        self.last_trace = result["trace"]
        self.last_sources = [{**s, "source_id": source_id(s)} for s in result["sources"]]
        return self.last_sources, result["sufficient"]

    def query(self, user_query, state, stream=True):
        sources, sufficient = self.retrieve(user_query, state)
        if not sufficient:
            answer = "I couldn't find enough relevant evidence in these documents to answer reliably. Try naming a section, term, or value from the documents."
            return iter([answer]) if stream else answer
        if self._chat is None:
            raise ValueError("Add GROQ_API_KEY to .env and restart the app to enable answers.")
        return self._chat.chat(user_query, sources, stream=stream)

    def reset_chat(self):
        self.last_sources, self.last_trace = [], []
        if self._chat:
            self._chat.reset()

    def close(self, state):
        if state and state.vector_store:
            state.vector_store.close()
            state.is_ready = False
