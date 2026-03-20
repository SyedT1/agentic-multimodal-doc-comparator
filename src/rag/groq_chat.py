"""
Groq Chat with Graph RAG context injection.
Uses llama-3.3-70b-versatile (fast + smart) via Groq API.
"""
import os
from typing import List, Dict, Any, Generator
from groq import Groq


_DEFAULT_MODEL = "llama-3.3-70b-versatile"
_MAX_CONTEXT_CHARS = 6000   # stay within context window safely


def _build_context(retrieved_nodes: List[Dict[str, Any]]) -> str:
    """
    Format retrieved graph nodes into a clean context block for the LLM.
    Groups by document for clarity.
    """
    doc1_nodes = [n for n in retrieved_nodes if n.get("doc_id") == "doc1"]
    doc2_nodes = [n for n in retrieved_nodes if n.get("doc_id") == "doc2"]

    parts = []

    if doc1_nodes:
        parts.append("### Relevant passages from Document 1:")
        for node in doc1_nodes:
            sec = f" [{node['section']}]" if node.get("section") else ""
            parts.append(f"- {node['text'][:500]}{sec}")

    if doc2_nodes:
        parts.append("\n### Relevant passages from Document 2:")
        for node in doc2_nodes:
            sec = f" [{node['section']}]" if node.get("section") else ""
            parts.append(f"- {node['text'][:500]}{sec}")

    context = "\n".join(parts)
    return context[:_MAX_CONTEXT_CHARS]


_SYSTEM_PROMPT = """You are an expert document analyst assistant with access to two documents that have been processed, chunked, and indexed using a Knowledge Graph RAG system.

You will be given:
1. CONTEXT: Relevant passages retrieved from both documents via graph-enhanced semantic search
2. USER QUESTION: What the user wants to know

Your job:
- Answer using ONLY the provided context
- Clearly indicate which document (Document 1 or Document 2) information comes from
- If comparing both documents, highlight similarities and differences
- If the context doesn't contain the answer, say so honestly
- Be concise, accurate, and helpful
"""


class GroqGraphChat:
    """
    Stateful chat session backed by Groq API + GraphRAG context injection.
    """

    def __init__(self, api_key: str, model: str = _DEFAULT_MODEL):
        self._client = Groq(api_key=api_key)
        self._model = model
        self._history: List[Dict[str, str]] = []

    def reset(self) -> None:
        self._history = []

    def chat(
        self,
        user_query: str,
        retrieved_nodes: List[Dict[str, Any]],
        stream: bool = True,
    ) -> str | Generator:
        """
        Send a message with GraphRAG context and get a response.

        Args:
            user_query: The user's question
            retrieved_nodes: Chunks from GraphBuilder.retrieve()
            stream: If True, returns a generator for streaming UI

        Returns:
            Full response string (if stream=False) or generator (if stream=True)
        """
        context = _build_context(retrieved_nodes)

        # Build the user turn with injected context
        augmented_user_message = f"""<context>
{context}
</context>

<question>
{user_query}
</question>"""

        # Append to history
        self._history.append({"role": "user", "content": augmented_user_message})

        messages = [{"role": "system", "content": _SYSTEM_PROMPT}] + self._history

        if stream:
            return self._stream_response(messages)
        else:
            return self._full_response(messages)

    def _full_response(self, messages: List[Dict]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=1024,
            temperature=0.3,
        )
        answer = response.choices[0].message.content
        self._history.append({"role": "assistant", "content": answer})
        return answer

    def _stream_response(self, messages: List[Dict]) -> Generator:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=1024,
            temperature=0.3,
            stream=True,
        )
        full_response = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full_response += delta
            yield delta
        self._history.append({"role": "assistant", "content": full_response})