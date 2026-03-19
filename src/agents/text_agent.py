"""
Text agent for chunking text and generating embeddings.
"""
import numpy as np
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer

from agents.base_agent import BaseAgent
from models.document import DocumentChunk, RawDocument
import config


class TextAgent(BaseAgent):
    """Agent responsible for text chunking and embedding generation."""

    def __init__(self, config_dict: Dict[str, Any] = None):
        super().__init__(config_dict)
        # Load embedding model
        self.model = SentenceTransformer(config.TEXT_EMBEDDING_MODEL)

    def get_agent_name(self) -> str:
        return "TextAgent"

    async def process(self, raw_document: RawDocument) -> Tuple[List[DocumentChunk], np.ndarray]:
        """
        Process raw document text into chunks and embeddings.

        Args:
            raw_document: Raw document with extracted text

        Returns:
            Tuple of (list of DocumentChunks, numpy array of embeddings)
        """
        # Chunk the text
        chunks = self.chunk_text(raw_document.raw_text, raw_document)

        # Generate embeddings
        if chunks:
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.generate_embeddings(chunk_texts)
        else:
            embeddings = np.array([])

        return chunks, embeddings

    def chunk_text(self, text: str, raw_document: RawDocument) -> List[DocumentChunk]:
        """
        Split text into chunks with overlap.

        Args:
            text: Text to chunk
            raw_document: Original document for metadata

        Returns:
            List of DocumentChunk objects
        """
        if not text or not text.strip():
            return []

        chunks = []

        # Simple character-based chunking (approximate token-based chunking)
        # Approximate: 1 token ~= 4 characters
        char_chunk_size = config.TEXT_CHUNK_SIZE * 4
        char_overlap = config.TEXT_CHUNK_OVERLAP * 4

        text_length = len(text)
        start = 0
        chunk_idx = 0

        while start < text_length:
            end = min(start + char_chunk_size, text_length)

            # Extract chunk
            chunk_text = text[start:end].strip()

            if chunk_text:
                # Try to find the page number for this chunk
                page_num = self._estimate_page_number(start, raw_document)

                chunk = DocumentChunk(
                    content=chunk_text,
                    chunk_type="text",
                    page_number=page_num,
                    metadata={
                        "chunk_index": chunk_idx,
                        "start_char": start,
                        "end_char": end
                    }
                )
                chunks.append(chunk)
                chunk_idx += 1

            # Move to next chunk with overlap
            start = end - char_overlap if end < text_length else text_length

        return chunks

    def _estimate_page_number(self, char_position: int, raw_document: RawDocument) -> int:
        """
        Estimate page number based on character position.

        Args:
            char_position: Character position in full text
            raw_document: Original document

        Returns:
            Estimated page number (1-indexed)
        """
        # Calculate based on pages
        current_pos = 0
        for page in raw_document.pages:
            page_text = page.get("text", "")
            current_pos += len(page_text)
            if char_position < current_pos:
                return page.get("page_num", 1)

        # Default to last page if not found
        return raw_document.total_pages if raw_document.total_pages > 0 else 1

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for list of texts.

        Args:
            texts: List of text strings

        Returns:
            Numpy array of embeddings (shape: num_texts x embedding_dim)
        """
        if not texts:
            return np.array([])

        # Generate embeddings using sentence-transformers
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        return embeddings
