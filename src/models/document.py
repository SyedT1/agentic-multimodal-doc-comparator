"""
Data models for documents and document chunks.
"""
from typing import List, Dict, Any
import uuid


class RawDocument:
    """Represents a raw document with extracted content."""

    def __init__(
        self,
        filename: str,
        file_type: str,
        pages: List[Dict[str, Any]],
        raw_text: str,
        raw_tables: List[Dict[str, Any]],
        total_pages: int,
    ):
        """
        Initialize a RawDocument.

        Args:
            filename: Name of the document file
            file_type: Type of file (e.g., 'pdf', 'docx')
            pages: List of page dictionaries with 'page_num' and 'text' keys
            raw_text: Full extracted text from the document
            raw_tables: List of tables extracted from the document
            total_pages: Total number of pages in the document
        """
        self.filename = filename
        self.file_type = file_type
        self.pages = pages
        self.raw_text = raw_text
        self.raw_tables = raw_tables
        self.total_pages = total_pages

    def __repr__(self) -> str:
        return f"RawDocument(filename={self.filename}, pages={self.total_pages})"


class DocumentChunk:
    """Represents a chunk of document content with metadata."""

    def __init__(
        self,
        content: str,
        chunk_type: str,
        page_number: int,
        metadata: Dict[str, Any] = None,
        chunk_id: str = None,
    ):
        """
        Initialize a DocumentChunk.

        Args:
            content: The text content of the chunk
            chunk_type: Type of chunk (e.g., 'text', 'table')
            page_number: Page number where this chunk appears
            metadata: Additional metadata about the chunk
            chunk_id: Unique identifier for the chunk (auto-generated if not provided)
        """
        self.content = content
        self.chunk_type = chunk_type
        self.page_number = page_number
        self.metadata = metadata or {}
        self.chunk_id = chunk_id or str(uuid.uuid4())

    def __repr__(self) -> str:
        return (
            f"DocumentChunk(type={self.chunk_type}, page={self.page_number}, "
            f"length={len(self.content)})"
        )


class TableExtraction:
    """Represents a table extracted from a document."""

    def __init__(
        self,
        headers: List[str],
        rows: List[List[str]],
        page_number: int,
        schema_summary: str,
        table_id: str = None,
    ):
        """
        Initialize a TableExtraction.

        Args:
            headers: List of column headers
            rows: List of rows, each containing cell values
            page_number: Page number where this table appears
            schema_summary: Summary description of the table schema
            table_id: Unique identifier for the table (auto-generated if not provided)
        """
        self.headers = headers
        self.rows = rows
        self.page_number = page_number
        self.schema_summary = schema_summary
        self.table_id = table_id or str(uuid.uuid4())

    def __repr__(self) -> str:
        return (
            f"TableExtraction(columns={len(self.headers)}, "
            f"rows={len(self.rows)}, page={self.page_number})"
        )


class ProcessedDocument:
    """Represents a fully processed document with text chunks and tables."""

    def __init__(
        self,
        filename: str,
        text_chunks: List[DocumentChunk],
        tables: List["TableExtraction"],
        total_pages: int,
        file_type: str,
    ):
        """
        Initialize a ProcessedDocument.

        Args:
            filename: Name of the document file
            text_chunks: List of text chunks extracted from the document
            tables: List of tables extracted from the document
            total_pages: Total number of pages in the document
            file_type: Type of file (e.g., 'pdf', 'docx')
        """
        self.filename = filename
        self.text_chunks = text_chunks
        self.tables = tables
        self.total_pages = total_pages
        self.file_type = file_type

    def __repr__(self) -> str:
        return (
            f"ProcessedDocument(filename={self.filename}, "
            f"text_chunks={len(self.text_chunks)}, "
            f"tables={len(self.tables)})"
        )
