"""
Data models for documents and document chunks.
"""
from typing import List, Dict, Any, Optional
import uuid
from PIL import Image


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
        metadata: Optional[Dict[str, Any]] = None,
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
            metadata: Additional metadata (file_path, author, etc.)
        """
        self.filename = filename
        self.file_type = file_type
        self.pages = pages
        self.raw_text = raw_text
        self.raw_tables = raw_tables
        self.total_pages = total_pages
        self.metadata = metadata or {}

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
        images: Optional[List["ImageExtraction"]] = None,
        layout: Optional["LayoutExtraction"] = None,
        metadata: Optional["MetadataExtraction"] = None,
    ):
        """
        Initialize a ProcessedDocument.

        Args:
            filename: Name of the document file
            text_chunks: List of text chunks extracted from the document
            tables: List of tables extracted from the document
            total_pages: Total number of pages in the document
            file_type: Type of file (e.g., 'pdf', 'docx')
            images: List of images extracted from the document (Phase 2)
            layout: Layout information (Phase 2)
            metadata: Document metadata (Phase 2)
        """
        self.filename = filename
        self.text_chunks = text_chunks
        self.tables = tables
        self.total_pages = total_pages
        self.file_type = file_type
        self.images = images or []
        self.layout = layout
        self.metadata = metadata

    def __repr__(self) -> str:
        return (
            f"ProcessedDocument(filename={self.filename}, "
            f"text_chunks={len(self.text_chunks)}, "
            f"tables={len(self.tables)}, "
            f"images={len(self.images)})"
        )


class ImageExtraction:
    """Represents an image extracted from a document."""

    def __init__(
        self,
        image: Image.Image,
        page_number: int,
        image_index: int,
        width: int,
        height: int,
        format: str,
        image_id: str = None,
    ):
        """
        Initialize an ImageExtraction.

        Args:
            image: PIL Image object
            page_number: Page number where this image appears
            image_index: Index of image on the page
            width: Image width in pixels
            height: Image height in pixels
            format: Image format (png, jpg, etc.)
            image_id: Unique identifier for the image (auto-generated if not provided)
        """
        self.image = image
        self.page_number = page_number
        self.image_index = image_index
        self.width = width
        self.height = height
        self.format = format
        self.image_id = image_id or str(uuid.uuid4())

    def __repr__(self) -> str:
        return (
            f"ImageExtraction(page={self.page_number}, "
            f"size={self.width}x{self.height}, format={self.format})"
        )


class LayoutExtraction:
    """Represents document layout and structure information."""

    def __init__(
        self,
        sections: List[Dict[str, Any]],
        hierarchy: Dict[str, Any],
        page_layouts: List[Dict[str, Any]],
        total_pages: int,
    ):
        """
        Initialize a LayoutExtraction.

        Args:
            sections: List of document sections with hierarchy info
            hierarchy: Document hierarchy tree
            page_layouts: Layout information per page
            total_pages: Total number of pages
        """
        self.sections = sections
        self.hierarchy = hierarchy
        self.page_layouts = page_layouts
        self.total_pages = total_pages

    def __repr__(self) -> str:
        return (
            f"LayoutExtraction(sections={len(self.sections)}, "
            f"pages={self.total_pages})"
        )


class MetadataExtraction:
    """Represents document metadata."""

    def __init__(
        self,
        title: Optional[str] = None,
        author: Optional[str] = None,
        subject: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        creator: Optional[str] = None,
        producer: Optional[str] = None,
        creation_date: Optional[str] = None,
        modification_date: Optional[str] = None,
        page_count: Optional[int] = None,
        custom_properties: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a MetadataExtraction.

        Args:
            title: Document title
            author: Document author
            subject: Document subject
            keywords: List of keywords
            creator: Creator application
            producer: Producer application
            creation_date: Creation date
            modification_date: Last modification date
            page_count: Number of pages
            custom_properties: Additional custom properties
        """
        self.title = title
        self.author = author
        self.subject = subject
        self.keywords = keywords or []
        self.creator = creator
        self.producer = producer
        self.creation_date = creation_date
        self.modification_date = modification_date
        self.page_count = page_count
        self.custom_properties = custom_properties or {}

    def __repr__(self) -> str:
        return (
            f"MetadataExtraction(title={self.title}, "
            f"author={self.author}, pages={self.page_count})"
        )
