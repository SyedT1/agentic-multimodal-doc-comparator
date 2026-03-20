"""
Meta agent for extracting and comparing document metadata.
Handles document properties like title, author, dates, etc.
"""
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import re

from agents.base_agent import BaseAgent
from models.document import MetadataExtraction, RawDocument

# Try to import PDF metadata libraries
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except (ImportError, OSError):
    PYMUPDF_AVAILABLE = False

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

# DOCX metadata
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


class MetaAgent(BaseAgent):
    """Agent responsible for extracting document metadata."""

    def __init__(self, config_dict: Dict[str, Any] = None):
        super().__init__(config_dict)

    def get_agent_name(self) -> str:
        return "MetaAgent"

    async def process(self, raw_document: RawDocument) -> MetadataExtraction:
        """
        Process raw document and extract metadata.

        Args:
            raw_document: Raw document

        Returns:
            MetadataExtraction object with document metadata
        """
        file_path = raw_document.metadata.get("file_path")
        file_type = raw_document.file_type

        if file_type == "pdf":
            return self._extract_pdf_metadata(file_path, raw_document)
        elif file_type == "docx":
            return self._extract_docx_metadata(file_path, raw_document)
        else:
            return MetadataExtraction()

    def _extract_pdf_metadata(
        self,
        file_path: str,
        raw_document: RawDocument
    ) -> MetadataExtraction:
        """
        Extract metadata from PDF file.

        Args:
            file_path: Path to PDF file
            raw_document: Raw document

        Returns:
            MetadataExtraction object
        """
        metadata = MetadataExtraction()

        if not file_path:
            return metadata

        # Try PyMuPDF first
        if PYMUPDF_AVAILABLE:
            try:
                with fitz.open(file_path) as pdf_doc:
                    pdf_metadata = pdf_doc.metadata

                    if pdf_metadata:
                        metadata.title = pdf_metadata.get("title")
                        metadata.author = pdf_metadata.get("author")
                        metadata.subject = pdf_metadata.get("subject")
                        metadata.creator = pdf_metadata.get("creator")
                        metadata.producer = pdf_metadata.get("producer")

                        # Parse keywords
                        keywords_str = pdf_metadata.get("keywords", "")
                        if keywords_str:
                            metadata.keywords = [k.strip() for k in keywords_str.split(",")]

                        # Parse dates
                        creation_date = pdf_metadata.get("creationDate")
                        if creation_date:
                            metadata.creation_date = self._parse_pdf_date(creation_date)

                        mod_date = pdf_metadata.get("modDate")
                        if mod_date:
                            metadata.modification_date = self._parse_pdf_date(mod_date)

                    metadata.page_count = pdf_doc.page_count

            except Exception as e:
                print(f"Error extracting PDF metadata with PyMuPDF: {e}")

        # Fallback to pypdf
        elif PYPDF_AVAILABLE:
            try:
                reader = PdfReader(file_path)
                pdf_metadata = reader.metadata

                if pdf_metadata:
                    metadata.title = pdf_metadata.get("/Title")
                    metadata.author = pdf_metadata.get("/Author")
                    metadata.subject = pdf_metadata.get("/Subject")
                    metadata.creator = pdf_metadata.get("/Creator")
                    metadata.producer = pdf_metadata.get("/Producer")

                    # Parse keywords
                    keywords_str = pdf_metadata.get("/Keywords", "")
                    if keywords_str:
                        metadata.keywords = [k.strip() for k in keywords_str.split(",")]

                    # Parse dates
                    creation_date = pdf_metadata.get("/CreationDate")
                    if creation_date:
                        metadata.creation_date = self._parse_pdf_date(creation_date)

                    mod_date = pdf_metadata.get("/ModDate")
                    if mod_date:
                        metadata.modification_date = self._parse_pdf_date(mod_date)

                metadata.page_count = len(reader.pages)

            except Exception as e:
                print(f"Error extracting PDF metadata with pypdf: {e}")

        # Extract title from text if not in metadata
        if not metadata.title:
            metadata.title = self._extract_title_from_text(raw_document.raw_text)

        return metadata

    def _extract_docx_metadata(
        self,
        file_path: str,
        raw_document: RawDocument
    ) -> MetadataExtraction:
        """
        Extract metadata from DOCX file.

        Args:
            file_path: Path to DOCX file
            raw_document: Raw document

        Returns:
            MetadataExtraction object
        """
        metadata = MetadataExtraction()

        if not file_path or not DOCX_AVAILABLE:
            return metadata

        try:
            doc = Document(file_path)
            core_props = doc.core_properties

            metadata.title = core_props.title
            metadata.author = core_props.author
            metadata.subject = core_props.subject
            metadata.creator = core_props.last_modified_by

            # Parse keywords
            if core_props.keywords:
                metadata.keywords = [k.strip() for k in core_props.keywords.split(",")]

            # Dates
            if core_props.created:
                metadata.creation_date = core_props.created.isoformat()

            if core_props.modified:
                metadata.modification_date = core_props.modified.isoformat()

            metadata.page_count = 1  # DOCX doesn't have clear page count

        except Exception as e:
            print(f"Error extracting DOCX metadata: {e}")

        # Extract title from text if not in metadata
        if not metadata.title:
            metadata.title = self._extract_title_from_text(raw_document.raw_text)

        return metadata

    def _parse_pdf_date(self, date_str: str) -> Optional[str]:
        """
        Parse PDF date format to ISO format.

        Args:
            date_str: PDF date string (e.g., "D:20230101120000")

        Returns:
            ISO format date string or None
        """
        if not date_str:
            return None

        try:
            # PDF date format: D:YYYYMMDDHHmmSSOHH'mm'
            match = re.match(r"D:(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", date_str)
            if match:
                year, month, day, hour, minute, second = match.groups()
                dt = datetime(
                    int(year), int(month), int(day),
                    int(hour), int(minute), int(second)
                )
                return dt.isoformat()
        except Exception as e:
            print(f"Error parsing PDF date: {e}")

        return date_str

    def _extract_title_from_text(self, text: str) -> Optional[str]:
        """
        Extract likely title from document text.

        Args:
            text: Document text

        Returns:
            Extracted title or None
        """
        if not text:
            return None

        # Get first few non-empty lines
        lines = [l.strip() for l in text.split('\n') if l.strip()]

        if not lines:
            return None

        # First line is often the title
        first_line = lines[0]

        # If first line is all caps and under 100 chars, likely a title
        if first_line.isupper() and len(first_line) < 100:
            return first_line

        # If first line is short (< 100 chars) and not starting with common text
        if len(first_line) < 100 and not first_line.lower().startswith(('the ', 'a ', 'an ')):
            return first_line

        return None

    def compute_metadata_similarity(
        self,
        meta1: MetadataExtraction,
        meta2: MetadataExtraction
    ) -> float:
        """
        Compute similarity between two metadata objects.

        Args:
            meta1: First metadata
            meta2: Second metadata

        Returns:
            Similarity score (0.0 to 1.0)
        """
        scores = []

        # Compare titles
        if meta1.title and meta2.title:
            title_sim = self._string_similarity(meta1.title, meta2.title)
            scores.append(title_sim * 2)  # Title is important

        # Compare authors
        if meta1.author and meta2.author:
            author_sim = self._string_similarity(meta1.author, meta2.author)
            scores.append(author_sim * 1.5)  # Author is important

        # Compare subjects
        if meta1.subject and meta2.subject:
            subject_sim = self._string_similarity(meta1.subject, meta2.subject)
            scores.append(subject_sim)

        # Compare keywords
        if meta1.keywords and meta2.keywords:
            keywords_sim = self._list_similarity(meta1.keywords, meta2.keywords)
            scores.append(keywords_sim)

        # Compare page counts
        if meta1.page_count and meta2.page_count:
            page_ratio = min(meta1.page_count, meta2.page_count) / \
                        max(meta1.page_count, meta2.page_count)
            scores.append(page_ratio)

        # Overall similarity (weighted average)
        if scores:
            total_weight = len(scores)
            return sum(scores) / total_weight
        else:
            return 0.0

    def _string_similarity(self, s1: str, s2: str) -> float:
        """
        Compute similarity between two strings.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not s1 or not s2:
            return 0.0

        # Normalize
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()

        # Exact match
        if s1 == s2:
            return 1.0

        # Jaccard similarity on words
        words1 = set(s1.split())
        words2 = set(s2.split())

        intersection = words1 & words2
        union = words1 | words2

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def _list_similarity(self, list1: list, list2: list) -> float:
        """
        Compute Jaccard similarity between two lists.

        Args:
            list1: First list
            list2: Second list

        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not list1 or not list2:
            return 0.0

        set1 = set(item.lower() if isinstance(item, str) else item for item in list1)
        set2 = set(item.lower() if isinstance(item, str) else item for item in list2)

        intersection = set1 & set2
        union = set1 | set2

        if not union:
            return 0.0

        return len(intersection) / len(union)
