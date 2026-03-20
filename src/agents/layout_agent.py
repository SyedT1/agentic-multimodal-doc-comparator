"""
Layout agent for analyzing document structure and hierarchy.
Extracts sections, headings, and document organization.
"""
import re
from typing import List, Tuple, Dict, Any
from collections import defaultdict

from agents.base_agent import BaseAgent
from models.document import LayoutExtraction, RawDocument


class LayoutAgent(BaseAgent):
    """Agent responsible for analyzing document layout and structure."""

    def __init__(self, config_dict: Dict[str, Any] = None):
        super().__init__(config_dict)

    def get_agent_name(self) -> str:
        return "LayoutAgent"

    async def process(self, raw_document: RawDocument) -> LayoutExtraction:
        """
        Process raw document and extract layout information.

        Args:
            raw_document: Raw document with text and pages

        Returns:
            LayoutExtraction object with structure information
        """
        # Extract sections from text
        sections = self.extract_sections(raw_document)

        # Build document hierarchy
        hierarchy = self.build_hierarchy(sections)

        # Analyze page layouts
        page_layouts = self.analyze_page_layouts(raw_document)

        return LayoutExtraction(
            sections=sections,
            hierarchy=hierarchy,
            page_layouts=page_layouts,
            total_pages=raw_document.total_pages
        )

    def extract_sections(self, raw_document: RawDocument) -> List[Dict[str, Any]]:
        """
        Extract sections from document based on headings and formatting.

        Args:
            raw_document: Raw document

        Returns:
            List of section dictionaries
        """
        sections = []

        # Common heading patterns
        heading_patterns = [
            r'^#{1,6}\s+(.+)$',  # Markdown-style headings
            r'^([A-Z][A-Z\s]+)$',  # ALL CAPS headings
            r'^(\d+\.?\s+[A-Z].+)$',  # Numbered sections (1. Introduction)
            r'^([IVX]+\.\s+.+)$',  # Roman numeral sections
            r'^(Chapter\s+\d+.*)$',  # Chapter headings
            r'^(Section\s+\d+.*)$',  # Section headings
            r'^(ABSTRACT|INTRODUCTION|CONCLUSION|REFERENCES|APPENDIX)',  # Common section names
        ]

        # Process each page
        for page in raw_document.pages:
            page_num = page.get("page_num", 1)
            page_text = page.get("text", "")

            if not page_text.strip():
                continue

            # Split into lines
            lines = page_text.split('\n')

            for line_idx, line in enumerate(lines):
                line = line.strip()

                if not line:
                    continue

                # Check if line matches any heading pattern
                for pattern in heading_patterns:
                    match = re.match(pattern, line, re.IGNORECASE)
                    if match:
                        # Estimate heading level
                        level = self._estimate_heading_level(line, pattern)

                        sections.append({
                            "title": line,
                            "level": level,
                            "page_number": page_num,
                            "line_number": line_idx,
                            "type": "heading"
                        })
                        break

        # If no sections found, create generic page-based sections
        if not sections:
            for page in raw_document.pages:
                sections.append({
                    "title": f"Page {page.get('page_num', 1)}",
                    "level": 1,
                    "page_number": page.get('page_num', 1),
                    "line_number": 0,
                    "type": "page"
                })

        return sections

    def _estimate_heading_level(self, line: str, pattern: str) -> int:
        """
        Estimate the hierarchical level of a heading.

        Args:
            line: Heading text
            pattern: Pattern that matched

        Returns:
            Heading level (1-6, where 1 is highest)
        """
        # Markdown headings
        if pattern.startswith(r'^#{'):
            return line.count('#')

        # ALL CAPS likely level 1
        if line.isupper():
            return 1

        # Numbered sections
        if re.match(r'^\d+\.', line):
            # Count depth: 1. = level 1, 1.1 = level 2, etc.
            depth = line.split()[0].count('.')
            return min(depth + 1, 6)

        # Roman numerals likely level 1
        if re.match(r'^[IVX]+\.', line):
            return 1

        # Chapter/Section likely level 1-2
        if line.lower().startswith(('chapter', 'section')):
            return 2

        # Common section names
        if line.upper() in ['ABSTRACT', 'INTRODUCTION', 'CONCLUSION', 'REFERENCES', 'APPENDIX']:
            return 1

        # Default
        return 3

    def build_hierarchy(self, sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build hierarchical structure from sections.

        Args:
            sections: List of section dictionaries

        Returns:
            Hierarchy tree dictionary
        """
        hierarchy = {
            "root": {
                "title": "Document",
                "level": 0,
                "children": []
            }
        }

        # Stack to track current parent at each level
        level_stack = [hierarchy["root"]]

        for section in sections:
            level = section["level"]

            # Create section node
            node = {
                "title": section["title"],
                "level": level,
                "page_number": section["page_number"],
                "children": []
            }

            # Find appropriate parent
            while len(level_stack) > level:
                level_stack.pop()

            # Add to parent
            if level_stack:
                level_stack[-1]["children"].append(node)

            # Update stack
            level_stack.append(node)

        return hierarchy

    def analyze_page_layouts(self, raw_document: RawDocument) -> List[Dict[str, Any]]:
        """
        Analyze layout characteristics of each page.

        Args:
            raw_document: Raw document

        Returns:
            List of page layout dictionaries
        """
        page_layouts = []

        for page in raw_document.pages:
            page_num = page.get("page_num", 1)
            page_text = page.get("text", "")

            # Basic layout analysis
            lines = page_text.split('\n')
            non_empty_lines = [l for l in lines if l.strip()]

            # Estimate columns (simple heuristic)
            avg_line_length = sum(len(l) for l in non_empty_lines) / max(len(non_empty_lines), 1)
            estimated_columns = 2 if avg_line_length < 50 else 1

            # Count elements
            num_lines = len(non_empty_lines)
            num_words = sum(len(l.split()) for l in non_empty_lines)

            # Find tables on this page
            page_tables = [t for t in raw_document.raw_tables if t.get("page_num") == page_num]

            page_layout = {
                "page_number": page_num,
                "num_lines": num_lines,
                "num_words": num_words,
                "estimated_columns": estimated_columns,
                "has_tables": len(page_tables) > 0,
                "num_tables": len(page_tables),
                "avg_line_length": avg_line_length,
            }

            page_layouts.append(page_layout)

        return page_layouts

    def compute_layout_similarity(
        self,
        layout1: LayoutExtraction,
        layout2: LayoutExtraction
    ) -> float:
        """
        Compute similarity between two document layouts.

        Args:
            layout1: First layout
            layout2: Second layout

        Returns:
            Similarity score (0.0 to 1.0)
        """
        scores = []

        # Compare number of sections
        if layout1.sections and layout2.sections:
            section_ratio = min(len(layout1.sections), len(layout2.sections)) / \
                          max(len(layout1.sections), len(layout2.sections))
            scores.append(section_ratio)

        # Compare hierarchy depth
        depth1 = self._get_hierarchy_depth(layout1.hierarchy)
        depth2 = self._get_hierarchy_depth(layout2.hierarchy)
        if depth1 > 0 and depth2 > 0:
            depth_ratio = min(depth1, depth2) / max(depth1, depth2)
            scores.append(depth_ratio)

        # Compare average page characteristics
        if layout1.page_layouts and layout2.page_layouts:
            avg_words1 = sum(p["num_words"] for p in layout1.page_layouts) / len(layout1.page_layouts)
            avg_words2 = sum(p["num_words"] for p in layout2.page_layouts) / len(layout2.page_layouts)

            if avg_words1 > 0 and avg_words2 > 0:
                words_ratio = min(avg_words1, avg_words2) / max(avg_words1, avg_words2)
                scores.append(words_ratio)

        # Overall similarity
        return sum(scores) / len(scores) if scores else 0.0

    def _get_hierarchy_depth(self, hierarchy: Dict[str, Any]) -> int:
        """
        Get maximum depth of hierarchy tree.

        Args:
            hierarchy: Hierarchy dictionary

        Returns:
            Maximum depth
        """
        def _depth(node: Dict[str, Any]) -> int:
            if not node.get("children"):
                return 0
            return 1 + max(_depth(child) for child in node["children"])

        return _depth(hierarchy.get("root", {}))
