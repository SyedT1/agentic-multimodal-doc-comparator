"""
Table agent for extracting and embedding table data.
"""
import numpy as np
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer

from agents.base_agent import BaseAgent
from models.document import TableExtraction, RawDocument
import config


class TableAgent(BaseAgent):
    """Agent responsible for table extraction and embedding generation."""

    def __init__(self, config_dict: Dict[str, Any] = None):
        super().__init__(config_dict)
        # Load embedding model (same as text agent for consistency)
        self.model = SentenceTransformer(config.TEXT_EMBEDDING_MODEL)

    def get_agent_name(self) -> str:
        return "TableAgent"

    async def process(self, raw_document: RawDocument) -> Tuple[List[TableExtraction], np.ndarray]:
        """
        Process raw tables into structured format and embeddings.

        Args:
            raw_document: Raw document with extracted tables

        Returns:
            Tuple of (list of TableExtraction objects, numpy array of embeddings)
        """
        # Parse tables
        tables = self.parse_tables(raw_document.raw_tables)

        # Generate embeddings
        if tables:
            table_texts = [self.linearize_table(table) for table in tables]
            embeddings = self.generate_embeddings(table_texts)
        else:
            embeddings = np.array([])

        return tables, embeddings

    def parse_tables(self, raw_tables: List[Dict[str, Any]]) -> List[TableExtraction]:
        """
        Parse raw table data into structured TableExtraction objects.

        Args:
            raw_tables: List of raw table dictionaries

        Returns:
            List of TableExtraction objects
        """
        tables = []

        for raw_table in raw_tables:
            table_data = raw_table.get("data", [])
            if not table_data or len(table_data) < 1:
                continue

            # First row is usually headers
            headers = [str(cell).strip() for cell in table_data[0]] if table_data else []

            # Remaining rows are data
            rows = []
            for row_data in table_data[1:]:
                row = [str(cell).strip() for cell in row_data]
                rows.append(row)

            # Generate schema summary
            schema_summary = self._generate_schema_summary(headers, rows)

            table = TableExtraction(
                headers=headers,
                rows=rows,
                page_number=raw_table.get("page_num", 1),
                schema_summary=schema_summary
            )
            tables.append(table)

        return tables

    def _generate_schema_summary(self, headers: List[str], rows: List[List[str]]) -> str:
        """
        Generate a summary of the table schema.

        Args:
            headers: Table headers
            rows: Table rows

        Returns:
            Schema summary string
        """
        num_columns = len(headers)
        num_rows = len(rows)

        summary = f"Table with {num_columns} columns and {num_rows} rows. "
        summary += f"Columns: {', '.join(headers[:5])}"  # Show first 5 headers

        if len(headers) > 5:
            summary += f" and {len(headers) - 5} more"

        return summary

    def linearize_table(self, table: TableExtraction) -> str:
        """
        Convert table to linear text format for embedding.

        Args:
            table: TableExtraction object

        Returns:
            Linearized table as string
        """
        # Format: "Header1: value1, Header2: value2, ..."
        lines = []

        # Add schema summary
        lines.append(table.schema_summary)

        # Add headers
        if table.headers:
            lines.append(f"Headers: {' | '.join(table.headers)}")

        # Add rows (sample first few for embedding)
        max_rows = 10  # Limit to avoid very long text
        for idx, row in enumerate(table.rows[:max_rows], start=1):
            if row:
                # Create row representation
                row_text = f"Row {idx}: {' | '.join(row)}"
                lines.append(row_text)

        if len(table.rows) > max_rows:
            lines.append(f"... and {len(table.rows) - max_rows} more rows")

        return "\n".join(lines)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for linearized tables.

        Args:
            texts: List of linearized table texts

        Returns:
            Numpy array of embeddings (shape: num_tables x embedding_dim)
        """
        if not texts:
            return np.array([])

        # Generate embeddings using sentence-transformers
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        return embeddings
