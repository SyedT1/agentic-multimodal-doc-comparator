"""
Data models for similarity scoring and comparison results.
"""
from typing import Dict, Any, List


class ModalityScore:
    """Represents similarity score for a specific modality (text, table, etc.)."""

    def __init__(
        self,
        modality: str,
        score: float,
        details: Dict[str, Any] = None,
        matched_items: List[Dict[str, Any]] = None,
    ):
        """
        Initialize a ModalityScore.

        Args:
            modality: Type of modality (e.g., 'text', 'table')
            score: Similarity score (0.0 to 1.0)
            details: Additional details about the scoring
            matched_items: List of matched items between documents
        """
        self.modality = modality
        self.score = score
        self.details = details or {}
        self.matched_items = matched_items or []

    def __repr__(self) -> str:
        return f"ModalityScore(modality={self.modality}, score={self.score:.3f})"


class SimilarityReport:
    """Contains comprehensive similarity comparison results between two documents."""

    def __init__(
        self,
        doc1_name: str,
        doc2_name: str,
        overall_score: float,
        text_score: ModalityScore,
        table_score: ModalityScore,
        matched_sections: List[Dict[str, Any]],
        weights_used: Dict[str, float] = None,
    ):
        """
        Initialize a SimilarityReport.

        Args:
            doc1_name: Name of first document
            doc2_name: Name of second document
            overall_score: Overall similarity score (0.0 to 1.0)
            text_score: ModalityScore for text
            table_score: ModalityScore for tables
            matched_sections: List of matched sections with details
            weights_used: Weights used for modality scoring
        """
        self.doc1_name = doc1_name
        self.doc2_name = doc2_name
        self.overall_score = overall_score
        self.text_score = text_score
        self.table_score = table_score
        self.matched_sections = matched_sections
        self.weights_used = weights_used or {}

    def __repr__(self) -> str:
        return (
            f"SimilarityReport(docs={self.doc1_name} vs {self.doc2_name}, "
            f"score={self.overall_score:.3f})"
        )
