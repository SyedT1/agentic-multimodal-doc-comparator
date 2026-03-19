"""
Data models for similarity scoring and comparison results.
"""
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime


class ModalityScore(BaseModel):
    """Represents similarity score for a specific modality (text, table, etc.)."""

    modality: str = Field(..., description="Type of modality (e.g., 'text', 'table')")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score (0.0 to 1.0)")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details about the scoring")
    matched_items: List[Dict[str, Any]] = Field(default_factory=list, description="List of matched items between documents")

    def __repr__(self) -> str:
        return f"ModalityScore(modality={self.modality}, score={self.score:.3f})"


class SimilarityReport(BaseModel):
    """Contains comprehensive similarity comparison results between two documents."""

    doc1_name: str = Field(..., description="Name of first document")
    doc2_name: str = Field(..., description="Name of second document")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall similarity score (0.0 to 1.0)")
    text_score: ModalityScore = Field(..., description="ModalityScore for text")
    table_score: ModalityScore = Field(..., description="ModalityScore for tables")
    matched_sections: List[Dict[str, Any]] = Field(default_factory=list, description="List of matched sections with details")
    weights_used: Dict[str, float] = Field(default_factory=dict, description="Weights used for modality scoring")
    timestamp: datetime = Field(default_factory=datetime.now, description="Time when report was generated")

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self) -> str:
        return (
            f"SimilarityReport(docs={self.doc1_name} vs {self.doc2_name}, "
            f"score={self.overall_score:.3f})"
        )
