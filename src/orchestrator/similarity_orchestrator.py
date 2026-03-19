"""
Similarity orchestrator for coordinating document comparison across modalities.
"""
from typing import Dict, Tuple
import numpy as np

from models.document import ProcessedDocument
from models.similarity import SimilarityReport, ModalityScore
from orchestrator.scorers import (
    compute_text_similarity,
    compute_table_similarity,
    compute_weighted_score
)
import config


class SimilarityOrchestrator:
    """Orchestrates similarity comparison across multiple modalities."""

    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize orchestrator.

        Args:
            weights: Custom modality weights (defaults to config.MODALITY_WEIGHTS)
        """
        self.weights = weights or config.MODALITY_WEIGHTS

    async def compare_documents(
        self,
        doc1: ProcessedDocument,
        doc1_text_embeddings: np.ndarray,
        doc1_table_embeddings: np.ndarray,
        doc2: ProcessedDocument,
        doc2_text_embeddings: np.ndarray,
        doc2_table_embeddings: np.ndarray
    ) -> SimilarityReport:
        """
        Compare two processed documents across all modalities.

        Args:
            doc1: First processed document
            doc1_text_embeddings: Text embeddings for doc1
            doc1_table_embeddings: Table embeddings for doc1
            doc2: Second processed document
            doc2_text_embeddings: Text embeddings for doc2
            doc2_table_embeddings: Table embeddings for doc2

        Returns:
            SimilarityReport with overall score and per-modality details
        """
        # Compute text similarity
        text_score = compute_text_similarity(
            doc1.text_chunks,
            doc1_text_embeddings,
            doc2.text_chunks,
            doc2_text_embeddings
        )

        # Compute table similarity
        table_score = compute_table_similarity(
            doc1.tables,
            doc1_table_embeddings,
            doc2.tables,
            doc2_table_embeddings
        )

        # Compute weighted overall score
        modality_scores = {
            "text": text_score,
            "table": table_score
        }

        overall_score = compute_weighted_score(modality_scores, self.weights)

        # Compile matched sections from both modalities
        matched_sections = []

        # Add top text matches
        for match in text_score.matched_items[:5]:  # Top 5 text matches
            matched_sections.append({
                "type": "text",
                "doc1_content": match["doc1_content"],
                "doc2_content": match["doc2_content"],
                "similarity": match["similarity"],
                "doc1_page": match["doc1_page"],
                "doc2_page": match["doc2_page"]
            })

        # Add top table matches
        for match in table_score.matched_items[:3]:  # Top 3 table matches
            matched_sections.append({
                "type": "table",
                "doc1_schema": match["doc1_schema"],
                "doc2_schema": match["doc2_schema"],
                "similarity": match["similarity"],
                "doc1_page": match["doc1_page"],
                "doc2_page": match["doc2_page"]
            })

        # Sort all matched sections by similarity
        matched_sections.sort(key=lambda x: x["similarity"], reverse=True)

        # Create report
        report = SimilarityReport(
            doc1_name=doc1.filename,
            doc2_name=doc2.filename,
            overall_score=overall_score,
            text_score=text_score,
            table_score=table_score,
            matched_sections=matched_sections,
            weights_used=self.weights
        )

        return report

    def adjust_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Adjust modality weights.

        Args:
            new_weights: New weight dictionary
        """
        # Validate weights sum to 1.0
        total = sum(new_weights.values())
        if abs(total - 1.0) > 0.01:
            # Normalize weights
            self.weights = {k: v / total for k, v in new_weights.items()}
        else:
            self.weights = new_weights
