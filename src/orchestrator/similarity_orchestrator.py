"""
Similarity orchestrator for coordinating document comparison across modalities.
"""
from typing import Dict, Tuple, Optional
import numpy as np

from models.document import ProcessedDocument, LayoutExtraction, MetadataExtraction
from models.similarity import SimilarityReport, ModalityScore
from orchestrator.scorers import (
    compute_text_similarity,
    compute_table_similarity,
    compute_weighted_score,
    compute_image_similarity,
    compute_layout_similarity,
    compute_metadata_similarity
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
        doc2_table_embeddings: np.ndarray,
        # Phase 2 parameters (optional)
        doc1_image_embeddings: Optional[np.ndarray] = None,
        doc2_image_embeddings: Optional[np.ndarray] = None,
        doc1_layout: Optional[LayoutExtraction] = None,
        doc2_layout: Optional[LayoutExtraction] = None,
        doc1_metadata: Optional[MetadataExtraction] = None,
        doc2_metadata: Optional[MetadataExtraction] = None
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
            doc1_image_embeddings: Image embeddings for doc1 (Phase 2)
            doc2_image_embeddings: Image embeddings for doc2 (Phase 2)
            doc1_layout: Layout information for doc1 (Phase 2)
            doc2_layout: Layout information for doc2 (Phase 2)
            doc1_metadata: Metadata for doc1 (Phase 2)
            doc2_metadata: Metadata for doc2 (Phase 2)

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

        # Phase 2: Compute image similarity (if enabled and available)
        image_score = None
        if (config.ENABLE_IMAGE_COMPARISON and
            doc1_image_embeddings is not None and
            doc2_image_embeddings is not None and
            doc1.images and doc2.images):
            image_score = compute_image_similarity(
                doc1.images,
                doc1_image_embeddings,
                doc2.images,
                doc2_image_embeddings
            )

        # Phase 2: Compute layout similarity (if enabled and available)
        layout_score = None
        if (config.ENABLE_LAYOUT_COMPARISON and
            doc1_layout is not None and
            doc2_layout is not None):
            layout_score = compute_layout_similarity(doc1_layout, doc2_layout)

        # Phase 2: Compute metadata similarity (if enabled and available)
        metadata_score = None
        if (config.ENABLE_METADATA_COMPARISON and
            doc1_metadata is not None and
            doc2_metadata is not None):
            metadata_score = compute_metadata_similarity(doc1_metadata, doc2_metadata)

        # Collect all modality scores
        modality_scores = {
            "text": text_score,
            "table": table_score
        }

        if image_score:
            modality_scores["image"] = image_score
        if layout_score:
            modality_scores["layout"] = layout_score
        if metadata_score:
            modality_scores["metadata"] = metadata_score

        # Compute weighted overall score
        overall_score = compute_weighted_score(modality_scores, self.weights)

        # Compile matched sections from all modalities
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

        # Phase 2: Add top image matches
        if image_score and image_score.matched_items:
            for match in image_score.matched_items[:3]:  # Top 3 image matches
                matched_sections.append({
                    "type": "image",
                    "doc1_image_id": match["doc1_image_id"],
                    "doc2_image_id": match["doc2_image_id"],
                    "doc1_page": match["doc1_page"],
                    "doc2_page": match["doc2_page"],
                    "similarity": match["similarity"]
                })

        # Phase 2: Add metadata matches
        if metadata_score and metadata_score.matched_items:
            for match in metadata_score.matched_items[:5]:
                matched_sections.append({
                    "type": "metadata",
                    "field": match["field"],
                    "doc1_value": match["doc1_value"],
                    "doc2_value": match["doc2_value"],
                    "similarity": match["similarity"]
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
            image_score=image_score,
            layout_score=layout_score,
            metadata_score=metadata_score,
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
