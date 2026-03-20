"""
Batch comparison orchestrator for comparing one document against multiple documents.
Supports 1-to-N document similarity comparisons.
"""
from typing import List, Dict, Any
import numpy as np
import asyncio

from models.document import ProcessedDocument
from models.similarity import SimilarityReport, ModalityScore
from orchestrator.similarity_orchestrator import SimilarityOrchestrator
import config


class BatchComparisonOrchestrator:
    """Orchestrates batch comparison of one document against multiple documents."""

    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize batch orchestrator.

        Args:
            weights: Custom modality weights (defaults to config.MODALITY_WEIGHTS)
        """
        self.weights = weights or config.MODALITY_WEIGHTS
        self.single_orchestrator = SimilarityOrchestrator(weights=self.weights)

    async def compare_one_to_many(
        self,
        query_doc: ProcessedDocument,
        query_embeddings: Dict[str, np.ndarray],
        candidate_docs: List[ProcessedDocument],
        candidate_embeddings: List[Dict[str, np.ndarray]]
    ) -> List[SimilarityReport]:
        """
        Compare one query document against multiple candidate documents.

        Args:
            query_doc: Query document
            query_embeddings: Embeddings for query document
                {
                    "text": text_embeddings,
                    "table": table_embeddings,
                    "image": image_embeddings (optional),
                }
            candidate_docs: List of candidate documents
            candidate_embeddings: List of embedding dicts for each candidate

        Returns:
            List of SimilarityReports, sorted by overall_score (descending)
        """
        reports = []

        # Compare against each candidate
        for candidate_doc, candidate_emb in zip(candidate_docs, candidate_embeddings):
            report = await self.single_orchestrator.compare_documents(
                query_doc,
                query_embeddings.get("text", np.array([])),
                query_embeddings.get("table", np.array([])),
                candidate_doc,
                candidate_emb.get("text", np.array([])),
                candidate_emb.get("table", np.array([])),
                # Phase 2 modalities
                query_embeddings.get("image", np.array([])),
                candidate_emb.get("image", np.array([])),
                query_doc.layout,
                candidate_doc.layout,
                query_doc.metadata,
                candidate_doc.metadata
            )

            reports.append(report)

        # Sort by overall score (descending)
        reports.sort(key=lambda r: r.overall_score, reverse=True)

        return reports

    async def compare_many_to_many(
        self,
        docs1: List[ProcessedDocument],
        embeddings1: List[Dict[str, np.ndarray]],
        docs2: List[ProcessedDocument],
        embeddings2: List[Dict[str, np.ndarray]]
    ) -> List[List[SimilarityReport]]:
        """
        Compare multiple documents against multiple documents (matrix comparison).

        Args:
            docs1: First list of documents
            embeddings1: Embeddings for first list
            docs2: Second list of documents
            embeddings2: Embeddings for second list

        Returns:
            Matrix of SimilarityReports (list of lists)
            result[i][j] = similarity between docs1[i] and docs2[j]
        """
        results = []

        for doc1, emb1 in zip(docs1, embeddings1):
            row_reports = await self.compare_one_to_many(
                doc1,
                emb1,
                docs2,
                embeddings2
            )
            results.append(row_reports)

        return results

    def get_top_matches(
        self,
        reports: List[SimilarityReport],
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[SimilarityReport]:
        """
        Get top K matching documents from a list of reports.

        Args:
            reports: List of similarity reports
            top_k: Number of top matches to return
            min_score: Minimum similarity score threshold

        Returns:
            Top K reports with score >= min_score
        """
        # Filter by minimum score
        filtered = [r for r in reports if r.overall_score >= min_score]

        # Sort by score (descending)
        filtered.sort(key=lambda r: r.overall_score, reverse=True)

        # Return top K
        return filtered[:top_k]

    def group_by_similarity(
        self,
        reports: List[SimilarityReport],
        thresholds: Dict[str, float] = None
    ) -> Dict[str, List[SimilarityReport]]:
        """
        Group reports by similarity level.

        Args:
            reports: List of similarity reports
            thresholds: Custom thresholds for grouping
                Default: {"high": 0.7, "medium": 0.4, "low": 0.0}

        Returns:
            Dictionary with keys: "high", "medium", "low"
        """
        if thresholds is None:
            thresholds = {"high": 0.7, "medium": 0.4, "low": 0.0}

        groups = {
            "high": [],
            "medium": [],
            "low": []
        }

        for report in reports:
            score = report.overall_score

            if score >= thresholds["high"]:
                groups["high"].append(report)
            elif score >= thresholds["medium"]:
                groups["medium"].append(report)
            else:
                groups["low"].append(report)

        return groups

    async def find_duplicates(
        self,
        docs: List[ProcessedDocument],
        embeddings: List[Dict[str, np.ndarray]],
        duplicate_threshold: float = 0.9
    ) -> List[Dict[str, Any]]:
        """
        Find potential duplicate documents in a collection.

        Args:
            docs: List of documents
            embeddings: List of embeddings
            duplicate_threshold: Similarity threshold for duplicates

        Returns:
            List of duplicate pairs with their similarity scores
        """
        duplicates = []

        # Compare each pair
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                report = await self.single_orchestrator.compare_documents(
                    docs[i],
                    embeddings[i].get("text", np.array([])),
                    embeddings[i].get("table", np.array([])),
                    docs[j],
                    embeddings[j].get("text", np.array([])),
                    embeddings[j].get("table", np.array([])),
                    # Phase 2 modalities
                    embeddings[i].get("image", np.array([])),
                    embeddings[j].get("image", np.array([])),
                    docs[i].layout,
                    docs[j].layout,
                    docs[i].metadata,
                    docs[j].metadata
                )

                if report.overall_score >= duplicate_threshold:
                    duplicates.append({
                        "doc1_index": i,
                        "doc2_index": j,
                        "doc1_name": docs[i].filename,
                        "doc2_name": docs[j].filename,
                        "similarity_score": report.overall_score,
                        "report": report
                    })

        # Sort by similarity score (descending)
        duplicates.sort(key=lambda d: d["similarity_score"], reverse=True)

        return duplicates

    def create_similarity_matrix(
        self,
        reports_matrix: List[List[SimilarityReport]]
    ) -> np.ndarray:
        """
        Create a similarity matrix from comparison reports.

        Args:
            reports_matrix: Matrix of similarity reports

        Returns:
            Numpy array of similarity scores
        """
        num_docs1 = len(reports_matrix)
        num_docs2 = len(reports_matrix[0]) if reports_matrix else 0

        matrix = np.zeros((num_docs1, num_docs2))

        for i, row in enumerate(reports_matrix):
            for j, report in enumerate(row):
                matrix[i, j] = report.overall_score

        return matrix
