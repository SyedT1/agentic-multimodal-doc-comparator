"""
Similarity scorers for different modalities.
"""
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity

from models.similarity import ModalityScore
from models.document import DocumentChunk, TableExtraction
import config


def compute_text_similarity(
    doc1_chunks: List[DocumentChunk],
    doc1_embeddings: np.ndarray,
    doc2_chunks: List[DocumentChunk],
    doc2_embeddings: np.ndarray
) -> ModalityScore:
    """
    Compute text similarity between two documents.

    Args:
        doc1_chunks: Chunks from document 1
        doc1_embeddings: Embeddings for document 1
        doc2_chunks: Chunks from document 2
        doc2_embeddings: Embeddings for document 2

    Returns:
        ModalityScore with text similarity details
    """
    if len(doc1_embeddings) == 0 or len(doc2_embeddings) == 0:
        return ModalityScore(
            modality="text",
            score=0.0,
            details={"reason": "One or both documents have no text"},
            matched_items=[]
        )

    # Compute pairwise cosine similarities
    similarities = cosine_similarity(doc1_embeddings, doc2_embeddings)

    # Find best matches for each chunk in doc1
    matched_items = []
    similarity_scores = []

    for i, chunk1 in enumerate(doc1_chunks):
        # Find best matching chunk in doc2
        best_match_idx = np.argmax(similarities[i])
        best_score = similarities[i][best_match_idx]

        if best_score > 0.5:  # Only include matches above threshold
            chunk2 = doc2_chunks[best_match_idx]

            matched_items.append({
                "doc1_chunk_id": chunk1.chunk_id,
                "doc2_chunk_id": chunk2.chunk_id,
                "doc1_content": chunk1.content[:200] + "..." if len(chunk1.content) > 200 else chunk1.content,
                "doc2_content": chunk2.content[:200] + "..." if len(chunk2.content) > 200 else chunk2.content,
                "similarity": float(best_score),
                "doc1_page": chunk1.page_number,
                "doc2_page": chunk2.page_number
            })

        similarity_scores.append(best_score)

    # Overall text score (mean of best matches)
    overall_score = float(np.mean(similarity_scores)) if similarity_scores else 0.0

    # Sort matched items by similarity (descending)
    matched_items.sort(key=lambda x: x["similarity"], reverse=True)

    return ModalityScore(
        modality="text",
        score=overall_score,
        details={
            "num_doc1_chunks": len(doc1_chunks),
            "num_doc2_chunks": len(doc2_chunks),
            "num_matches": len(matched_items),
            "average_similarity": overall_score
        },
        matched_items=matched_items[:config.TOP_K_MATCHES]  # Limit to top K
    )


def compute_table_similarity(
    doc1_tables: List[TableExtraction],
    doc1_embeddings: np.ndarray,
    doc2_tables: List[TableExtraction],
    doc2_embeddings: np.ndarray
) -> ModalityScore:
    """
    Compute table similarity between two documents.

    Args:
        doc1_tables: Tables from document 1
        doc1_embeddings: Embeddings for document 1 tables
        doc2_tables: Tables from document 2
        doc2_embeddings: Embeddings for document 2 tables

    Returns:
        ModalityScore with table similarity details
    """
    if len(doc1_tables) == 0 and len(doc2_tables) == 0:
        # Both documents have no tables - perfectly similar in this modality
        return ModalityScore(
            modality="table",
            score=1.0,
            details={"reason": "Neither document has tables"},
            matched_items=[]
        )

    if len(doc1_embeddings) == 0 or len(doc2_embeddings) == 0:
        # One has tables, the other doesn't
        return ModalityScore(
            modality="table",
            score=0.0,
            details={"reason": "One document has tables, the other doesn't"},
            matched_items=[]
        )

    # Compute pairwise cosine similarities
    similarities = cosine_similarity(doc1_embeddings, doc2_embeddings)

    # Find best matches
    matched_items = []
    similarity_scores = []

    for i, table1 in enumerate(doc1_tables):
        # Find best matching table in doc2
        best_match_idx = np.argmax(similarities[i])
        best_score = similarities[i][best_match_idx]

        if best_score > 0.3:  # Lower threshold for tables
            table2 = doc2_tables[best_match_idx]

            matched_items.append({
                "doc1_table_id": table1.table_id,
                "doc2_table_id": table2.table_id,
                "doc1_schema": table1.schema_summary,
                "doc2_schema": table2.schema_summary,
                "similarity": float(best_score),
                "doc1_page": table1.page_number,
                "doc2_page": table2.page_number
            })

        similarity_scores.append(best_score)

    # Overall table score
    overall_score = float(np.mean(similarity_scores)) if similarity_scores else 0.0

    # Sort matched items by similarity
    matched_items.sort(key=lambda x: x["similarity"], reverse=True)

    return ModalityScore(
        modality="table",
        score=overall_score,
        details={
            "num_doc1_tables": len(doc1_tables),
            "num_doc2_tables": len(doc2_tables),
            "num_matches": len(matched_items),
            "average_similarity": overall_score
        },
        matched_items=matched_items
    )


def compute_weighted_score(
    modality_scores: Dict[str, ModalityScore],
    weights: Dict[str, float] = None
) -> float:
    """
    Compute weighted overall similarity score.

    Args:
        modality_scores: Dictionary of modality -> ModalityScore
        weights: Dictionary of modality -> weight (defaults to config.MODALITY_WEIGHTS)

    Returns:
        Weighted overall score (0.0 to 1.0)
    """
    if weights is None:
        weights = config.MODALITY_WEIGHTS

    total_score = 0.0
    total_weight = 0.0

    for modality, score_obj in modality_scores.items():
        if modality in weights:
            weight = weights[modality]
            total_score += score_obj.score * weight
            total_weight += weight

    # Normalize by total weight
    if total_weight > 0:
        return total_score / total_weight
    else:
        return 0.0
