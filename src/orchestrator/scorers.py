"""
Similarity scorers for different modalities.
"""
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity

from models.similarity import ModalityScore
from models.document import DocumentChunk, TableExtraction, ImageExtraction, LayoutExtraction, MetadataExtraction
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


def compute_image_similarity(
    doc1_images: List[ImageExtraction],
    doc1_embeddings: np.ndarray,
    doc2_images: List[ImageExtraction],
    doc2_embeddings: np.ndarray
) -> ModalityScore:
    """
    Compute image similarity between two documents.

    Args:
        doc1_images: Images from document 1
        doc1_embeddings: Embeddings for document 1 images
        doc2_images: Images from document 2
        doc2_embeddings: Embeddings for document 2 images

    Returns:
        ModalityScore with image similarity details
    """
    if len(doc1_images) == 0 and len(doc2_images) == 0:
        # Both documents have no images - perfectly similar
        return ModalityScore(
            modality="image",
            score=1.0,
            details={"reason": "Neither document has images"},
            matched_items=[]
        )

    if len(doc1_embeddings) == 0 or len(doc2_embeddings) == 0:
        # One has images, the other doesn't
        return ModalityScore(
            modality="image",
            score=0.0,
            details={"reason": "One document has images, the other doesn't"},
            matched_items=[]
        )

    # Compute pairwise cosine similarities
    similarities = cosine_similarity(doc1_embeddings, doc2_embeddings)

    # Find best matches
    matched_items = []
    similarity_scores = []

    for i, img1 in enumerate(doc1_images):
        # Find best matching image in doc2
        best_match_idx = np.argmax(similarities[i])
        best_score = similarities[i][best_match_idx]

        if best_score > 0.7:  # High threshold for images
            img2 = doc2_images[best_match_idx]

            matched_items.append({
                "doc1_image_id": img1.image_id,
                "doc2_image_id": img2.image_id,
                "doc1_page": img1.page_number,
                "doc2_page": img2.page_number,
                "doc1_size": f"{img1.width}x{img1.height}",
                "doc2_size": f"{img2.width}x{img2.height}",
                "similarity": float(best_score)
            })

        similarity_scores.append(best_score)

    # Overall image score
    overall_score = float(np.mean(similarity_scores)) if similarity_scores else 0.0

    # Sort matched items by similarity
    matched_items.sort(key=lambda x: x["similarity"], reverse=True)

    return ModalityScore(
        modality="image",
        score=overall_score,
        details={
            "num_doc1_images": len(doc1_images),
            "num_doc2_images": len(doc2_images),
            "num_matches": len(matched_items),
            "average_similarity": overall_score
        },
        matched_items=matched_items
    )


def compute_layout_similarity(
    layout1: LayoutExtraction,
    layout2: LayoutExtraction
) -> ModalityScore:
    """
    Compute layout similarity between two documents.

    Args:
        layout1: Layout from document 1
        layout2: Layout from document 2

    Returns:
        ModalityScore with layout similarity details
    """
    if not layout1 or not layout2:
        return ModalityScore(
            modality="layout",
            score=0.0,
            details={"reason": "Layout information not available"},
            matched_items=[]
        )

    scores = []

    # Compare number of sections
    if layout1.sections and layout2.sections:
        section_ratio = min(len(layout1.sections), len(layout2.sections)) / \
                       max(len(layout1.sections), len(layout2.sections))
        scores.append(("sections", section_ratio))

    # Compare hierarchy depth
    depth1 = _get_hierarchy_depth(layout1.hierarchy)
    depth2 = _get_hierarchy_depth(layout2.hierarchy)
    if depth1 > 0 and depth2 > 0:
        depth_ratio = min(depth1, depth2) / max(depth1, depth2)
        scores.append(("hierarchy_depth", depth_ratio))

    # Compare section titles (textual similarity)
    if layout1.sections and layout2.sections:
        titles1 = [s["title"] for s in layout1.sections if s.get("title")]
        titles2 = [s["title"] for s in layout2.sections if s.get("title")]

        if titles1 and titles2:
            title_sim = _compute_text_list_similarity(titles1, titles2)
            scores.append(("section_titles", title_sim))

    # Compare page layouts
    if layout1.page_layouts and layout2.page_layouts:
        avg_words1 = sum(p["num_words"] for p in layout1.page_layouts) / len(layout1.page_layouts)
        avg_words2 = sum(p["num_words"] for p in layout2.page_layouts) / len(layout2.page_layouts)

        if avg_words1 > 0 and avg_words2 > 0:
            words_ratio = min(avg_words1, avg_words2) / max(avg_words1, avg_words2)
            scores.append(("page_density", words_ratio))

    # Overall similarity
    if scores:
        overall_score = sum(s[1] for s in scores) / len(scores)
        details = {name: float(score) for name, score in scores}
        details["num_metrics"] = len(scores)
    else:
        overall_score = 0.0
        details = {"reason": "No comparable layout features"}

    return ModalityScore(
        modality="layout",
        score=overall_score,
        details=details,
        matched_items=[]
    )


def compute_metadata_similarity(
    meta1: MetadataExtraction,
    meta2: MetadataExtraction
) -> ModalityScore:
    """
    Compute metadata similarity between two documents.

    Args:
        meta1: Metadata from document 1
        meta2: Metadata from document 2

    Returns:
        ModalityScore with metadata similarity details
    """
    if not meta1 or not meta2:
        return ModalityScore(
            modality="metadata",
            score=0.0,
            details={"reason": "Metadata not available"},
            matched_items=[]
        )

    scores = []
    matched_fields = []

    # Compare titles
    if meta1.title and meta2.title:
        title_sim = _string_similarity(meta1.title, meta2.title)
        scores.append(("title", title_sim * 2))  # Title is important
        matched_fields.append({
            "field": "title",
            "doc1_value": meta1.title,
            "doc2_value": meta2.title,
            "similarity": float(title_sim)
        })

    # Compare authors
    if meta1.author and meta2.author:
        author_sim = _string_similarity(meta1.author, meta2.author)
        scores.append(("author", author_sim * 1.5))  # Author is important
        matched_fields.append({
            "field": "author",
            "doc1_value": meta1.author,
            "doc2_value": meta2.author,
            "similarity": float(author_sim)
        })

    # Compare subjects
    if meta1.subject and meta2.subject:
        subject_sim = _string_similarity(meta1.subject, meta2.subject)
        scores.append(("subject", subject_sim))
        matched_fields.append({
            "field": "subject",
            "doc1_value": meta1.subject,
            "doc2_value": meta2.subject,
            "similarity": float(subject_sim)
        })

    # Compare keywords
    if meta1.keywords and meta2.keywords:
        keywords_sim = _list_similarity(meta1.keywords, meta2.keywords)
        scores.append(("keywords", keywords_sim))
        matched_fields.append({
            "field": "keywords",
            "doc1_value": ", ".join(meta1.keywords),
            "doc2_value": ", ".join(meta2.keywords),
            "similarity": float(keywords_sim)
        })

    # Compare page counts
    if meta1.page_count and meta2.page_count:
        page_ratio = min(meta1.page_count, meta2.page_count) / \
                    max(meta1.page_count, meta2.page_count)
        scores.append(("page_count", page_ratio))

    # Overall similarity (weighted average)
    if scores:
        total_weight = sum(s[1] for s in scores)
        overall_score = total_weight / len(scores)
        details = {name: float(score) for name, score in scores}
        details["num_fields_compared"] = len(scores)
    else:
        overall_score = 0.0
        details = {"reason": "No comparable metadata fields"}
        matched_fields = []

    return ModalityScore(
        modality="metadata",
        score=overall_score,
        details=details,
        matched_items=matched_fields
    )


# Helper functions

def _get_hierarchy_depth(hierarchy: Dict[str, Any]) -> int:
    """Get maximum depth of hierarchy tree."""
    def _depth(node: Dict[str, Any]) -> int:
        if not node.get("children"):
            return 0
        return 1 + max(_depth(child) for child in node["children"])

    return _depth(hierarchy.get("root", {}))


def _compute_text_list_similarity(list1: List[str], list2: List[str]) -> float:
    """Compute average textual similarity between two lists of strings."""
    if not list1 or not list2:
        return 0.0

    # Create all pairs and compute similarities
    similarities = []
    for s1 in list1:
        best_sim = max(_string_similarity(s1, s2) for s2 in list2)
        similarities.append(best_sim)

    return sum(similarities) / len(similarities) if similarities else 0.0


def _string_similarity(s1: str, s2: str) -> float:
    """Compute Jaccard similarity between two strings based on words."""
    if not s1 or not s2:
        return 0.0

    s1 = s1.lower().strip()
    s2 = s2.lower().strip()

    if s1 == s2:
        return 1.0

    words1 = set(s1.split())
    words2 = set(s2.split())

    intersection = words1 & words2
    union = words1 | words2

    if not union:
        return 0.0

    return len(intersection) / len(union)


def _list_similarity(list1: list, list2: list) -> float:
    """Compute Jaccard similarity between two lists."""
    if not list1 or not list2:
        return 0.0

    set1 = set(item.lower() if isinstance(item, str) else item for item in list1)
    set2 = set(item.lower() if isinstance(item, str) else item for item in list2)

    intersection = set1 & set2
    union = set1 | set2

    if not union:
        return 0.0

    return len(intersection) / len(union)
