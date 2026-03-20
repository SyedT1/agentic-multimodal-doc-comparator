"""
Smart Semantic Chunker
Chunks documents efficiently using sentence boundaries + structural signals.
"""
import re
from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str          # "doc1" or "doc2"
    text: str
    chunk_index: int
    section: str = ""    # heading/section title if detected
    page: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _detect_heading(line: str) -> bool:
    """Detect if a line looks like a section heading."""
    line = line.strip()
    if not line:
        return False
    if re.match(r'^(\d+[\.\)]\s+|[A-Z][A-Z\s]{3,50}$)', line):
        return True
    if len(line) < 80 and not line.endswith('.') and line[0].isupper():
        if re.match(r'^(Abstract|Introduction|Conclusion|Method|Result|Discussion|Background|Overview|Summary)', line, re.I):
            return True
    return False


def chunk_text(
    text: str,
    doc_id: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> List[Chunk]:
    """
    Semantic chunking with section awareness, sentence boundary respect,
    and sliding window overlap.
    """
    chunks = []
    lines = text.split('\n')

    current_section = "General"
    buffer_sentences = []
    buffer_words = 0
    chunk_index = 0

    def flush_buffer(section: str) -> None:
        nonlocal chunk_index, buffer_sentences, buffer_words
        if not buffer_sentences:
            return
        chunk_text_val = ' '.join(buffer_sentences)
        chunks.append(Chunk(
            chunk_id=f"{doc_id}_chunk_{chunk_index}",
            doc_id=doc_id,
            text=chunk_text_val,
            chunk_index=chunk_index,
            section=section,
            metadata={"word_count": buffer_words}
        ))
        chunk_index += 1
        overlap_sentences = []
        overlap_words = 0
        for sent in reversed(buffer_sentences):
            w = len(sent.split())
            if overlap_words + w <= overlap:
                overlap_sentences.insert(0, sent)
                overlap_words += w
            else:
                break
        buffer_sentences = overlap_sentences
        buffer_words = overlap_words

    paragraph_buffer = []

    for line in lines:
        stripped = line.strip()

        if _detect_heading(stripped):
            if paragraph_buffer:
                full_text = ' '.join(paragraph_buffer)
                sentences = _split_sentences(full_text)
                for sent in sentences:
                    buffer_sentences.append(sent)
                    buffer_words += len(sent.split())
                    if buffer_words >= chunk_size:
                        flush_buffer(current_section)
                paragraph_buffer = []
            flush_buffer(current_section)
            current_section = stripped
            continue

        if stripped:
            paragraph_buffer.append(stripped)
        else:
            if paragraph_buffer:
                full_text = ' '.join(paragraph_buffer)
                sentences = _split_sentences(full_text)
                for sent in sentences:
                    buffer_sentences.append(sent)
                    buffer_words += len(sent.split())
                    if buffer_words >= chunk_size:
                        flush_buffer(current_section)
                paragraph_buffer = []

    if paragraph_buffer:
        full_text = ' '.join(paragraph_buffer)
        sentences = _split_sentences(full_text)
        for sent in sentences:
            buffer_sentences.append(sent)
            buffer_words += len(sent.split())
    flush_buffer(current_section)

    return chunks


# ── Debug helper ──────────────────────────────────────────────────────────────

def debug_raw_doc(raw_doc) -> str:
    """Return a string summarising all attributes of a raw_doc for debugging."""
    lines = [f"Type: {type(raw_doc).__name__}"]
    try:
        d = raw_doc.model_dump() if hasattr(raw_doc, 'model_dump') else vars(raw_doc)
        for k, v in d.items():
            if isinstance(v, str):
                lines.append(f"  str attr '{k}': len={len(v)} preview={repr(v[:80])}")
            elif isinstance(v, list):
                lines.append(f"  list attr '{k}': len={len(v)}")
            else:
                lines.append(f"  attr '{k}': {type(v).__name__} = {repr(str(v)[:60])}")
    except Exception as e:
        lines.append(f"  (could not introspect: {e})")
    return '\n'.join(lines)


# ── Robust text extraction ────────────────────────────────────────────────────

def extract_text_from_raw_doc(raw_doc) -> str:
    """
    Robustly extract text from whatever RawDocument the ingestion agent returns.
    Tries all known attribute names and fallback strategies.
    """
    # Strategy 1: Common direct string attributes
    for attr in ['text_content', 'content', 'text', 'raw_text', 'full_text', 'body',
                 'extracted_text', 'plain_text', 'document_text']:
        val = getattr(raw_doc, attr, None)
        if val and isinstance(val, str) and len(val.strip()) > 10:
            return val.strip()

    # Strategy 2: List of pages / sections
    for attr in ['pages', 'sections', 'chunks', 'paragraphs', 'text_chunks']:
        val = getattr(raw_doc, attr, None)
        if val and isinstance(val, list):
            parts = []
            for item in val:
                if isinstance(item, str):
                    parts.append(item)
                elif hasattr(item, 'text') and isinstance(item.text, str):
                    parts.append(item.text)
                elif hasattr(item, 'content') and isinstance(item.content, str):
                    parts.append(item.content)
                elif isinstance(item, dict):
                    parts.append(str(item.get('text') or item.get('content') or ''))
            combined = '\n'.join(p for p in parts if p.strip())
            if len(combined.strip()) > 10:
                return combined.strip()

    # Strategy 3: Pydantic model_dump / __dict__ — grab longest string field
    try:
        d = raw_doc.model_dump() if hasattr(raw_doc, 'model_dump') else vars(raw_doc)
        # Preferred keys first
        for key in ['text_content', 'content', 'text', 'raw_text', 'full_text', 'body']:
            if key in d and isinstance(d[key], str) and len(d[key].strip()) > 10:
                return d[key].strip()
        # Any long string
        best = max(
            ((k, v) for k, v in d.items() if isinstance(v, str)),
            key=lambda kv: len(kv[1]),
            default=(None, ''),
        )
        if len(best[1]) > 100:
            return best[1].strip()
    except Exception:
        pass

    # Strategy 4: str() fallback
    fallback = str(raw_doc)
    if len(fallback) > 50 and not fallback.startswith('<'):
        return fallback

    return ""


def chunk_document(raw_doc, doc_id: str, chunk_size: int = 300, overlap: int = 50) -> List[Chunk]:
    """
    Chunk a RawDocument object from the ingestion agent.
    Robustly handles any attribute structure.
    """
    text = extract_text_from_raw_doc(raw_doc)

    if not text:
        return [Chunk(
            chunk_id=f"{doc_id}_chunk_0",
            doc_id=doc_id,
            text=f"[Could not extract text from {doc_id}. Attributes: {debug_raw_doc(raw_doc)[:200]}]",
            chunk_index=0,
            section="Error",
        )]

    chunks = chunk_text(text, doc_id, chunk_size, overlap)

    if not chunks:
        return [Chunk(
            chunk_id=f"{doc_id}_chunk_0",
            doc_id=doc_id,
            text=text[:500],
            chunk_index=0,
            section="General",
        )]

    return chunks