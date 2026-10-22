"""Page-aware chunks with bounded overlap and table content for retrieval."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    section: str = ""
    page: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_text(text, doc_id, chunk_size=300, overlap=50):
    if chunk_size < 20 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk size must be at least 20 words and overlap smaller than chunk size")
    words, chunks = text.split(), []
    for start in range(0, len(words), chunk_size - overlap):
        window = words[start:start + chunk_size]
        if not window:
            break
        index = len(chunks)
        chunks.append(Chunk(f"{doc_id}_chunk_{index}", doc_id, " ".join(window), index,
                            metadata={"word_count": len(window)}))
        if start + chunk_size >= len(words):
            break
    return chunks


def extract_text_from_raw_doc(raw_doc):
    return getattr(raw_doc, "raw_text", "") or "\n".join(p.get("text", "") for p in raw_doc.pages)


def debug_raw_doc(raw_doc):
    return f"{raw_doc.filename}: {len(raw_doc.pages)} pages"


def chunk_document(raw_doc, doc_id, chunk_size=300, overlap=50):
    chunks = []
    pages = raw_doc.pages or [{"page_num": 1, "text": extract_text_from_raw_doc(raw_doc)}]
    for page in pages:
        text = page.get("text", "")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for chunk in chunk_text(text, doc_id, chunk_size, overlap):
            chunk.chunk_index = len(chunks)
            chunk.chunk_id = f"{doc_id}_chunk_{chunk.chunk_index}"
            chunk.page = int(page.get("page_num", 1)) if raw_doc.file_type == "pdf" else 0
            chunk.section = lines[0][:100] if lines else "Document content"
            chunk.metadata = {"filename": raw_doc.filename, "kind": "text", **chunk.metadata}
            chunks.append(chunk)
    for table in raw_doc.raw_tables:
        text = "\n".join(" | ".join(str(cell or "") for cell in row) for row in table.get("data", []))
        for chunk in chunk_text(text, doc_id, chunk_size, overlap):
            chunk.chunk_index = len(chunks)
            chunk.chunk_id = f"{doc_id}_chunk_{chunk.chunk_index}"
            chunk.page = int(table.get("page_num", 1)) if raw_doc.file_type == "pdf" else 0
            chunk.section = f"Table {table.get('table_idx', 0) + 1}"
            chunk.metadata = {"filename": raw_doc.filename, "kind": "table", **chunk.metadata}
            chunks.append(chunk)
    return chunks
