"""UI-independent document ingestion, analysis, and comparison."""
from dataclasses import dataclass, field
import config
from agents.ingestion_agent import IngestionAgent
from agents.text_agent import TextAgent
from agents.table_agent import TableAgent
from agents.image_agent import ImageAgent
from agents.layout_agent import LayoutAgent
from agents.meta_agent import MetaAgent
from models.document import ProcessedDocument
from orchestrator.similarity_orchestrator import SimilarityOrchestrator
from utils.file_handler import validate_file


@dataclass
class ComparisonResult:
    report: object
    raw_documents: tuple
    processed_documents: tuple
    warnings: list = field(default_factory=list)


async def compare_files(path1, path2, weights=None, extended=True, progress=None):
    notify = progress or (lambda fraction, message: None)
    for path in (path1, path2):
        valid, error = validate_file(path)
        if not valid:
            raise ValueError(error)
    notify(0.05, "Reading documents")
    ingestion = IngestionAgent()
    raw = (await ingestion.process(path1), await ingestion.process(path2))
    warnings = []
    failed_modalities = set()
    for document in raw:
        if not document.raw_text.strip() and not document.raw_tables:
            raise ValueError(f"{document.filename} has no readable text or tables. For scanned PDFs, run OCR and upload a searchable PDF.")
    notify(0.15, "Comparing text and tables")
    text_agent, table_agent = TextAgent(), TableAgent()
    processed, text_vectors, table_vectors, image_vectors = [], [], [], []
    image_agent = None
    if extended:
        notify(0.25, "Preparing visual analysis")
        try:
            image_agent = ImageAgent()
        except Exception as exc:
            warnings.append(f"Image comparison unavailable: {exc}")
    for index, document in enumerate(raw):
        chunks, text_embedding = await text_agent.process(document)
        tables, table_embedding = await table_agent.process(document)
        images, image_embedding, layout, metadata = [], [], None, None
        if extended:
            for label, agent in (("images", image_agent), ("layout", LayoutAgent()), ("metadata", MetaAgent())):
                if agent is None:
                    continue
                try:
                    output = await agent.process(document)
                    if label == "images":
                        images, image_embedding = output
                    elif label == "layout":
                        layout = output
                    else:
                        metadata = output
                except Exception as exc:
                    warnings.append(f"{document.filename}: {label} analysis unavailable: {exc}")
                    failed_modalities.add(label)
        processed.append(ProcessedDocument(filename=document.filename, text_chunks=chunks, tables=tables,
            total_pages=document.total_pages, file_type=document.file_type, images=images, layout=layout, metadata=metadata))
        text_vectors.append(text_embedding)
        table_vectors.append(table_embedding)
        image_vectors.append(image_embedding)
        notify(0.5 + index * 0.25, f"Analyzed document {index + 1} of 2")
    if "images" in failed_modalities:
        image_vectors = [None, None]
    report = await SimilarityOrchestrator(weights or config.MODALITY_WEIGHTS).compare_documents(
        processed[0], text_vectors[0], table_vectors[0], processed[1], text_vectors[1], table_vectors[1],
        image_vectors[0], image_vectors[1], processed[0].layout, processed[1].layout,
        processed[0].metadata, processed[1].metadata)
    notify(1.0, "Comparison complete")
    return ComparisonResult(report, raw, tuple(processed), warnings)
