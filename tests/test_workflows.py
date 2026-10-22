import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config
from models.document import RawDocument, DocumentChunk
from models.similarity import ModalityScore
from orchestrator.scorers import compute_text_similarity, compute_table_similarity, compute_weighted_score
from rag.chunker import Chunk, chunk_document, chunk_text
from rag.graph_builder import GraphBuilder
from rag.vector_store import VectorStore
from rag.rag_pipeline import GraphRAGPipeline, PipelineState
from rag.groq_chat import GroqGraphChat, _build_context
from utils.file_handler import save_uploaded_file, validate_file
from utils.diff import changed_values, document_text, diff_html


class ConstantEmbeddings:
    def encode(self, texts, **kwargs):
        return np.array([[1., 0., 0.] for _ in texts])


class RetrievalTests(unittest.TestCase):
    def test_collections_are_isolated_and_clear_is_local(self):
        left, right = VectorStore(embedding_model=ConstantEmbeddings()), VectorStore(embedding_model=ConstantEmbeddings())
        try:
            left.add_chunks([Chunk("doc1_chunk_0", "doc1", "Invoice alpha", 0, page=3)])
            right.add_chunks([Chunk("doc1_chunk_0", "doc1", "Secret beta", 0, page=8)])
            self.assertNotEqual(left.collection_name, right.collection_name)
            self.assertEqual(left.search("invoice")[0]["text"], "Invoice alpha")
            self.assertEqual(right.search("beta")[0]["page"], 8)
            left.clear()
            self.assertEqual(left.search("invoice"), [])
            self.assertEqual(right.count(), 1)
        finally:
            left.close(); right.close()

    def test_keywords_break_semantic_ties(self):
        store = VectorStore(embedding_model=ConstantEmbeddings())
        try:
            store.add_chunks([Chunk(f"doc1_chunk_{i}", "doc1", text, i)
                              for i, text in enumerate(["Ordinary delivery report", "Invoice ZX9988 amount 9400"])])
            self.assertIn("ZX9988", store.search("ZX9988", n_results=1)[0]["text"])
        finally:
            store.close()

    def test_page_and_table_provenance_survives_chunking(self):
        doc = RawDocument("report.pdf", "pdf", [{"page_num": 7, "text": "Budget is twelve million."}],
                          "Budget is twelve million.", [{"page_num": 9, "table_idx": 0, "data": [["Staff", "24"]]}], 9)
        chunks = chunk_document(doc, "doc1")
        self.assertEqual([chunk.page for chunk in chunks], [7, 9])
        self.assertIn("24", chunks[1].text)
        self.assertEqual(chunks[1].metadata["filename"], "report.pdf")

    def test_empty_documents_do_not_become_fabricated_evidence(self):
        doc = RawDocument("empty.pdf", "pdf", [], "", [], 0)
        self.assertEqual(chunk_document(doc, "doc1"), [])
        with self.assertRaises(ValueError):
            chunk_text("words", "doc1", chunk_size=100, overlap=100)

    def test_weak_evidence_retries_once_then_abstains(self):
        hit = {"chunk_id": "doc1_chunk_0", "doc_id": "doc1", "chunk_index": 0,
               "text": "Unrelated content", "page": 1, "score": 0.0}
        context = PipelineState(is_ready=True, vector_store=Mock(), graph_builder=Mock())
        context.vector_store.search.return_value = [hit]
        context.graph_builder.retrieve.return_value = []
        chat = Mock()
        chat.plan.return_value = {"query": "question", "scope": "both"}
        pipeline = GraphRAGPipeline(chat_client=chat, rerank=False)
        answer = pipeline.query("question", context, stream=False)
        self.assertIn("enough relevant evidence", answer)
        self.assertEqual(chat.plan.call_count, 2)
        chat.chat.assert_not_called()

    def test_context_keeps_both_documents_and_stable_citations(self):
        nodes = [{"doc_id": doc, "chunk_id": f"{doc}_chunk_0", "chunk_index": 0,
                  "text": "Budget evidence " * 300, "page": 2, "filename": "report.pdf"}
                 for doc in ("doc1", "doc2")]
        context = _build_context(nodes)
        self.assertIn("[D1-C1]", context)
        self.assertIn("[D2-C1]", context)
        self.assertIn("page 2", context)

    def test_provider_can_recognize_relevant_evidence_with_a_low_rerank_signal(self):
        hit = {"chunk_id":"doc1_chunk_0","doc_id":"doc1","chunk_index":0,
               "text":"The budget is 12 million.","page":1,"score":0.1}
        context = PipelineState(is_ready=True,vector_store=Mock(),graph_builder=Mock())
        context.vector_store.search.return_value = [hit]
        context.graph_builder.retrieve.return_value = []
        chat = Mock()
        chat.plan.return_value = {"query":"budget amounts","scope":"both"}
        chat.grade.return_value = True
        chat.chat.return_value = "The budget is 12 million [D1-C1]."
        pipeline = GraphRAGPipeline(chat_client=chat,rerank=False)
        self.assertIn("12 million",pipeline.query("Compare budgets",context,stream=False))
        self.assertEqual(chat.plan.call_count,1)
        chat.chat.assert_called_once()

    def test_failed_generation_does_not_pollute_conversation(self):
        chat = GroqGraphChat("test-key")
        chat._client = Mock()
        chat._client.chat.completions.create.side_effect = RuntimeError("unavailable")
        with self.assertRaises(RuntimeError):
            chat.chat("question", [], stream=False)
        self.assertEqual(chat._history, [])


class ScoringTests(unittest.TestCase):
    def test_numeric_changes_are_visible_despite_similar_wording(self):
        removed, added = changed_values("Budget $12 million. Delivery June.", "Budget $18 million. Delivery September.")
        self.assertIn("$12 million", removed)
        self.assertIn("June", removed)
        self.assertIn("$18 million", added)
        self.assertIn("September", added)

    def test_diff_escapes_document_names_and_content(self):
        markup = diff_html("<script>alert(1)</script>", "new text", "<script>name</script>", "revised")
        self.assertNotIn("<script>", markup)
        self.assertIn("&lt;script&gt;", markup)

    def test_text_similarity_is_symmetric_and_bounded(self):
        a = [DocumentChunk("A", "text", 1)]
        b = [DocumentChunk("B", "text", 1), DocumentChunk("C", "text", 2)]
        x, y = np.array([[1., 0.]]), np.array([[1., 0.], [0., 1.]])
        first = compute_text_similarity(a, x, b, y)
        second = compute_text_similarity(b, y, a, x)
        self.assertAlmostEqual(first.score, second.score)
        self.assertAlmostEqual(first.score, 0.75)

    def test_absent_tables_do_not_inflate_similarity(self):
        table = compute_table_similarity([], np.array([]), [], np.array([]))
        score = compute_weighted_score({"text": ModalityScore(modality="text", score=0.2), "table": table},
                                       {"text": 0.6, "table": 0.4})
        self.assertAlmostEqual(score, 0.2)


class UploadTests(unittest.TestCase):
    def test_same_names_do_not_overwrite_or_escape_upload_directory(self):
        class Upload(io.BytesIO):
            name = "../../same.pdf"
            @property
            def size(self): return len(self.getvalue())
        with tempfile.TemporaryDirectory() as directory:
            first = Path(save_uploaded_file(Upload(b"%PDF-first"), Path(directory)))
            second = Path(save_uploaded_file(Upload(b"%PDF-second"), Path(directory)))
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_relative_to(directory))
            self.assertEqual(first.read_bytes(), b"%PDF-first")

    def test_fake_pdf_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fake.pdf"
            path.write_text("not a PDF")
            self.assertFalse(validate_file(str(path))[0])


if __name__ == "__main__":
    unittest.main()
