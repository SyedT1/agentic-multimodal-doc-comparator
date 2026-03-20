"""
Multi-Agent Document Comparison Streamlit App
+ Graph RAG Chat Tab (new)
"""
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import asyncio
import json

from agents.ingestion_agent import IngestionAgent
from agents.text_agent import TextAgent
from agents.table_agent import TableAgent
from orchestrator.similarity_orchestrator import SimilarityOrchestrator
from utils.file_handler import save_uploaded_file, validate_file, get_file_type
from utils.visualization import (
    create_similarity_gauge,
    create_modality_breakdown_chart,
    format_matched_sections,
    create_score_legend
)
from models.document import ProcessedDocument
import config

# Graph RAG imports
from rag.rag_pipeline import GraphRAGPipeline, PipelineState

# Phase 2 imports (conditional)
try:
    from agents.image_agent import ImageAgent
    IMAGE_AGENT_AVAILABLE = True
except ImportError:
    IMAGE_AGENT_AVAILABLE = False

try:
    from agents.layout_agent import LayoutAgent
    LAYOUT_AGENT_AVAILABLE = True
except ImportError:
    LAYOUT_AGENT_AVAILABLE = False

try:
    from agents.meta_agent import MetaAgent
    META_AGENT_AVAILABLE = True
except ImportError:
    META_AGENT_AVAILABLE = False


st.set_page_config(
    page_title="Multi-Agent Document Comparator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    st.title("📄 Multi-Agent Document Comparator + Graph RAG Chat")
    st.markdown("**Agentic document similarity · Knowledge Graph RAG · Groq-powered chat**")

    with st.expander("🏗️ View System Architecture", expanded=False):
        arch_path = Path("src/img/multi_agent_doc_similarity_architecture.svg")
        if arch_path.exists():
            st.image(str(arch_path), use_container_width=True)

    st.markdown("---")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("Phase 2 Features")
        enable_phase2 = st.checkbox(
            "Enable Phase 2 Modalities",
            value=config.ENABLE_IMAGE_COMPARISON,
            help="Enable image, layout, and metadata comparison"
        )
        st.markdown("---")
        st.subheader("Modality Weights")

        if enable_phase2:
            text_weight   = st.slider("Text Weight",     0.0, 1.0, config.MODALITY_WEIGHTS["text"],     0.05)
            table_weight  = st.slider("Table Weight",    0.0, 1.0, config.MODALITY_WEIGHTS["table"],    0.05)
            image_weight  = st.slider("Image Weight",    0.0, 1.0, config.MODALITY_WEIGHTS["image"],    0.05)
            layout_weight = st.slider("Layout Weight",   0.0, 1.0, config.MODALITY_WEIGHTS["layout"],   0.05)
            meta_weight   = st.slider("Metadata Weight", 0.0, 1.0, config.MODALITY_WEIGHTS["metadata"], 0.05)

            total = text_weight + table_weight + image_weight + layout_weight + meta_weight
            if total > 0:
                weights = {
                    "text":     text_weight   / total,
                    "table":    table_weight  / total,
                    "image":    image_weight  / total,
                    "layout":   layout_weight / total,
                    "metadata": meta_weight   / total,
                }
            else:
                weights = config.MODALITY_WEIGHTS
            st.info("Weights normalised to 1.0")
        else:
            text_weight  = st.slider("Text Weight", 0.0, 1.0, config.MODALITY_WEIGHTS_PHASE1["text"], 0.05)
            table_weight = 1.0 - text_weight
            st.write(f"Table Weight: {table_weight:.2f}")
            weights = {"text": text_weight, "table": table_weight}

        st.markdown("---")
        st.subheader("📋 Status")
        st.write("✅ Text comparison")
        st.write("✅ Table comparison")
        if enable_phase2:
            st.write(f"{'✅' if IMAGE_AGENT_AVAILABLE  else '⚠️'} Image comparison")
            st.write(f"{'✅' if LAYOUT_AGENT_AVAILABLE else '⚠️'} Layout comparison")
            st.write(f"{'✅' if META_AGENT_AVAILABLE   else '⚠️'} Metadata comparison")
        else:
            st.write("⏸️ Image / Layout / Metadata (disabled)")

        st.markdown("---")
        st.subheader("🔗 Graph RAG Settings")
        chunk_size    = st.slider("Chunk size (words)",   100, 600, 300, 50)
        chunk_overlap = st.slider("Overlap (words)",       20, 150,  50, 10)
        top_k         = st.slider("Vector top-k",           3,  15,   5,  1)
        graph_hops    = st.slider("Graph hops",             1,   4,   2,  1)

    # ── Main tabs ─────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["📊 Document Comparison", "💬 Graph RAG Chat"])

    # ── Session state init ────────────────────────────────────────────────────
    for key in ["raw_doc1", "raw_doc2", "rag_state", "rag_pipeline", "chat_history"]:
        if key not in st.session_state:
            st.session_state[key] = None if key != "chat_history" else []

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — Comparison
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📤 Document 1 (Main)")
            uploaded_file1 = st.file_uploader(
                "Upload PDF or DOCX", type=["pdf", "docx"], key="file1",
                help="Maximum file size: 50MB"
            )

        with col2:
            st.subheader("📤 Document 2 (Comparison)")
            uploaded_file2 = st.file_uploader(
                "Upload PDF or DOCX", type=["pdf", "docx"], key="file2",
                help="Maximum file size: 50MB"
            )

        st.markdown("---")

        if st.button("🔍 Compare Documents", type="primary", use_container_width=True):
            if not uploaded_file1 or not uploaded_file2:
                st.error("Please upload both documents before comparing.")
            else:
                with st.spinner("Processing documents..."):
                    try:
                        file1_path = save_uploaded_file(uploaded_file1)
                        file2_path = save_uploaded_file(uploaded_file2)

                        valid1, error1 = validate_file(file1_path)
                        valid2, error2 = validate_file(file2_path)

                        if not valid1:
                            st.error(f"Document 1 error: {error1}"); st.stop()
                        if not valid2:
                            st.error(f"Document 2 error: {error2}"); st.stop()

                        report, raw_doc1, raw_doc2 = asyncio.run(
                            process_and_compare(file1_path, file2_path, weights, enable_phase2)
                        )

                        # Store raw docs for Graph RAG tab
                        st.session_state["raw_doc1"] = raw_doc1
                        st.session_state["raw_doc2"] = raw_doc2
                        # Reset any previous RAG state
                        st.session_state["rag_state"] = None
                        st.session_state["chat_history"] = []

                        display_results(report)

                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — Graph RAG Chat
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("💬 Chat with your Documents (Graph RAG + Groq)")

        docs_ready = (
            st.session_state["raw_doc1"] is not None
            and st.session_state["raw_doc2"] is not None
        )

        if not docs_ready:
            st.info("📂 Please upload and compare documents in the **Document Comparison** tab first.")
        else:
            # Load Groq API key from environment (Hugging Face Spaces secrets)
            groq_key = os.environ.get("GROQ_API_KEY", "")

            if not groq_key:
                st.warning("⚠️ GROQ_API_KEY not found in environment. Please set it in Hugging Face Spaces secrets.")

            col_build, col_reset = st.columns([2, 1])

            with col_build:
                build_btn = st.button(
                    "🔨 Build Graph RAG Index",
                    disabled=not groq_key,
                    help="Chunks docs → embeds → builds vector DB + knowledge graph",
                )

            with col_reset:
                if st.button("🔄 Reset Chat"):
                    st.session_state["chat_history"] = []
                    if st.session_state["rag_pipeline"]:
                        st.session_state["rag_pipeline"].reset_chat()
                    st.rerun()

            if build_btn:
                with st.spinner("Chunking, embedding, building knowledge graph — this takes ~30s…"):
                    pipeline = GraphRAGPipeline(
                        groq_api_key=groq_key,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        top_k_vector=top_k,
                        graph_hops=graph_hops,
                    )
                    rag_state = pipeline.ingest(
                        st.session_state["raw_doc1"],
                        st.session_state["raw_doc2"],
                    )
                    st.session_state["rag_pipeline"] = pipeline
                    st.session_state["rag_state"]    = rag_state
                    st.session_state["chat_history"] = []

                st.success("✅ Graph RAG index ready!")

                s = rag_state.stats
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Doc 1 Chunks",  s.get("doc1_chunks", 0))
                c2.metric("Doc 2 Chunks",  s.get("doc2_chunks", 0))
                c3.metric("Graph Nodes",   s.get("nodes", 0))
                c4.metric("Graph Edges",   s.get("edges", 0))

                with st.expander("Edge type breakdown"):
                    for etype, cnt in s.get("edge_types", {}).items():
                        st.write(f"**{etype}**: {cnt}")

            # ── Chat UI ───────────────────────────────────────────────────────
            rag_ready = st.session_state["rag_state"] is not None

            if rag_ready:
                for msg in st.session_state["chat_history"]:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                if user_input := st.chat_input("Ask anything about the two documents…"):
                    st.session_state["chat_history"].append(
                        {"role": "user", "content": user_input}
                    )
                    with st.chat_message("user"):
                        st.markdown(user_input)

                    with st.chat_message("assistant"):
                        pipeline: GraphRAGPipeline = st.session_state["rag_pipeline"]
                        rag_state_obj: PipelineState = st.session_state["rag_state"]

                        response_gen = pipeline.query(user_input, rag_state_obj, stream=True)
                        full_response = st.write_stream(response_gen)

                    st.session_state["chat_history"].append(
                        {"role": "assistant", "content": full_response}
                    )
            else:
                st.info("👆 Click **Build Graph RAG Index** to start chatting. (Ensure GROQ_API_KEY is set in HF Spaces secrets)")


# ── Helpers ───────────────────────────────────────────────────────────────────

async def process_and_compare(file1_path, file2_path, weights, enable_phase2=False):
    ingestion_agent = IngestionAgent()
    text_agent      = TextAgent()
    table_agent     = TableAgent()
    orchestrator    = SimilarityOrchestrator(weights=weights)

    image_agent  = ImageAgent()  if enable_phase2 and IMAGE_AGENT_AVAILABLE  else None
    layout_agent = LayoutAgent() if enable_phase2 and LAYOUT_AGENT_AVAILABLE else None
    meta_agent   = MetaAgent()   if enable_phase2 and META_AGENT_AVAILABLE   else None

    progress_bar = st.progress(0)
    status_text  = st.empty()

    status_text.text("⏳ Ingesting documents...")
    progress_bar.progress(10)
    raw_doc1 = await ingestion_agent.process(file1_path)
    raw_doc2 = await ingestion_agent.process(file2_path)
    progress_bar.progress(15)

    status_text.text("⏳ Extracting text…")
    text_chunks1, text_embeddings1 = await text_agent.process(raw_doc1)
    text_chunks2, text_embeddings2 = await text_agent.process(raw_doc2)
    progress_bar.progress(30)

    status_text.text("⏳ Extracting tables…")
    tables1, table_embeddings1 = await table_agent.process(raw_doc1)
    tables2, table_embeddings2 = await table_agent.process(raw_doc2)
    progress_bar.progress(45)

    images1 = images2 = image_embeddings1 = image_embeddings2 = []
    if image_agent:
        status_text.text("⏳ Extracting images…")
        try:
            images1, image_embeddings1 = await image_agent.process(raw_doc1)
            images2, image_embeddings2 = await image_agent.process(raw_doc2)
        except Exception as e:
            st.warning(f"Image extraction failed: {e}")
    progress_bar.progress(60)

    layout1 = layout2 = None
    if layout_agent:
        status_text.text("⏳ Analysing layout…")
        try:
            layout1 = await layout_agent.process(raw_doc1)
            layout2 = await layout_agent.process(raw_doc2)
        except Exception as e:
            st.warning(f"Layout analysis failed: {e}")
    progress_bar.progress(70)

    metadata1 = metadata2 = None
    if meta_agent:
        status_text.text("⏳ Extracting metadata…")
        try:
            metadata1 = await meta_agent.process(raw_doc1)
            metadata2 = await meta_agent.process(raw_doc2)
        except Exception as e:
            st.warning(f"Metadata extraction failed: {e}")
    progress_bar.progress(80)

    processed_doc1 = ProcessedDocument(
        filename=raw_doc1.filename, text_chunks=text_chunks1, tables=tables1,
        total_pages=raw_doc1.total_pages, file_type=raw_doc1.file_type,
        images=images1, layout=layout1, metadata=metadata1
    )
    processed_doc2 = ProcessedDocument(
        filename=raw_doc2.filename, text_chunks=text_chunks2, tables=tables2,
        total_pages=raw_doc2.total_pages, file_type=raw_doc2.file_type,
        images=images2, layout=layout2, metadata=metadata2
    )

    status_text.text("⏳ Comparing documents…")
    report = await orchestrator.compare_documents(
        processed_doc1, text_embeddings1, table_embeddings1,
        processed_doc2, text_embeddings2, table_embeddings2,
        image_embeddings1, image_embeddings2,
        layout1, layout2, metadata1, metadata2
    )

    progress_bar.progress(100)
    status_text.text("✅ Comparison complete!")

    # Return report + raw docs (needed for Graph RAG)
    return report, raw_doc1, raw_doc2


def display_results(report):
    st.markdown("---")
    st.header("📊 Comparison Results")

    col1, col2 = st.columns([1, 1])
    with col1:
        gauge_fig = create_similarity_gauge(report.overall_score)
        st.plotly_chart(gauge_fig, use_container_width=True)
    with col2:
        st.markdown(create_score_legend())

    st.markdown("---")
    st.subheader("📈 Per-Modality Breakdown")
    breakdown_fig = create_modality_breakdown_chart(report)
    st.plotly_chart(breakdown_fig, use_container_width=True)

    cols = st.columns(5)
    scores = [
        ("Text Similarity",     report.text_score,     "num_matches"),
        ("Table Similarity",    report.table_score,    "num_matches"),
        ("Image Similarity",    report.image_score,    "num_matches"),
        ("Layout Similarity",   report.layout_score,   "num_metrics"),
        ("Metadata Similarity", report.metadata_score, "num_fields_compared"),
    ]
    for col, (label, score_obj, detail_key) in zip(cols, scores):
        if score_obj:
            col.metric(label, f"{score_obj.score:.1%}",
                       f"{score_obj.details.get(detail_key, 0)} items")

    st.markdown("---")
    st.subheader("🔗 Top Matched Sections")
    if report.matched_sections:
        st.markdown(format_matched_sections(report.matched_sections[:10]))
    else:
        st.info("No significant matches found.")

    if report.image_score or report.layout_score or report.metadata_score:
        st.markdown("---")
        st.subheader("🎨 Phase 2 Modality Details")
        if report.image_score and report.image_score.matched_items:
            with st.expander(f"🖼️ Image Matches ({len(report.image_score.matched_items)})"):
                for idx, m in enumerate(report.image_score.matched_items[:5], 1):
                    st.markdown(f"**Match {idx}** — {m['similarity']:.2%}")
        if report.layout_score:
            with st.expander(f"📐 Layout (Score: {report.layout_score.score:.1%})"):
                for k, v in report.layout_score.details.items():
                    if k != "num_metrics":
                        st.metric(k.replace("_", " ").title(), f"{v:.2%}")
        if report.metadata_score and report.metadata_score.matched_items:
            with st.expander(f"📋 Metadata ({len(report.metadata_score.matched_items)} fields)"):
                for m in report.metadata_score.matched_items:
                    st.markdown(f"**{m['field'].title()}** — {m['similarity']:.2%}")

    st.markdown("---")
    report_json = json.dumps(report.model_dump(), indent=2, default=str)
    st.download_button(
        "📥 Download Report (JSON)", data=report_json,
        file_name=f"similarity_report_{report.timestamp.strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


if __name__ == "__main__":
    main()