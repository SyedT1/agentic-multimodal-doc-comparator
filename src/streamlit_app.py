"""
Multi-Agent Document Comparison Streamlit App
"""
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import asyncio
import json

# Import agents and utilities
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

# Phase 2 imports (conditional based on availability)
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


# Page configuration
st.set_page_config(
    page_title="Multi-Agent Document Comparator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """Main application function."""

    # Header
    st.title("📄 Multi-Agent Document Comparator")
    st.markdown("**An agentic system to accurately match document similarity**")

    # Show architecture diagram
    with st.expander("🏗️ View System Architecture", expanded=False):
        arch_path = Path("src/img/multi_agent_doc_similarity_architecture.svg")
        if arch_path.exists():
            st.image(str(arch_path), use_container_width=True)
        else:
            st.info("Architecture diagram not found")

    st.markdown("---")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Phase 2 feature toggles
        st.subheader("Phase 2 Features")
        enable_phase2 = st.checkbox(
            "Enable Phase 2 Modalities",
            value=config.ENABLE_IMAGE_COMPARISON,
            help="Enable image, layout, and metadata comparison"
        )

        # Modality weights
        st.markdown("---")
        st.subheader("Modality Weights")

        if enable_phase2:
            # Phase 2: All 5 modalities
            text_weight = st.slider(
                "Text Weight",
                min_value=0.0,
                max_value=1.0,
                value=config.MODALITY_WEIGHTS["text"],
                step=0.05
            )
            table_weight = st.slider(
                "Table Weight",
                min_value=0.0,
                max_value=1.0,
                value=config.MODALITY_WEIGHTS["table"],
                step=0.05
            )
            image_weight = st.slider(
                "Image Weight",
                min_value=0.0,
                max_value=1.0,
                value=config.MODALITY_WEIGHTS["image"],
                step=0.05
            )
            layout_weight = st.slider(
                "Layout Weight",
                min_value=0.0,
                max_value=1.0,
                value=config.MODALITY_WEIGHTS["layout"],
                step=0.05
            )
            metadata_weight = st.slider(
                "Metadata Weight",
                min_value=0.0,
                max_value=1.0,
                value=config.MODALITY_WEIGHTS["metadata"],
                step=0.05
            )

            # Normalize weights to sum to 1.0
            total_weight = text_weight + table_weight + image_weight + layout_weight + metadata_weight
            if total_weight > 0:
                weights = {
                    "text": text_weight / total_weight,
                    "table": table_weight / total_weight,
                    "image": image_weight / total_weight,
                    "layout": layout_weight / total_weight,
                    "metadata": metadata_weight / total_weight
                }
            else:
                weights = config.MODALITY_WEIGHTS

            st.info(f"Weights normalized to sum to 1.0")

        else:
            # Phase 1: Only text and tables
            text_weight = st.slider(
                "Text Weight",
                min_value=0.0,
                max_value=1.0,
                value=config.MODALITY_WEIGHTS_PHASE1["text"],
                step=0.05
            )
            table_weight = 1.0 - text_weight
            st.write(f"Table Weight: {table_weight:.2f}")

            weights = {"text": text_weight, "table": table_weight}

        # Phase status
        st.markdown("---")
        st.subheader("📋 Implementation Status")
        st.write("✅ Text comparison")
        st.write("✅ Table comparison")

        if enable_phase2:
            st.write(f"{'✅' if IMAGE_AGENT_AVAILABLE else '⚠️'} Image comparison")
            st.write(f"{'✅' if LAYOUT_AGENT_AVAILABLE else '⚠️'} Layout comparison")
            st.write(f"{'✅' if META_AGENT_AVAILABLE else '⚠️'} Metadata comparison")
        else:
            st.write("⏸️ Image comparison (disabled)")
            st.write("⏸️ Layout comparison (disabled)")
            st.write("⏸️ Metadata comparison (disabled)")

    # Main content area
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📤 Document 1 (Main)")
        uploaded_file1 = st.file_uploader(
            "Upload PDF or DOCX",
            type=["pdf", "docx"],
            key="file1",
            help="Maximum file size: 50MB"
        )

    with col2:
        st.subheader("📤 Document 2 (Comparison)")
        uploaded_file2 = st.file_uploader(
            "Upload PDF or DOCX",
            type=["pdf", "docx"],
            key="file2",
            help="Maximum file size: 50MB"
        )

    # Compare button
    st.markdown("---")

    if st.button("🔍 Compare Documents", type="primary", use_container_width=True):
        if not uploaded_file1 or not uploaded_file2:
            st.error("Please upload both documents before comparing.")
            return

        # Process documents and compare
        with st.spinner("Processing documents..."):
            try:
                # Save uploaded files
                file1_path = save_uploaded_file(uploaded_file1)
                file2_path = save_uploaded_file(uploaded_file2)

                # Validate files
                valid1, error1 = validate_file(file1_path)
                valid2, error2 = validate_file(file2_path)

                if not valid1:
                    st.error(f"Document 1 error: {error1}")
                    return
                if not valid2:
                    st.error(f"Document 2 error: {error2}")
                    return

                # Process documents
                report = asyncio.run(process_and_compare(
                    file1_path,
                    file2_path,
                    weights,
                    enable_phase2
                ))

                # Display results
                display_results(report)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


async def process_and_compare(file1_path: str, file2_path: str, weights: dict, enable_phase2: bool = False):
    """
    Process two documents and compare them.

    Args:
        file1_path: Path to first document
        file2_path: Path to second document
        weights: Modality weights
        enable_phase2: Enable Phase 2 modalities (image, layout, metadata)

    Returns:
        SimilarityReport
    """
    # Initialize agents
    ingestion_agent = IngestionAgent()
    text_agent = TextAgent()
    table_agent = TableAgent()
    orchestrator = SimilarityOrchestrator(weights=weights)

    # Phase 2 agents (conditional)
    image_agent = ImageAgent() if enable_phase2 and IMAGE_AGENT_AVAILABLE else None
    layout_agent = LayoutAgent() if enable_phase2 and LAYOUT_AGENT_AVAILABLE else None
    meta_agent = MetaAgent() if enable_phase2 and META_AGENT_AVAILABLE else None

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step 1: Ingest documents
    status_text.text("⏳ Ingesting documents...")
    progress_bar.progress(10)

    raw_doc1 = await ingestion_agent.process(file1_path)
    raw_doc2 = await ingestion_agent.process(file2_path)

    progress_bar.progress(15)

    # Step 2: Extract text
    status_text.text("⏳ Extracting and embedding text...")

    text_chunks1, text_embeddings1 = await text_agent.process(raw_doc1)
    text_chunks2, text_embeddings2 = await text_agent.process(raw_doc2)

    progress_bar.progress(30)

    # Step 3: Extract tables
    status_text.text("⏳ Extracting and embedding tables...")

    tables1, table_embeddings1 = await table_agent.process(raw_doc1)
    tables2, table_embeddings2 = await table_agent.process(raw_doc2)

    progress_bar.progress(45)

    # Phase 2: Extract images
    images1, image_embeddings1 = [], None
    images2, image_embeddings2 = [], None
    if image_agent:
        status_text.text("⏳ Extracting and embedding images...")
        try:
            images1, image_embeddings1 = await image_agent.process(raw_doc1)
            images2, image_embeddings2 = await image_agent.process(raw_doc2)
        except Exception as e:
            st.warning(f"Image extraction failed: {e}")

    progress_bar.progress(60)

    # Phase 2: Extract layout
    layout1, layout2 = None, None
    if layout_agent:
        status_text.text("⏳ Analyzing document structure...")
        try:
            layout1 = await layout_agent.process(raw_doc1)
            layout2 = await layout_agent.process(raw_doc2)
        except Exception as e:
            st.warning(f"Layout analysis failed: {e}")

    progress_bar.progress(70)

    # Phase 2: Extract metadata
    metadata1, metadata2 = None, None
    if meta_agent:
        status_text.text("⏳ Extracting metadata...")
        try:
            metadata1 = await meta_agent.process(raw_doc1)
            metadata2 = await meta_agent.process(raw_doc2)
        except Exception as e:
            st.warning(f"Metadata extraction failed: {e}")

    progress_bar.progress(80)

    # Create processed documents
    processed_doc1 = ProcessedDocument(
        filename=raw_doc1.filename,
        text_chunks=text_chunks1,
        tables=tables1,
        total_pages=raw_doc1.total_pages,
        file_type=raw_doc1.file_type,
        images=images1,
        layout=layout1,
        metadata=metadata1
    )

    processed_doc2 = ProcessedDocument(
        filename=raw_doc2.filename,
        text_chunks=text_chunks2,
        tables=tables2,
        total_pages=raw_doc2.total_pages,
        file_type=raw_doc2.file_type,
        images=images2,
        layout=layout2,
        metadata=metadata2
    )

    # Compare documents
    status_text.text("⏳ Comparing documents...")

    report = await orchestrator.compare_documents(
        processed_doc1,
        text_embeddings1,
        table_embeddings1,
        processed_doc2,
        text_embeddings2,
        table_embeddings2,
        # Phase 2 parameters
        image_embeddings1,
        image_embeddings2,
        layout1,
        layout2,
        metadata1,
        metadata2
    )

    progress_bar.progress(100)
    status_text.text("✅ Comparison complete!")

    return report


def display_results(report):
    """
    Display comparison results.

    Args:
        report: SimilarityReport object
    """
    st.markdown("---")
    st.header("📊 Comparison Results")

    # Overall similarity gauge
    col1, col2 = st.columns([1, 1])

    with col1:
        gauge_fig = create_similarity_gauge(report.overall_score)
        st.plotly_chart(gauge_fig, use_container_width=True)

    with col2:
        st.markdown(create_score_legend())

    # Modality breakdown
    st.markdown("---")
    st.subheader("📈 Per-Modality Breakdown")

    breakdown_fig = create_modality_breakdown_chart(report)
    st.plotly_chart(breakdown_fig, use_container_width=True)

    # Detailed scores
    cols = st.columns(5)

    with cols[0]:
        if report.text_score:
            st.metric(
                "Text Similarity",
                f"{report.text_score.score:.1%}",
                f"{report.text_score.details.get('num_matches', 0)} matches"
            )

    with cols[1]:
        if report.table_score:
            st.metric(
                "Table Similarity",
                f"{report.table_score.score:.1%}",
                f"{report.table_score.details.get('num_matches', 0)} matches"
            )

    with cols[2]:
        if report.image_score:
            st.metric(
                "Image Similarity",
                f"{report.image_score.score:.1%}",
                f"{report.image_score.details.get('num_matches', 0)} matches"
            )

    with cols[3]:
        if report.layout_score:
            st.metric(
                "Layout Similarity",
                f"{report.layout_score.score:.1%}",
                f"{report.layout_score.details.get('num_metrics', 0)} metrics"
            )

    with cols[4]:
        if report.metadata_score:
            st.metric(
                "Metadata Similarity",
                f"{report.metadata_score.score:.1%}",
                f"{report.metadata_score.details.get('num_fields_compared', 0)} fields"
            )

    # Matched sections
    st.markdown("---")
    st.subheader("🔗 Top Matched Sections")

    if report.matched_sections:
        formatted_sections = format_matched_sections(report.matched_sections[:10])
        st.markdown(formatted_sections)
    else:
        st.info("No significant matches found between documents.")

    # Phase 2: Additional modality details
    if report.image_score or report.layout_score or report.metadata_score:
        st.markdown("---")
        st.subheader("🎨 Phase 2 Modality Details")

        # Image matches
        if report.image_score and report.image_score.matched_items:
            with st.expander(f"🖼️ Image Matches ({len(report.image_score.matched_items)} found)", expanded=False):
                for idx, match in enumerate(report.image_score.matched_items[:5], 1):
                    st.markdown(f"**Match {idx}** - Similarity: {match['similarity']:.2%}")
                    st.write(f"Doc1: Page {match['doc1_page']}, Size: {match['doc1_size']}")
                    st.write(f"Doc2: Page {match['doc2_page']}, Size: {match['doc2_size']}")
                    st.markdown("---")

        # Layout details
        if report.layout_score:
            with st.expander(f"📐 Layout Analysis (Score: {report.layout_score.score:.1%})", expanded=False):
                for metric, value in report.layout_score.details.items():
                    if metric != "num_metrics":
                        st.metric(metric.replace("_", " ").title(), f"{value:.2%}")

        # Metadata matches
        if report.metadata_score and report.metadata_score.matched_items:
            with st.expander(f"📋 Metadata Comparison ({len(report.metadata_score.matched_items)} fields)", expanded=False):
                for match in report.metadata_score.matched_items:
                    st.markdown(f"**{match['field'].title()}** - Similarity: {match['similarity']:.2%}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"Doc1: {match['doc1_value']}")
                    with col2:
                        st.write(f"Doc2: {match['doc2_value']}")
                    st.markdown("---")

    # Download report
    st.markdown("---")
    report_json = json.dumps(report.model_dump(), indent=2, default=str)

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        st.download_button(
            label="📥 Download Report (JSON)",
            data=report_json,
            file_name=f"similarity_report_{report.timestamp.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main()