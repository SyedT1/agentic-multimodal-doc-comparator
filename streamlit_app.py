"""
Multi-Agent Document Comparison Streamlit App
"""
import streamlit as st
import asyncio
import json
from pathlib import Path

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
        arch_path = Path("img/multi_agent_doc_similarity_architecture.svg")
        if arch_path.exists():
            st.image(str(arch_path), use_container_width=True)
        else:
            st.info("Architecture diagram not found")

    st.markdown("---")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Modality weights
        st.subheader("Modality Weights")
        text_weight = st.slider(
            "Text Weight",
            min_value=0.0,
            max_value=1.0,
            value=config.MODALITY_WEIGHTS["text"],
            step=0.05
        )
        table_weight = 1.0 - text_weight

        st.write(f"Table Weight: {table_weight:.2f}")

        # Phase info
        st.markdown("---")
        st.subheader("📋 Phase 1 Implementation")
        st.write("✅ Text comparison")
        st.write("✅ Table comparison")
        st.write("⏳ Image comparison (Phase 2)")
        st.write("⏳ Layout comparison (Phase 2)")
        st.write("⏳ Metadata comparison (Phase 2)")

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
                    {"text": text_weight, "table": table_weight}
                ))

                # Display results
                display_results(report)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


async def process_and_compare(file1_path: str, file2_path: str, weights: dict):
    """
    Process two documents and compare them.

    Args:
        file1_path: Path to first document
        file2_path: Path to second document
        weights: Modality weights

    Returns:
        SimilarityReport
    """
    # Initialize agents
    ingestion_agent = IngestionAgent()
    text_agent = TextAgent()
    table_agent = TableAgent()
    orchestrator = SimilarityOrchestrator(weights=weights)

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step 1: Ingest documents
    status_text.text("⏳ Ingesting documents...")
    progress_bar.progress(10)

    raw_doc1 = await ingestion_agent.process(file1_path)
    raw_doc2 = await ingestion_agent.process(file2_path)

    progress_bar.progress(25)

    # Step 2: Extract text
    status_text.text("⏳ Extracting and embedding text...")

    text_chunks1, text_embeddings1 = await text_agent.process(raw_doc1)
    text_chunks2, text_embeddings2 = await text_agent.process(raw_doc2)

    progress_bar.progress(50)

    # Step 3: Extract tables
    status_text.text("⏳ Extracting and embedding tables...")

    tables1, table_embeddings1 = await table_agent.process(raw_doc1)
    tables2, table_embeddings2 = await table_agent.process(raw_doc2)

    progress_bar.progress(75)

    # Step 4: Create processed documents
    processed_doc1 = ProcessedDocument(
        filename=raw_doc1.filename,
        text_chunks=text_chunks1,
        tables=tables1,
        total_pages=raw_doc1.total_pages,
        file_type=raw_doc1.file_type
    )

    processed_doc2 = ProcessedDocument(
        filename=raw_doc2.filename,
        text_chunks=text_chunks2,
        tables=tables2,
        total_pages=raw_doc2.total_pages,
        file_type=raw_doc2.file_type
    )

    # Step 5: Compare documents
    status_text.text("⏳ Comparing documents...")

    report = await orchestrator.compare_documents(
        processed_doc1,
        text_embeddings1,
        table_embeddings1,
        processed_doc2,
        text_embeddings2,
        table_embeddings2
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
    col1, col2 = st.columns(2)

    with col1:
        if report.text_score:
            st.metric(
                "Text Similarity",
                f"{report.text_score.score:.1%}",
                f"{report.text_score.details.get('num_matches', 0)} matches"
            )

    with col2:
        if report.table_score:
            st.metric(
                "Table Similarity",
                f"{report.table_score.score:.1%}",
                f"{report.table_score.details.get('num_matches', 0)} matches"
            )

    # Matched sections
    st.markdown("---")
    st.subheader("🔗 Top Matched Sections")

    if report.matched_sections:
        formatted_sections = format_matched_sections(report.matched_sections[:5])
        st.markdown(formatted_sections)
    else:
        st.info("No significant matches found between documents.")

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
