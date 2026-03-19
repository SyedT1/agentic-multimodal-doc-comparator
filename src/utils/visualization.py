"""
Visualization utilities for displaying similarity results.
"""
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any
import difflib

from models.similarity import SimilarityReport, ModalityScore


def create_similarity_gauge(score: float, title: str = "Overall Similarity") -> go.Figure:
    """
    Create a gauge chart showing similarity score.

    Args:
        score: Similarity score (0.0 to 1.0)
        title: Chart title

    Returns:
        Plotly figure
    """
    # Determine color based on score
    if score >= 0.7:
        color = "green"
    elif score >= 0.4:
        color = "orange"
    else:
        color = "red"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score * 100,  # Convert to percentage
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#ffcccc'},
                {'range': [40, 70], 'color': '#fff4cc'},
                {'range': [70, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def create_modality_breakdown_chart(report: SimilarityReport) -> go.Figure:
    """
    Create a bar chart showing per-modality similarity scores.

    Args:
        report: SimilarityReport object

    Returns:
        Plotly figure
    """
    modalities = []
    scores = []
    weights = []

    if report.text_score:
        modalities.append("Text")
        scores.append(report.text_score.score * 100)
        weights.append(report.weights_used.get("text", 0) * 100)

    if report.table_score:
        modalities.append("Table")
        scores.append(report.table_score.score * 100)
        weights.append(report.weights_used.get("table", 0) * 100)

    # Create bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Similarity Score',
        x=modalities,
        y=scores,
        marker_color='lightblue',
        text=[f"{s:.1f}%" for s in scores],
        textposition='auto',
    ))

    fig.add_trace(go.Bar(
        name='Weight',
        x=modalities,
        y=weights,
        marker_color='lightcoral',
        text=[f"{w:.0f}%" for w in weights],
        textposition='auto',
    ))

    fig.update_layout(
        title="Per-Modality Similarity Breakdown",
        xaxis_title="Modality",
        yaxis_title="Percentage (%)",
        yaxis_range=[0, 100],
        barmode='group',
        height=400,
        showlegend=True
    )

    return fig


def format_matched_sections(matched_sections: List[Dict[str, Any]]) -> str:
    """
    Format matched sections for display.

    Args:
        matched_sections: List of matched section dictionaries

    Returns:
        Formatted string
    """
    if not matched_sections:
        return "No matched sections found."

    output = []
    for idx, section in enumerate(matched_sections, start=1):
        section_type = section.get("type", "unknown")
        similarity = section.get("similarity", 0.0)

        output.append(f"**Match {idx}** ({section_type.upper()}) - Similarity: {similarity:.2%}")
        output.append("")

        if section_type == "text":
            output.append(f"📄 Doc 1 (Page {section.get('doc1_page', '?')}):")
            output.append(f"```\n{section.get('doc1_content', '')}\n```")
            output.append("")
            output.append(f"📄 Doc 2 (Page {section.get('doc2_page', '?')}):")
            output.append(f"```\n{section.get('doc2_content', '')}\n```")

        elif section_type == "table":
            output.append(f"📊 Doc 1 Table (Page {section.get('doc1_page', '?')}):")
            output.append(f"_{section.get('doc1_schema', '')}_")
            output.append("")
            output.append(f"📊 Doc 2 Table (Page {section.get('doc2_page', '?')}):")
            output.append(f"_{section.get('doc2_schema', '')}_")

        output.append("")
        output.append("---")
        output.append("")

    return "\n".join(output)


def generate_diff_html(text1: str, text2: str) -> str:
    """
    Generate HTML diff highlighting differences between two texts.

    Args:
        text1: First text
        text2: Second text

    Returns:
        HTML string with diff highlighting
    """
    # Split into words for better diff
    words1 = text1.split()
    words2 = text2.split()

    # Generate diff
    diff = difflib.ndiff(words1, words2)

    html_parts = []
    html_parts.append('<div style="font-family: monospace; line-height: 1.5;">')

    for item in diff:
        if item.startswith('  '):  # Unchanged
            word = item[2:]
            html_parts.append(f'<span>{word} </span>')
        elif item.startswith('- '):  # Removed from text1
            word = item[2:]
            html_parts.append(f'<span style="background-color: #ffcccc; text-decoration: line-through;">{word} </span>')
        elif item.startswith('+ '):  # Added in text2
            word = item[2:]
            html_parts.append(f'<span style="background-color: #ccffcc;">{word} </span>')

    html_parts.append('</div>')

    return ''.join(html_parts)


def create_score_legend() -> str:
    """
    Create a legend explaining similarity scores.

    Returns:
        Markdown formatted legend
    """
    legend = """
    ### 📊 Similarity Score Guide

    - **90-100%**: Nearly identical documents
    - **70-89%**: Very similar with minor differences
    - **40-69%**: Moderately similar with notable differences
    - **0-39%**: Significantly different documents
    """
    return legend
