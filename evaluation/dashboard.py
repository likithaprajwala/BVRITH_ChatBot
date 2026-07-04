"""
Streamlit Evaluation Dashboard for the BVRIT RAG Chatbot.

Displays evaluation results with summary cards, dimension scores, charts, and detailed results.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from evaluation.utils import (
    get_latest_results,
    load_testcases,
    format_latency,
)
from evaluation.report import ReportGenerator, ALL_DIMENSIONS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Color scheme
COLORS = {
    "primary": "#1E88E5",
    "success": "#43A047",
    "warning": "#FB8C00",
    "danger": "#E53935",
    "info": "#00ACC1",
    "dark": "#263238",
    "light": "#ECEFF1",
}

DIMENSION_COLORS = {
    "Functional": "#1E88E5",
    "Quality": "#43A047",
    "Safety": "#FB8C00",
    "Security": "#E53935",
    "Robustness": "#8E24AA",
    "Performance": "#00ACC1",
    "Context": "#F4511E",
    "RAGAS": "#3949AB",
}


def render_summary_cards(summary: Dict[str, Any]) -> None:
    """
    Render summary metric cards at the top of the dashboard.

    Args:
        summary: Dictionary with total_tests, passed, failed, warnings, pass_rate.
    """
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Total Tests",
            value=summary.get("total_tests", 0),
            delta=None,
        )

    with col2:
        passed = summary.get("passed", 0)
        st.metric(
            label="✅ Passed",
            value=passed,
            delta=f"{summary.get('pass_rate', 0):.1f}%",
            delta_color="normal",
        )

    with col3:
        failed = summary.get("failed", 0)
        st.metric(
            label="❌ Failed",
            value=failed,
            delta=None,
        )

    with col4:
        warnings = summary.get("warnings", 0)
        st.metric(
            label="⚠️ Warnings",
            value=warnings,
            delta=None,
        )

    with col5:
        pass_rate = summary.get("pass_rate", 0.0)
        st.metric(
            label="📊 Pass Rate",
            value=f"{pass_rate:.1f}%",
            delta=None,
        )


def render_dimension_cards(dimension_scores: Dict[str, Dict[str, Any]]) -> None:
    """
    Render per-dimension score cards with progress bars.

    Args:
        dimension_scores: Dictionary mapping dimension names to score data.
    """
    st.subheader("📊 Per-Dimension Scores")

    cols = st.columns(4)
    for i, dim in enumerate(ALL_DIMENSIONS):
        data = dimension_scores.get(dim, {})
        avg_score = data.get("avg_score", 0.0)
        pass_pct = data.get("pass_pct", 0.0)
        tests = data.get("tests", 0)
        passed = data.get("passed", 0)

        color = DIMENSION_COLORS.get(dim, COLORS["primary"])

        with cols[i % 4]:
            st.markdown(
                f"""
                <div style="
                    background: white;
                    border-radius: 10px;
                    padding: 15px;
                    margin: 5px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    border-left: 4px solid {color};
                ">
                    <h4 style="margin: 0; color: {color};">{dim}</h4>
                    <div style="font-size: 24px; font-weight: bold; margin: 5px 0;">
                        {avg_score:.1f}/10
                    </div>
                    <div style="font-size: 14px; color: #666;">
                        {passed}/{tests} passed ({pass_pct:.1f}%)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Progress bar
            progress_color = (
                COLORS["success"] if pass_pct >= 70
                else COLORS["warning"] if pass_pct >= 50
                else COLORS["danger"]
            )
            st.markdown(
                f"""
                <div style="
                    width: 100%;
                    background: #e0e0e0;
                    border-radius: 10px;
                    height: 8px;
                    margin-top: 5px;
                ">
                    <div style="
                        width: {pass_pct}%;
                        background: {progress_color};
                        border-radius: 10px;
                        height: 8px;
                    "></div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_ragas_cards(ragas_metrics: Dict[str, float]) -> None:
    """
    Render RAGAS metric cards with progress bars.

    Args:
        ragas_metrics: Dictionary of RAGAS metric scores.
    """
    st.subheader("🎯 RAGAS Metrics")

    cols = st.columns(4)
    ragas_labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
    }

    for i, (metric_key, metric_label) in enumerate(ragas_labels.items()):
        score = ragas_metrics.get(metric_key, 0.0)
        score_pct = score * 100

        color = (
            COLORS["success"] if score >= 0.7
            else COLORS["warning"] if score >= 0.4
            else COLORS["danger"]
        )

        with cols[i]:
            st.markdown(
                f"""
                <div style="
                    background: white;
                    border-radius: 10px;
                    padding: 15px;
                    margin: 5px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    border-left: 4px solid {color};
                ">
                    <h4 style="margin: 0; color: {color};">{metric_label}</h4>
                    <div style="font-size: 24px; font-weight: bold; margin: 5px 0;">
                        {score:.4f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div style="
                    width: 100%;
                    background: #e0e0e0;
                    border-radius: 10px;
                    height: 8px;
                    margin-top: 5px;
                ">
                    <div style="
                        width: {score_pct}%;
                        background: {color};
                        border-radius: 10px;
                        height: 8px;
                    "></div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_bar_chart(dimension_scores: Dict[str, Dict[str, Any]]) -> None:
    """
    Render a bar chart of pass rate per dimension.

    Args:
        dimension_scores: Dictionary of per-dimension scores.
    """
    st.subheader("📈 Pass Rate by Dimension")

    df_data = []
    for dim in ALL_DIMENSIONS:
        data = dimension_scores.get(dim, {})
        df_data.append({
            "Dimension": dim,
            "Pass Rate (%)": data.get("pass_pct", 0.0),
            "Average Score": data.get("avg_score", 0.0),
            "Tests": data.get("tests", 0),
        })

    df = pd.DataFrame(df_data)

    fig = px.bar(
        df,
        x="Dimension",
        y="Pass Rate (%)",
        color="Dimension",
        color_discrete_map=DIMENSION_COLORS,
        title="Pass Rate per Dimension",
        text="Pass Rate (%)",
        height=400,
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        showlegend=False,
        yaxis_range=[0, 110],
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_pie_chart(summary: Dict[str, Any]) -> None:
    """
    Render a pie chart of pass vs fail distribution.

    Args:
        summary: Dictionary with passed, failed, warnings counts.
    """
    st.subheader("🥧 Pass vs Fail Distribution")

    labels = ["Passed", "Failed", "Warnings"]
    values = [
        summary.get("passed", 0),
        summary.get("failed", 0),
        summary.get("warnings", 0),
    ]
    colors = [COLORS["success"], COLORS["danger"], COLORS["warning"]]

    # Filter out zero values
    non_zero = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
    if not non_zero:
        st.info("No test results to display.")
        return

    fig = go.Figure(data=[
        go.Pie(
            labels=[l for l, v, c in non_zero],
            values=[v for l, v, c in non_zero],
            marker=dict(colors=[c for l, v, c in non_zero]),
            textinfo="label+percent",
            hole=0.4,
        )
    ])
    fig.update_layout(
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_radar_chart(dimension_scores: Dict[str, Dict[str, Any]]) -> None:
    """
    Render a radar chart showing all 8 dimensions.

    Args:
        dimension_scores: Dictionary of per-dimension scores.
    """
    st.subheader("🕸️ Eight Dimensions Radar")

    dims = []
    scores = []
    for dim in ALL_DIMENSIONS:
        data = dimension_scores.get(dim, {})
        dims.append(dim)
        scores.append(data.get("avg_score", 0.0))

    fig = go.Figure(data=go.Scatterpolar(
        r=scores,
        theta=dims,
        fill="toself",
        line=dict(color=COLORS["primary"], width=2),
        marker=dict(color=COLORS["primary"], size=8),
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickvals=[0, 2, 4, 6, 8, 10],
            ),
        ),
        showlegend=False,
        height=450,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_latency_chart(test_results: List[Dict[str, Any]]) -> None:
    """
    Render a line chart of latency per test.

    Args:
        test_results: List of test result dictionaries.
    """
    st.subheader("⏱️ Latency per Test")

    if not test_results:
        st.info("No test results to display.")
        return

    df_data = []
    for i, t in enumerate(test_results):
        df_data.append({
            "Test #": i + 1,
            "Latency (s)": t.get("latency", 0.0),
            "Dimension": t.get("dimension", "Unknown"),
            "ID": t.get("id", f"TC_{i+1:03d}"),
        })

    df = pd.DataFrame(df_data)

    fig = px.line(
        df,
        x="Test #",
        y="Latency (s)",
        color="Dimension",
        color_discrete_map=DIMENSION_COLORS,
        markers=True,
        title="Response Latency per Test",
        hover_data=["ID", "Dimension"],
        height=400,
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Latency (seconds)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_results_table(test_results: List[Dict[str, Any]]) -> None:
    """
    Render a detailed results table with expandable rows.

    Args:
        test_results: List of test result dictionaries.
    """
    st.subheader("📋 Detailed Test Results")

    if not test_results:
        st.info("No test results to display.")
        return

    # Create summary dataframe
    table_data = []
    for t in test_results:
        status = "✅ PASS" if t.get("pass") else "❌ FAIL"
        table_data.append({
            "ID": t.get("id", ""),
            "Dimension": t.get("dimension", ""),
            "Question": t.get("question", "")[:60] + ("..." if len(t.get("question", "")) > 60 else ""),
            "Status": status,
            "Score": f"{t.get('judge_score', 0)}/10",
            "Latency": f"{t.get('latency', 0):.2f}s",
            "Chunks": t.get("retrieved_chunk_count", 0),
        })

    df = pd.DataFrame(table_data)

    # Color the status column
    def color_status(val):
        if "PASS" in val:
            return f"color: {COLORS['success']}; font-weight: bold"
        return f"color: {COLORS['danger']}; font-weight: bold"

    styled_df = df.style.map(color_status, subset=["Status"])

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400,
    )

    # Expandable failed test details
    failed_tests = [t for t in test_results if not t.get("pass")]
    if failed_tests:
        st.subheader("🔍 Failed Test Details")
        for t in failed_tests:
            with st.expander(
                f"❌ [{t.get('dimension', '')}] {t.get('question', '')[:80]}..."
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Question:**")
                    st.info(t.get("question", ""))
                    st.markdown("**Expected Answer:**")
                    st.info(t.get("expected_answer", ""))
                    st.markdown("**Actual Answer:**")
                    st.error(t.get("actual_answer", ""))

                with col2:
                    st.markdown("**Judge Score:**")
                    st.metric("Score", f"{t.get('judge_score', 0)}/10")
                    st.markdown("**Judge Reason:**")
                    st.warning(t.get("judge_reason", ""))
                    st.markdown("**Suggestions:**")
                    st.info(t.get("judge_suggestions", ""))

                # Retrieved chunks
                st.markdown("**Retrieved Chunks:**")
                chunks = t.get("retrieved_chunks", [])
                for i, chunk in enumerate(chunks[:3]):
                    with st.container():
                        st.markdown(
                            f"*Chunk {i+1}* — Section: **{chunk.get('section', 'Unknown')}** "
                            f"— Score: {chunk.get('score', 0):.4f}"
                        )
                        st.caption(chunk.get("content", "")[:200] + "...")

                # Failure analysis
                if "failure_analysis" in t:
                    fa = t["failure_analysis"]
                    st.markdown("**Root Cause:**")
                    st.error(fa.get("root_cause", "Unknown"))
                    st.markdown("**Recommended Fix:**")
                    st.success(fa.get("recommended_fix", "No fix suggested"))


def render_recommendations(recommendations: List[Dict[str, str]]) -> None:
    """
    Render recommendations section.

    Args:
        recommendations: List of recommendation dictionaries.
    """
    st.subheader("💡 Recommendations")

    if not recommendations:
        st.info("No recommendations available.")
        return

    for rec in recommendations:
        severity = rec.get("severity", "INFO")
        message = rec.get("message", "")
        suggestion = rec.get("suggestion", "")

        if severity == "CRITICAL":
            st.error(f"**{message}**")
        elif severity == "WARNING":
            st.warning(f"**{message}**")
        else:
            st.info(f"**{message}**")

        if suggestion:
            st.caption(f"💡 *Suggestion:* {suggestion}")


def render_download_buttons(report: Dict[str, Any]) -> None:
    """
    Render download buttons for JSON, CSV, and report formats.

    Args:
        report: The complete evaluation report dictionary.
    """
    st.subheader("📥 Download Reports")

    col1, col2, col3 = st.columns(3)

    # JSON download
    with col1:
        json_str = json.dumps(report, indent=2, ensure_ascii=False)
        st.download_button(
            label="📄 Download JSON Report",
            data=json_str,
            file_name=f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )

    # CSV download (flattened results)
    with col2:
        detailed = report.get("detailed_results", [])
        if detailed:
            csv_data = []
            for r in detailed:
                csv_data.append({
                    "ID": r.get("id", ""),
                    "Dimension": r.get("dimension", ""),
                    "Question": r.get("question", ""),
                    "Pass": r.get("pass", False),
                    "Score": r.get("score", 0),
                    "Latency": r.get("latency", 0),
                    "Reason": r.get("reason", ""),
                })
            df_csv = pd.DataFrame(csv_data)
            csv_str = df_csv.to_csv(index=False)
            st.download_button(
                label="📊 Download CSV Report",
                data=csv_str,
                file_name=f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # Summary text download
    with col3:
        summary_text = generate_summary_text(report)
        st.download_button(
            label="📝 Download Summary",
            data=summary_text,
            file_name=f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
        )


def generate_summary_text(report: Dict[str, Any]) -> str:
    """
    Generate a plain text summary of the evaluation report.

    Args:
        report: The evaluation report dictionary.

    Returns:
        Formatted text summary.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("BVRIT RAG Chatbot - Evaluation Report")
    lines.append("=" * 60)
    lines.append("")

    summary = report.get("summary", {})
    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total Tests: {summary.get('total_tests', 0)}")
    lines.append(f"Passed: {summary.get('passed', 0)}")
    lines.append(f"Failed: {summary.get('failed', 0)}")
    lines.append(f"Warnings: {summary.get('warnings', 0)}")
    lines.append(f"Pass Rate: {summary.get('pass_rate', 0)}%")
    lines.append("")

    lines.append("PER-DIMENSION SCORES")
    lines.append("-" * 40)
    dim_scores = report.get("dimension_scores", {})
    for dim in ALL_DIMENSIONS:
        data = dim_scores.get(dim, {})
        lines.append(f"{dim}: {data.get('avg_score', 0)}/10 ({data.get('passed', 0)}/{data.get('tests', 0)} passed)")
    lines.append("")

    weakest = report.get("weakest_dimension", {})
    lines.append(f"WEAKEST DIMENSION: {weakest.get('dimension', 'N/A')} ({weakest.get('score', 0)}/10)")
    lines.append("")

    lines.append("RAGAS METRICS")
    lines.append("-" * 40)
    for metric, score in report.get("ragas_metrics", {}).items():
        lines.append(f"{metric}: {score:.4f}")
    lines.append("")

    lines.append("RECOMMENDATIONS")
    lines.append("-" * 40)
    for rec in report.get("recommendations", []):
        lines.append(f"[{rec.get('severity', 'INFO')}] {rec.get('message', '')}")
        if rec.get("suggestion"):
            lines.append(f"  Suggestion: {rec['suggestion']}")
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def render_evaluation_dashboard() -> None:
    """
    Main entry point for the evaluation dashboard.
    Renders the complete dashboard UI.
    """
    st.markdown(
        """
        <h1 style="text-align: center; color: #1E88E5;">
            🧪 Evaluation Dashboard
        </h1>
        <p style="text-align: center; color: #666;">
            Comprehensive evaluation of the BVRIT RAG Chatbot across 8 dimensions
        </p>
        <hr>
        """,
        unsafe_allow_html=True,
    )

    # Load latest results
    data = get_latest_results()
    test_results = data.get("test_results", [])
    ragas_scores = data.get("ragas_scores", {})
    report = data.get("report", {})

    if not test_results:
        st.warning(
            "No evaluation results found. Please run an evaluation first "
            "using the 'Run Evaluation' button in the sidebar."
        )
        return

    # Summary
    summary = report.get("summary", {})
    if not summary:
        # Compute from test results
        total = len(test_results)
        passed = sum(1 for t in test_results if t.get("pass", False))
        failed = total - passed
        warnings = sum(1 for t in test_results if 5 <= t.get("judge_score", 10) < 7)
        pass_rate = round((passed / total * 100), 2) if total > 0 else 0.0
        summary = {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "pass_rate": pass_rate,
        }

    # Render summary cards
    render_summary_cards(summary)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Dimension scores
    dimension_scores = report.get("dimension_scores", {})
    if not dimension_scores:
        # Compute from test results
        rg = ReportGenerator()
        dimension_scores = rg._compute_dimension_scores(test_results)

    render_dimension_cards(dimension_scores)

    st.markdown("<hr>", unsafe_allow_html=True)

    # RAGAS metrics
    render_ragas_cards(ragas_scores)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Charts row
    col1, col2 = st.columns(2)
    with col1:
        render_bar_chart(dimension_scores)
    with col2:
        render_pie_chart(summary)

    # Second row of charts
    col1, col2 = st.columns(2)
    with col1:
        render_radar_chart(dimension_scores)
    with col2:
        render_latency_chart(test_results)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Results table
    render_results_table(test_results)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Recommendations
    recommendations = report.get("recommendations", [])
    if not recommendations and dimension_scores:
        rg = ReportGenerator()
        weakest = rg._find_weakest_dimension(dimension_scores)
        recommendations = rg._generate_recommendations(
            dimension_scores, ragas_scores, weakest
        )

    render_recommendations(recommendations)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Download buttons
    if report:
        render_download_buttons(report)
    else:
        # Build a temporary report for download
        rg = ReportGenerator()
        temp_report = rg.generate_report(test_results, ragas_scores)
        render_download_buttons(temp_report)


if __name__ == "__main__":
    render_evaluation_dashboard()