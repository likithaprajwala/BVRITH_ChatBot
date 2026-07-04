"""
Report generation module for the BVRIT RAG Chatbot Evaluation Framework.

Generates comprehensive evaluation reports with summaries, per-dimension scores,
and recommendations.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from evaluation.utils import (
    save_final_report,
    load_results,
    load_ragas_scores,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# All 8 evaluation dimensions
ALL_DIMENSIONS = [
    "Functional",
    "Quality",
    "Safety",
    "Security",
    "Robustness",
    "Performance",
    "Context",
    "RAGAS",
]


class ReportGenerator:
    """
    Generates comprehensive evaluation reports with summaries, per-dimension scores,
    and recommendations.
    """

    def __init__(self) -> None:
        """Initialize the report generator."""
        pass

    def generate_report(
        self,
        test_results: List[Dict[str, Any]],
        ragas_metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.

        Args:
            test_results: List of individual test result dictionaries.
            ragas_metrics: Dictionary of RAGAS metric scores.
            metadata: Optional metadata about the evaluation run.

        Returns:
            Dictionary with the complete report.
        """
        total_tests = len(test_results)
        passed_tests = sum(1 for t in test_results if t.get("pass", False))
        failed_tests = total_tests - passed_tests
        warnings_count = self._count_warnings(test_results)
        pass_rate = round((passed_tests / total_tests * 100), 2) if total_tests > 0 else 0.0

        # Per-dimension scores
        dimension_scores = self._compute_dimension_scores(test_results)

        # Find weakest dimension
        weakest_dim = self._find_weakest_dimension(dimension_scores)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            dimension_scores, ragas_metrics, weakest_dim
        )

        # Build detailed results
        detailed_results = self._build_detailed_results(test_results)

        # Build report
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "warnings": warnings_count,
                "pass_rate": pass_rate,
            },
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "warnings": warnings_count,
                "pass_rate": pass_rate,
                "weakest_dimension": weakest_dim,
            },
            "dimension_scores": dimension_scores,
            "weakest_dimension": weakest_dim,
            "ragas_metrics": ragas_metrics,
            "recommendations": recommendations,
            "detailed_results": detailed_results,
        }

        # Add metadata if provided
        if metadata:
            report["metadata"] = metadata

        # Save report
        save_final_report(report)

        logger.info(
            f"Report generated: {total_tests} tests, "
            f"{passed_tests} passed, {failed_tests} failed, "
            f"pass rate {pass_rate}%"
        )

        return report

    def _count_warnings(self, test_results: List[Dict[str, Any]]) -> int:
        """
        Count tests with borderline scores (warning range).

        Args:
            test_results: List of test results.

        Returns:
            Number of warnings.
        """
        return sum(1 for t in test_results if 5 <= t.get("judge_score", 10) < 7)

    def _compute_dimension_scores(
        self,
        test_results: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute per-dimension statistics.

        Args:
            test_results: List of test results.

        Returns:
            Dictionary mapping dimension names to score stats.
        """
        dimensions: Dict[str, List[Dict[str, Any]]] = {}

        for t in test_results:
            dim = t.get("dimension", "Unknown")
            if dim not in dimensions:
                dimensions[dim] = []
            dimensions[dim].append(t)

        dimension_scores = {}
        for dim in ALL_DIMENSIONS:
            tests = dimensions.get(dim, [])
            if tests:
                scores = [t.get("judge_score", 0) for t in tests]
                avg_score = round(sum(scores) / len(scores), 2)
                passed = sum(1 for t in tests if t.get("pass", False))
                total = len(tests)
                pass_pct = round((passed / total * 100), 2) if total > 0 else 0.0

                dimension_scores[dim] = {
                    "avg_score": avg_score,
                    "pass_pct": pass_pct,
                    "tests": total,
                    "passed": passed,
                    "failed": total - passed,
                    "scores": scores,
                }
            else:
                dimension_scores[dim] = {
                    "avg_score": 0.0,
                    "pass_pct": 0.0,
                    "tests": 0,
                    "passed": 0,
                    "failed": 0,
                    "scores": [],
                }

        return dimension_scores

    def _find_weakest_dimension(
        self,
        dimension_scores: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Find the weakest dimension (lowest average score).

        Args:
            dimension_scores: Dictionary of per-dimension scores.

        Returns:
            Dictionary with dimension name, score, and details.
        """
        weakest_dim = "N/A"
        weakest_score = 10.0

        for dim, data in dimension_scores.items():
            if data["tests"] > 0 and data["avg_score"] < weakest_score:
                weakest_score = data["avg_score"]
                weakest_dim = dim

        return {
            "dimension": weakest_dim,
            "score": weakest_score,
        }

    def _generate_recommendations(
        self,
        dimension_scores: Dict[str, Dict[str, Any]],
        ragas_metrics: Dict[str, float],
        weakest_dim: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """
        Generate actionable recommendations based on evaluation results.

        Args:
            dimension_scores: Per-dimension score data.
            ragas_metrics: RAGAS metric scores.
            weakest_dim: Weakest dimension info.

        Returns:
            List of recommendation dictionaries.
        """
        recommendations = []

        # Weakest dimension recommendation
        wd_name = weakest_dim.get("dimension", "")
        wd_score = weakest_dim.get("score", 10.0)
        if wd_name != "N/A" and wd_score < 7:
            recommendations.append({
                "dimension": wd_name,
                "severity": "CRITICAL" if wd_score < 5 else "WARNING",
                "message": (
                    f"Lowest scoring dimension: '{wd_name}' "
                    f"(average score: {wd_score}/10). "
                    f"Review chunking strategy, retrieval quality, and "
                    f"prompt engineering for this dimension."
                ),
                "suggestion": (
                    f"Consider adjusting chunk_size, overlap, or top_k. "
                    f"Add more specific {wd_name.lower()}-related content to the grounding document."
                ),
            })

        # RAGAS metrics recommendations
        ragas_recs = self._generate_ragas_recommendations(ragas_metrics)
        recommendations.extend(ragas_recs)

        # Per-dimension recommendations
        for dim, data in dimension_scores.items():
            if data["tests"] > 0 and data["pass_pct"] < 70:
                recommendations.append({
                    "dimension": dim,
                    "severity": "WARNING",
                    "message": (
                        f"Dimension '{dim}' has {data['passed']}/{data['tests']} "
                        f"tests passing ({data['pass_pct']}%). "
                        f"Average score: {data['avg_score']}/10."
                    ),
                    "suggestion": self._get_dimension_suggestion(dim),
                })

        if not recommendations:
            recommendations.append({
                "dimension": "Overall",
                "severity": "INFO",
                "message": "All dimensions meet acceptable thresholds.",
                "suggestion": "Consider running more comprehensive tests for further optimization.",
            })

        return recommendations

    def _generate_ragas_recommendations(
        self,
        ragas_metrics: Dict[str, float],
    ) -> List[Dict[str, str]]:
        """
        Generate recommendations based on RAGAS metrics.

        Args:
            ragas_metrics: RAGAS metric scores.

        Returns:
            List of recommendation dictionaries.
        """
        recs = []

        ragas_thresholds = {
            "faithfulness": {
                "threshold": 0.7,
                "message": "Low faithfulness score. The model may be hallucinating.",
                "suggestion": "Strengthen the grounding prompt and enforce strict citation requirements.",
            },
            "answer_relevancy": {
                "threshold": 0.7,
                "message": "Low answer relevancy. Responses may not directly address queries.",
                "suggestion": "Improve retrieval quality or adjust top_k parameter.",
            },
            "context_precision": {
                "threshold": 0.7,
                "message": "Low context precision. Retrieved chunks may not be sufficiently relevant.",
                "suggestion": "Consider reducing chunk_size or improving the embedding model.",
            },
            "context_recall": {
                "threshold": 0.7,
                "message": "Low context recall. The retriever may miss relevant information.",
                "suggestion": "Increase top_k or improve embedding quality with better chunk overlap.",
            },
        }

        for metric, config in ragas_thresholds.items():
            score = ragas_metrics.get(metric, 1.0)
            if score < config["threshold"]:
                recs.append({
                    "dimension": f"RAGAS-{metric}",
                    "severity": "WARNING",
                    "message": f"{config['message']} (score: {score:.4f})",
                    "suggestion": config["suggestion"],
                })

        return recs

    def _get_dimension_suggestion(self, dimension: str) -> str:
        """
        Get improvement suggestions for a specific dimension.

        Args:
            dimension: The dimension name.

        Returns:
            Suggestion string.
        """
        suggestions = {
            "Functional": "Ensure the grounding document covers all functional questions comprehensively. Consider adding more specific Q&A examples.",
            "Quality": "Improve the system prompt to enforce better citation formatting and response structure. Add quality checks.",
            "Safety": "Add stronger safety disclaimers to the system prompt. Implement content filtering for harmful requests.",
            "Security": "Strengthen prompt injection defenses in the system prompt. Add input sanitization for sensitive queries.",
            "Robustness": "Add input validation and preprocessing. Handle edge cases like empty input, special characters, and off-topic questions explicitly.",
            "Performance": "Optimize retrieval and generation pipeline. Consider caching frequent queries and reducing chunk retrieval time.",
            "Context": "Improve conversation history management. Ensure context is properly maintained across multi-turn conversations.",
            "RAGAS": "Focus on retrieval quality. Improve chunking strategy and embedding model selection.",
        }
        return suggestions.get(dimension, "Review and improve the relevant system components.")

    def _build_detailed_results(
        self,
        test_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Build detailed result entries for the report.

        Args:
            test_results: List of test result dictionaries.

        Returns:
            List of detailed result dictionaries.
        """
        detailed = []
        for t in test_results:
            entry = {
                "id": t.get("id", "TC_000"),
                "dimension": t.get("dimension", "Unknown"),
                "question": t.get("question", ""),
                "expected_answer": t.get("expected_answer", ""),
                "actual_answer": t.get("actual_answer", "")[:1000],
                "pass": t.get("pass", False),
                "score": t.get("judge_score", 0),
                "reason": t.get("judge_reason", ""),
                "suggestions": t.get("judge_suggestions", ""),
                "latency": t.get("latency", 0.0),
                "retrieved_chunks_count": t.get("retrieved_chunk_count", 0),
                "retrieved_sections": t.get("retrieved_sections", []),
                "citations": t.get("citations", []),
                "is_refusal": t.get("is_refusal", False),
            }

            # Add failure analysis for failed tests
            if not t.get("pass", False):
                entry["failure_analysis"] = {
                    "root_cause": self._analyze_failure(t),
                    "recommended_fix": self._recommend_fix(t),
                }

            detailed.append(entry)

        return detailed

    def _analyze_failure(self, test_result: Dict[str, Any]) -> str:
        """
        Analyze the root cause of a test failure.

        Args:
            test_result: The failed test result.

        Returns:
            Root cause analysis string.
        """
        dimension = test_result.get("dimension", "Unknown")
        judge_reason = test_result.get("judge_reason", "")
        is_refusal = test_result.get("is_refusal", False)

        if is_refusal:
            return "Chatbot refused to answer, but an answer was expected."

        if "citation" in judge_reason.lower():
            return "Missing or incorrect citations in the response."

        if "hallucination" in judge_reason.lower() or "not in context" in judge_reason.lower():
            return "Chatbot hallucinated or used external knowledge not in the retrieved context."

        if not test_result.get("retrieved_chunk_count", 0):
            return "No relevant chunks were retrieved from the vector database."

        return (
            f"Response quality issue in {dimension} dimension. "
            f"Judge feedback: {judge_reason[:200]}"
        )

    def _recommend_fix(self, test_result: Dict[str, Any]) -> str:
        """
        Recommend a fix for a failed test.

        Args:
            test_result: The failed test result.

        Returns:
            Fix recommendation string.
        """
        dimension = test_result.get("dimension", "Unknown")
        is_refusal = test_result.get("is_refusal", False)

        if is_refusal:
            return "Ensure the grounding document contains the information for this query. Adjust chunk_size and overlap to capture relevant content."

        if not test_result.get("retrieved_chunk_count", 0):
            return "Increase top_k or improve embedding quality. Consider reducing chunk_size for more precise retrieval."

        if dimension == "Security":
            return "Ensure system prompt has clear security boundaries. Add more explicit security rules."

        if dimension == "Safety":
            return "Add stronger safety disclaimers to the system prompt. Ensure the chatbot consistently refuses harmful requests."

        if dimension == "Robustness":
            return "Add input validation and preprocessing for edge cases. Update system prompt to handle unusual inputs gracefully."

        return self._get_dimension_suggestion(dimension)


def generate_report_from_latest() -> Dict[str, Any]:
    """
    Generate a report from the latest evaluation results.

    Returns:
        Generated report dictionary.
    """
    results_data = load_results()
    test_results = results_data.get("test_results", [])
    metadata = results_data.get("metadata", {})
    ragas_scores = load_ragas_scores()

    generator = ReportGenerator()
    report = generator.generate_report(test_results, ragas_scores, metadata)
    return report


if __name__ == "__main__":
    report = generate_report_from_latest()
    print(f"\n{'=' * 60}")
    print("EVALUATION REPORT")
    print(f"{'=' * 60}")
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Passed: {report['summary']['passed']}")
    print(f"Failed: {report['summary']['failed']}")
    print(f"Warnings: {report['summary']['warnings']}")
    print(f"Pass Rate: {report['summary']['pass_rate']}%")
    print(f"\nWeakest Dimension: {report['weakest_dimension']['dimension']} ({report['weakest_dimension']['score']}/10)")
    print(f"\nRAGAS Metrics:")
    for metric, score in report['ragas_metrics'].items():
        print(f"  {metric}: {score:.4f}")
    print(f"\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  [{rec['severity']}] {rec['message']}")