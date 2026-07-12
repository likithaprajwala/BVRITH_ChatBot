"""
Utility functions for the BVRIT RAG Chatbot Evaluation Framework.

Handles loading, saving, logging, and common helper functions.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Path Constants ---
SCRIPT_DIR = Path(__file__).parent.parent.absolute()
TESTCASES_DIR = SCRIPT_DIR / "testcases"
RESULTS_DIR = SCRIPT_DIR / "results"
REPORTS_DIR = SCRIPT_DIR / "reports"

# Ensure directories exist
for d in [TESTCASES_DIR, RESULTS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)


def load_testcases(filepath: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load test cases from a JSON file.

    Args:
        filepath: Path to the test cases JSON file. Uses default if None.

    Returns:
        List of test case dictionaries.
    """
    if filepath is None:
        filepath = str(TESTCASES_DIR / "generated_testcases.json")

    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} test cases from {filepath}")
            return data
        except Exception as e:
            logger.error(f"Failed to load test cases: {e}")
            return []
    else:
        logger.warning(f"Test cases file not found: {filepath}")
        return []


def save_testcases(test_cases: List[Dict[str, Any]], filepath: Optional[str] = None) -> str:
    """
    Save test cases to a JSON file.

    Args:
        test_cases: List of test case dictionaries.
        filepath: Path to save to. Uses default if None.

    Returns:
        Path to the saved file.
    """
    if filepath is None:
        filepath = str(TESTCASES_DIR / "generated_testcases.json")

    # Add IDs if missing
    for i, tc in enumerate(test_cases):
        if "id" not in tc:
            tc["id"] = f"TC_{i+1:03d}"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(test_cases)} test cases to {filepath}")
    return filepath


def _is_valid_results(data: Dict[str, Any]) -> bool:
    """
    Check whether a results file contains real answers (not all API errors).

    Returns True if at least half the test results have non-error actual answers.
    """
    test_results = data.get("test_results", [])
    if not test_results:
        return False
    error_count = sum(
        1 for t in test_results
        if str(t.get("actual_answer", "")).startswith("Error:")
        or "401" in str(t.get("actual_answer", ""))
        or "402" in str(t.get("actual_answer", ""))
    )
    # Treat as invalid if more than half the answers are errors
    return error_count < len(test_results) / 2


def load_results(filepath: Optional[str] = None) -> Dict[str, Any]:
    """
    Load evaluation results from a JSON file.
    Skips files where the majority of answers are API errors.

    Args:
        filepath: Path to the results JSON file.

    Returns:
        Results dictionary.
    """
    if filepath is not None:
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load results: {e}")
                return {}
        return {}

    # Walk files newest-first, return first valid one
    files = sorted(RESULTS_DIR.glob("evaluation_results_*.json"), reverse=True)
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if _is_valid_results(data):
                logger.info(f"Loaded valid results from {f}")
                return data
            else:
                logger.warning(f"Skipping broken results file (API errors): {f.name}")
        except Exception as e:
            logger.error(f"Failed to load {f}: {e}")
    return {}


def save_results(results: Dict[str, Any], filepath: Optional[str] = None) -> str:
    """
    Save evaluation results to a JSON file.

    Args:
        results: Results dictionary to save.
        filepath: Path to save to.

    Returns:
        Path to the saved file.
    """
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = str(RESULTS_DIR / f"evaluation_results_{timestamp}.json")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved results to {filepath}")
    return filepath


def load_ragas_scores(filepath: Optional[str] = None) -> Dict[str, float]:
    """
    Load RAGAS scores from a JSON file.
    Skips files that contain only zeros (produced by broken API-error runs).

    Args:
        filepath: Path to the RAGAS scores JSON file.

    Returns:
        Dictionary of RAGAS metric scores.
    """
    default = {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
    }

    if filepath is not None:
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load RAGAS scores: {e}")
        return default

    # Walk newest-first, return first non-zero file
    files = sorted(RESULTS_DIR.glob("ragas_scores_*.json"), reverse=True)
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                scores = json.load(fh)
            # Skip if all scores are 0 (broken run)
            if any(v > 0 for v in scores.values()):
                logger.info(f"Loaded valid RAGAS scores from {f}")
                return scores
            else:
                logger.warning(f"Skipping zero RAGAS scores file: {f.name}")
        except Exception as e:
            logger.error(f"Failed to load {f}: {e}")
    return default


def save_ragas_scores(scores: Dict[str, float], filepath: Optional[str] = None) -> str:
    """
    Save RAGAS scores to a JSON file.

    Args:
        scores: Dictionary of RAGAS metric scores.
        filepath: Path to save to.

    Returns:
        Path to the saved file.
    """
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = str(RESULTS_DIR / f"ragas_scores_{timestamp}.json")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved RAGAS scores to {filepath}")
    return filepath


def load_final_report(filepath: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the final evaluation report.
    Skips reports where pass_rate is suspiciously low due to API errors (all zeros).

    Args:
        filepath: Path to the report JSON file.

    Returns:
        Report dictionary.
    """
    if filepath is not None:
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load report: {e}")
        return {}

    # Walk newest-first, skip reports caused by API errors
    files = sorted(RESULTS_DIR.glob("final_report_*.json"), reverse=True)
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                report = json.load(fh)
            # Check if this report corresponds to a broken run:
            # All dimension scores being 0 with pass_rate near 0 means API errors
            dim_scores = report.get("dimension_scores", {})
            all_zero = all(
                v.get("avg_score", 0) == 0
                for v in dim_scores.values()
            ) if dim_scores else True
            if all_zero and report.get("report_metadata", {}).get("pass_rate", 0) < 10:
                logger.warning(f"Skipping broken report (all-zero scores): {f.name}")
                continue
            logger.info(f"Loaded valid report from {f}")
            return report
        except Exception as e:
            logger.error(f"Failed to load {f}: {e}")
    return {}


def save_final_report(report: Dict[str, Any], filepath: Optional[str] = None) -> str:
    """
    Save the final evaluation report.

    Args:
        report: Report dictionary to save.
        filepath: Path to save to.

    Returns:
        Path to the saved file.
    """
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = str(RESULTS_DIR / f"final_report_{timestamp}.json")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved final report to {filepath}")
    return filepath


def get_latest_results() -> Dict[str, Any]:
    """
    Get the latest evaluation results from saved files.

    Returns:
        Combined results dictionary with test_results, ragas_scores, and report.
    """
    results = load_results()
    ragas = load_ragas_scores()
    report = load_final_report()

    return {
        "test_results": results.get("test_results", []),
        "ragas_scores": ragas,
        "report": report,
    }


def format_latency(seconds: float) -> str:
    """
    Format latency in a human-readable format.

    Args:
        seconds: Latency in seconds.

    Returns:
        Formatted latency string.
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.0f}s"