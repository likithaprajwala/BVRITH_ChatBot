"""
Test execution module for the BVRIT RAG Chatbot Evaluation Framework.

Executes test cases against the chatbot and captures responses, chunks, latency, etc.
"""

import os
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from dotenv import load_dotenv

from chatbot import Chatbot
from evaluation.utils import save_results, load_testcases
from evaluation.llm_judge import LLMJudge
from evaluation.ragas_eval import RAGASEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


class TestExecutor:
    """
    Executes test cases against the RAG chatbot and captures detailed results.
    """

    def __init__(self, top_k: int = 5) -> None:
        """
        Initialize the test executor with a chatbot instance.

        Args:
            top_k: Number of documents to retrieve per query.
        """
        self.chatbot: Chatbot = Chatbot(top_k=top_k)
        self.judge: LLMJudge = LLMJudge()
        self.ragas_evaluator: RAGASEvaluator = RAGASEvaluator()
        self.top_k: int = top_k

    def execute_single_test(
        self,
        test_case: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a single test case against the chatbot.

        Args:
            test_case: Dictionary with 'question', 'expected_answer', 'dimension', 'pass_criteria'.
            conversation_history: Optional list of previous conversation turns for context tests.

        Returns:
            Dictionary with detailed test results.
        """
        question = test_case.get("question", "")
        dimension = test_case.get("dimension", "Functional")
        expected_answer = test_case.get("expected_answer", "")
        pass_criteria = test_case.get("pass_criteria", "")

        logger.info(f"Executing test [{dimension}]: '{question[:60]}...'")

        # Handle empty question
        if not question or not question.strip():
            return {
                "id": test_case.get("id", "TC_000"),
                "question": question,
                "expected_answer": expected_answer,
                "actual_answer": "I could not find that information in the official BVRIT knowledge base. Please contact the college using the Contact section.",
                "dimension": dimension,
                "pass_criteria": pass_criteria,
                "pass": True,
                "judge_score": 10,
                "judge_reason": "Empty input handled gracefully with refusal.",
                "judge_suggestions": "",
                "retrieved_chunk_count": 0,
                "retrieved_sections": [],
                "retrieved_chunks": [],
                "latency": 0.0,
                "citations": [],
                "is_refusal": True,
                "timestamp": datetime.now().isoformat(),
            }

        # Set up conversation history if provided
        if conversation_history:
            self.chatbot.clear_history()
            for turn in conversation_history:
                if "user" in turn and "assistant" in turn:
                    self.chatbot.conversation_history.append(
                        (turn["user"], turn["assistant"])
                    )

        start_time = time.time()

        # Query the chatbot
        try:
            result = self.chatbot.ask(question=question)
            actual_answer = result["answer"]
        except Exception as e:
            logger.error(f"Chatbot query failed: {e}")
            actual_answer = f"Error: {str(e)}"
            result = {
                "answer": actual_answer,
                "citations": [],
                "retrieved_chunk_count": 0,
                "retrieved_sections": [],
                "retrieved_chunks": [],
                "is_refusal": False,
            }

        total_latency = time.time() - start_time

        # Format retrieved chunks for judge
        retrieved_chunks_list = result.get("retrieved_chunks", [])
        retrieved_chunks_str = "\n\n".join([
            f"Chunk {i+1} [{c.get('section', 'Unknown')}]: {c.get('content', '')[:300]}"
            for i, c in enumerate(retrieved_chunks_list[:5])
        ])

        # LLM Judge evaluation
        judge_result = self.judge.evaluate(
            question=question,
            expected_answer=expected_answer,
            actual_answer=actual_answer,
            dimension=dimension,
            retrieved_chunks=retrieved_chunks_str,
        )

        test_result = {
            "id": test_case.get("id", "TC_000"),
            "question": question,
            "expected_answer": expected_answer,
            "actual_answer": actual_answer,
            "dimension": dimension,
            "pass_criteria": pass_criteria,
            "pass": judge_result.get("pass", False),
            "judge_score": judge_result.get("score", 0),
            "judge_reason": judge_result.get("reason", ""),
            "judge_suggestions": judge_result.get("suggestions", ""),
            "retrieved_chunk_count": result["retrieved_chunk_count"],
            "retrieved_sections": result.get("retrieved_sections", []),
            "retrieved_chunks": [
                {
                    "content": c.get("content", "")[:500],
                    "section": c.get("section", "Unknown"),
                    "score": c.get("score", 0.0),
                }
                for c in retrieved_chunks_list
            ],
            "latency": round(total_latency, 3),
            "citations": result.get("citations", []),
            "is_refusal": result.get("is_refusal", False),
            "timestamp": datetime.now().isoformat(),
        }

        # For RAGAS dimension, add ground_truth
        if dimension == "RAGAS" and "ground_truth" in test_case:
            test_result["ground_truth"] = test_case["ground_truth"]

        status = "PASS" if test_result["pass"] else "FAIL"
        logger.info(
            f"Test {test_result['id']}: {status} "
            f"(score: {test_result['judge_score']}/10, "
            f"latency: {test_result['latency']}s, "
            f"chunks: {test_result['retrieved_chunk_count']})"
        )

        return test_result

    def execute_all_tests(
        self,
        test_cases: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute all test cases.

        Args:
            test_cases: List of test case dictionaries. Loads from file if None.

        Returns:
            List of test result dictionaries.
        """
        if test_cases is None:
            test_cases = load_testcases()

        if not test_cases:
            logger.error("No test cases to execute")
            return []

        logger.info(f"Executing {len(test_cases)} test cases...")
        results = []

        for i, tc in enumerate(test_cases):
            logger.info(f"[{i+1}/{len(test_cases)}] Running test {tc.get('id', 'TC_???')}...")

            # Get conversation history for context tests
            conversation_history = tc.get("conversation_history", None)

            result = self.execute_single_test(tc, conversation_history)
            results.append(result)

            # Clear chatbot history between tests (unless it's a context test)
            if not conversation_history:
                self.chatbot.clear_history()

        # Compute RAGAS metrics
        ragas_scores = self.ragas_evaluator.evaluate(results)

        # Save results
        output = {
            "test_results": results,
            "ragas_scores": ragas_scores,
            "metadata": {
                "executed_at": datetime.now().isoformat(),
                "total_tests": len(results),
                "passed": sum(1 for r in results if r.get("pass", False)),
                "failed": sum(1 for r in results if not r.get("pass", False)),
                "top_k": self.top_k,
            }
        }

        save_results(output)

        # Save RAGAS scores separately
        from evaluation.utils import save_ragas_scores
        save_ragas_scores(ragas_scores)

        return results

    def run_with_progress(
        self,
        test_cases: Optional[List[Dict[str, Any]]] = None,
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        """
        Execute all tests with a progress callback for UI updates.

        Args:
            test_cases: List of test case dictionaries.
            progress_callback: Function called with (current, total, status) after each test.

        Returns:
            List of test result dictionaries.
        """
        if test_cases is None:
            test_cases = load_testcases()

        if not test_cases:
            logger.error("No test cases to execute")
            return []

        logger.info(f"Executing {len(test_cases)} test cases with progress tracking...")
        results = []

        for i, tc in enumerate(test_cases):
            dim = tc.get("dimension", "Unknown")
            q = tc.get("question", "")[:50]
            if progress_callback:
                progress_callback(i, len(test_cases), f"Testing [{dim}]: {q}...")

            conversation_history = tc.get("conversation_history", None)
            result = self.execute_single_test(tc, conversation_history)
            results.append(result)

            if not conversation_history:
                self.chatbot.clear_history()

        # Compute RAGAS metrics
        ragas_scores = self.ragas_evaluator.evaluate(results)

        output = {
            "test_results": results,
            "ragas_scores": ragas_scores,
            "metadata": {
                "executed_at": datetime.now().isoformat(),
                "total_tests": len(results),
                "passed": sum(1 for r in results if r.get("pass", False)),
                "failed": sum(1 for r in results if not r.get("pass", False)),
                "top_k": self.top_k,
            }
        }

        save_results(output)
        from evaluation.utils import save_ragas_scores
        save_ragas_scores(ragas_scores)

        if progress_callback:
            progress_callback(len(test_cases), len(test_cases), "Complete!")

        return results


if __name__ == "__main__":
    executor = TestExecutor()
    results = executor.execute_all_tests()

    passed = sum(1 for r in results if r.get("pass", False))
    failed = len(results) - passed
    print(f"\n{'=' * 60}")
    print(f"TEST EXECUTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    print(f"Pass Rate: {passed / len(results) * 100:.1f}%" if results else "N/A")