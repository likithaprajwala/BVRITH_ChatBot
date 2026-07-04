"""
RAGAS evaluation module for the BVRIT RAG Chatbot Evaluation Framework.

Computes Faithfulness, Answer Relevancy, Context Precision, and Context Recall.
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

from build_index import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")


RAGAS_EVALUATION_PROMPT = """You are an expert RAGAS (Retrieval-Augmented Generation Assessment) evaluator.

Evaluate the following RAG system response on four metrics. Be strict and objective.

## Question
{question}

## Answer
{answer}

## Retrieved Context
{context}

## Ground Truth (if available)
{ground_truth}

## Metrics to Evaluate

### 1. Faithfulness (0.0 - 1.0)
Does the answer stay faithful to the retrieved context? Score based on:
- 1.0: All claims in the answer are directly supported by the context.
- 0.7: Most claims are supported, minor unsupported claims.
- 0.4: Some claims are supported, some are not.
- 0.0: Most or all claims are not supported by the context.

### 2. Answer Relevancy (0.0 - 1.0)
Is the answer relevant to the question? Score based on:
- 1.0: Directly answers the question completely.
- 0.7: Answers the question but with some irrelevant information.
- 0.4: Partially answers the question.
- 0.0: Does not answer the question at all.

### 3. Context Precision (0.0 - 1.0)
Are the retrieved chunks precise and relevant? Score based on:
- 1.0: All retrieved chunks are highly relevant to the question.
- 0.7: Most chunks are relevant, some are not.
- 0.4: Some chunks are relevant.
- 0.0: Retrieved chunks are not relevant.

### 4. Context Recall (0.0 - 1.0)
Did the retrieval find all relevant information? Score based on:
- 1.0: All necessary information was retrieved.
- 0.7: Most necessary information was retrieved.
- 0.4: Some necessary information was retrieved.
- 0.0: Necessary information was not retrieved.

Return ONLY a valid JSON object:
{{
    "faithfulness": 0.0-1.0,
    "answer_relevancy": 0.0-1.0,
    "context_precision": 0.0-1.0,
    "context_recall": 0.0-1.0,
    "faithfulness_reason": "Brief explanation",
    "answer_relevancy_reason": "Brief explanation",
    "context_precision_reason": "Brief explanation",
    "context_recall_reason": "Brief explanation"
}}
"""


class RAGASEvaluator:
    """
    Evaluates RAGAS metrics for RAG system responses.
    Computes Faithfulness, Answer Relevancy, Context Precision, Context Recall.
    """

    def __init__(self) -> None:
        """Initialize the RAGAS evaluator with OpenRouter client."""
        self.llm_client: OpenAI = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
        self.llm_model: str = LLM_MODEL

    def evaluate_single(
        self,
        question: str,
        answer: str,
        context: str,
        ground_truth: str = "",
    ) -> Dict[str, float]:
        """
        Evaluate RAGAS metrics for a single Q&A pair.

        Args:
            question: The user's question.
            answer: The system's answer.
            context: The retrieved context chunks.
            ground_truth: Optional ground truth answer.

        Returns:
            Dictionary with RAGAS metric scores.
        """
        prompt = RAGAS_EVALUATION_PROMPT.format(
            question=question,
            answer=answer,
            context=context[:3000],
            ground_truth=ground_truth if ground_truth else "Not available.",
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert RAGAS evaluator. Return ONLY valid JSON with the specified metrics. No markdown, no code fences, no explanation outside the JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            result_text = response.choices[0].message.content.strip()

            # Parse JSON
            result = self._parse_json(result_text)

            # Ensure all required fields are present with defaults
            defaults = {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
            }
            for key, default_val in defaults.items():
                if key not in result:
                    result[key] = default_val
                else:
                    # Clamp to 0-1
                    result[key] = max(0.0, min(1.0, float(result[key])))

            return result

        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
            }

    def evaluate(
        self,
        test_results: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Compute RAGAS metrics across all test results.

        Args:
            test_results: List of test result dictionaries with question, actual_answer, and retrieved_chunks.

        Returns:
            Dictionary of average RAGAS metric scores.
        """
        if not test_results:
            logger.warning("No test results for RAGAS evaluation")
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
            }

        logger.info(f"Computing RAGAS metrics for {len(test_results)} test cases...")

        # Use RAGAS-specific test cases if available
        ragas_cases = [tc for tc in test_results if tc.get("dimension") == "RAGAS"]
        if not ragas_cases:
            ragas_cases = test_results[:5]  # Use first 5 as fallback

        all_metrics = {
            "faithfulness": [],
            "answer_relevancy": [],
            "context_precision": [],
            "context_recall": [],
        }

        for tc in ragas_cases:
            question = tc.get("question", "")
            answer = tc.get("actual_answer", "")
            ground_truth = tc.get("ground_truth", tc.get("expected_answer", ""))

            # Build context string from retrieved chunks
            chunks = tc.get("retrieved_chunks", [])
            context_parts = []
            for i, chunk in enumerate(chunks[:5]):
                section = chunk.get("section", "Unknown")
                content = chunk.get("content", "")[:500]
                context_parts.append(f"[Chunk {i+1}] Section: {section}\n{content}")
            context = "\n\n".join(context_parts)

            if not context:
                context = "No retrieved chunks available."

            # Skip empty answers
            if not answer or answer == "Error:":
                continue

            metrics = self.evaluate_single(question, answer, context, ground_truth)

            for key in all_metrics:
                all_metrics[key].append(metrics.get(key, 0.0))

        # Average the metrics
        averaged_metrics = {}
        for key, values in all_metrics.items():
            if values:
                averaged_metrics[key] = round(sum(values) / len(values), 4)
            else:
                averaged_metrics[key] = 0.0

        logger.info(f"RAGAS metrics: {averaged_metrics}")
        return averaged_metrics

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response.

        Args:
            text: Text potentially containing JSON.

        Returns:
            Parsed JSON dictionary.
        """
        # Try to extract JSON from code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        # Try to find JSON object directly
        obj_match = re.search(r'(\{.*\})', text, re.DOTALL)
        if obj_match:
            text = obj_match.group(1)

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fix common issues
            text = text.replace("'", '"')
            text = re.sub(r',\s*}', '}', text)
            text = re.sub(r',\s*]', ']', text)
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse RAGAS JSON: {text[:200]}")
                return {
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                    "context_precision": 0.0,
                    "context_recall": 0.0,
                }


if __name__ == "__main__":
    # Quick test
    evaluator = RAGASEvaluator()
    test_results = [
        {
            "question": "What is BVRIT College?",
            "actual_answer": "BVRIT Hyderabad College of Engineering for Women was established in 2012 [About].",
            "expected_answer": "BVRIT College is for women, established in 2012.",
            "dimension": "RAGAS",
            "retrieved_chunks": [
                {"section": "About", "content": "BVRIT Hyderabad College of Engineering for Women was established in the year 2012."}
            ],
        }
    ]
    metrics = evaluator.evaluate(test_results)
    print(f"RAGAS Metrics: {json.dumps(metrics, indent=2)}")