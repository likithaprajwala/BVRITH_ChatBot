"""
Conversational chatbot module for BVRIT RAG Chatbot.

Manages conversation history and provides an interactive chat interface.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any

from rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── Greeting patterns ──────────────────────────────────────────────────────────
_GREETING_PATTERNS = re.compile(
    r"^\s*("
    r"hi|hello|hey|hiya|howdy|"
    r"good\s*morning|good\s*afternoon|good\s*evening|good\s*night|"
    r"greetings|sup|what'?s\s*up|yo|"
    r"namaste|namaskar|hai|helo|hii|hiii|"
    r"morning|afternoon|evening"
    r")\s*[!?.]*\s*$",
    re.IGNORECASE,
)

_GREETING_RESPONSE = (
    "Hello! 👋 Welcome to the **BVRIT College FAQ Assistant**.\n\n"
    "I can help you with information about BVRIT Hyderabad College of Engineering for Women — "
    "departments, admissions, fee structure, placements, facilities, faculty, and more.\n\n"
    "How can I help you today? 😊"
)


class Chatbot:
    """
    Conversational chatbot that uses the RAG pipeline for BVRIT College Q&A.
    """

    def __init__(self, top_k: int = 5) -> None:
        """
        Initialize the chatbot with a RAG pipeline.

        Args:
            top_k: Number of documents to retrieve per query.
        """
        self.rag_pipeline: RAGPipeline = RAGPipeline(top_k=top_k)
        self.conversation_history: List[Tuple[str, str]] = []
        logger.info(f"Chatbot initialized with top_k={top_k}")

    def _is_greeting(self, text: str) -> bool:
        """Return True if the message is purely a greeting."""
        return bool(_GREETING_PATTERNS.match(text.strip()))

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")

    def get_history(self) -> List[Tuple[str, str]]:
        """
        Get the conversation history.

        Returns:
            List of (user_message, assistant_response) tuples.
        """
        return self.conversation_history.copy()

    def ask(
        self,
        question: str,
        section: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Ask a question and get an answer with citations.

        Args:
            question: The user's question.
            section: Optional section filter.
            top_k: Number of documents to retrieve (overrides default).

        Returns:
            Dictionary with answer, citations, and metadata.
        """
        import time
        logger.info(f"User question: '{question[:50]}...'")

        # ── Short-circuit: handle greetings without RAG ──────────────────────
        if self._is_greeting(question):
            logger.info("Greeting detected — returning canned response")
            self.conversation_history.append((question, _GREETING_RESPONSE))
            return {
                "answer": _GREETING_RESPONSE,
                "citations": [],
                "retrieved_chunk_count": 0,
                "latency": 0.0,
                "retrieved_sections": [],
                "retrieved_chunks": [],
                "is_refusal": False,
            }

        # ── Normal RAG path ──────────────────────────────────────────────────
        result = self.rag_pipeline.query(
            question=question,
            section=section,
            chat_history=self.conversation_history,
            top_k=top_k,
        )

        # Add to conversation history
        self.conversation_history.append((question, result["answer"]))

        # Keep only the last 20 turns to avoid context overflow
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        return result

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics about the chatbot session.

        Returns:
            Dictionary with session statistics.
        """
        total_questions = len(self.conversation_history)
        if total_questions == 0:
            return {
                "total_questions": 0,
                "average_latency": 0,
                "total_chunks_retrieved": 0,
            }

        return {
            "total_questions": total_questions,
        }
