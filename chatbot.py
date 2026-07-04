"""
Conversational chatbot module for BVRIT RAG Chatbot.

Manages conversation history and provides an interactive chat interface.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

from rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
        logger.info(f"User question: '{question[:50]}...'")

        # Get response from RAG pipeline
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