"""
RAG Pipeline for BVRIT College Chatbot.

Handles retrieval, prompt formatting, and LLM interaction.
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from openai import OpenAI

from build_index import (
    create_embeddings,
    get_vector_store,
    CHROMA_DB_PATH,
    OPENROUTER_API_KEY,
    EMBEDDING_MODEL,
    TOP_K,
    OPENROUTER_BASE_URL,
    AVAILABLE_SECTIONS,
)
from prompts import format_system_prompt, SYSTEM_PROMPT, JUDGE_SYSTEM_PROMPT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

# LLM Configuration
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for BVRIT College Q&A.
    """

    def __init__(self, top_k: int = TOP_K) -> None:
        """
        Initialize the RAG pipeline with vector store and LLM client.

        Args:
            top_k: Number of documents to retrieve.
        """
        self.top_k: int = top_k
        self.vector_store: Chroma = get_vector_store()
        self.llm_client: OpenAI = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
        self.llm_model: str = LLM_MODEL
        logger.info(f"RAGPipeline initialized with top_k={top_k}, model={LLM_MODEL}")

    def retrieve(
        self,
        query: str,
        section: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[List[Any], List[Dict[str, Any]], List[float]]:
        """
        Retrieve relevant documents from the vector store.

        Args:
            query: The user's question.
            section: Optional section filter.
            top_k: Number of documents to retrieve (overrides default).

        Returns:
            Tuple of (retrieved_documents, metadata_list, score_list).
        """
        k = top_k if top_k is not None else self.top_k

        logger.info(f"Retrieving for query: '{query[:50]}...' with top_k={k}")

        if section and section in AVAILABLE_SECTIONS:
            logger.info(f"Filtering by section: {section}")
            # Use ChromaDB's metadata filtering
            results = self.vector_store.similarity_search_with_score(
                query,
                k=k * 2,  # Fetch more to filter
                filter={"section": section},
            )
        else:
            results = self.vector_store.similarity_search_with_score(
                query,
                k=k,
            )

        # If no results with filter, fall back to unfiltered
        if not results and section:
            logger.warning(f"No results with section '{section}', falling back to unfiltered")
            results = self.vector_store.similarity_search_with_score(
                query,
                k=k,
            )

        # Limit to top_k
        results = results[:k]

        documents = [doc for doc, _ in results]
        metadata_list = [doc.metadata for doc, _ in results]
        scores = [float(score) for _, score in results]

        logger.info(f"Retrieved {len(documents)} documents")
        return documents, metadata_list, scores

    def format_context(self, documents: List[Any]) -> str:
        """
        Format retrieved documents into a context string for the prompt.

        Args:
            documents: List of retrieved Document objects.

        Returns:
            Formatted context string with citations.
        """
        context_parts = []
        for i, doc in enumerate(documents):
            section = doc.metadata.get("section", "Unknown")
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(
                f"[Document {i + 1}] (Source: {source}, Section: {section})\n"
                f"{doc.page_content}\n"
            )
        return "\n---\n".join(context_parts)

    def format_chat_history(self, history: List[Tuple[str, str]]) -> str:
        """
        Format conversation history for the prompt.

        Args:
            history: List of (user_message, assistant_response) tuples.

        Returns:
            Formatted chat history string.
        """
        if not history:
            return "No previous conversation."

        formatted = []
        for user_msg, assistant_msg in history[-5:]:  # Last 5 turns
            formatted.append(f"User: {user_msg}")
            formatted.append(f"Assistant: {assistant_msg}")
        return "\n".join(formatted)

    def generate(
        self,
        question: str,
        documents: List[Any],
        chat_history: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        """
        Generate an answer using the LLM based on retrieved context.

        Args:
            question: The user's question.
            documents: Retrieved document chunks.
            chat_history: Previous conversation history.

        Returns:
            Generated answer string.
        """
        context = self.format_context(documents)
        history_str = self.format_chat_history(chat_history or [])

        prompt = format_system_prompt(context, history_str, question)

        logger.info(f"Generating answer for question: '{question[:50]}...'")

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated answer of length {len(answer)}")
            return answer
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"I encountered an error while generating a response. Please try again later. (Error: {str(e)})"

    def query(
        self,
        question: str,
        section: Optional[str] = None,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Complete RAG query: retrieve + generate.

        Args:
            question: The user's question.
            section: Optional section filter.
            chat_history: Previous conversation history.
            top_k: Number of documents to retrieve.

        Returns:
            Dictionary with answer, citations, metadata, and performance metrics.
        """
        start_time = time.time()

        # Retrieve
        documents, metadata_list, scores = self.retrieve(question, section, top_k)

        # Generate
        answer = self.generate(question, documents, chat_history)

        # Extract citations from answer
        citations = self.extract_citations(answer)
        if not citations and documents:
            citations = [doc.metadata.get("section", "Unknown") for doc in documents]

        latency = time.time() - start_time

        result = {
            "answer": answer,
            "citations": list(set(citations)),
            "retrieved_chunk_count": len(documents),
            "latency": round(latency, 3),
            "retrieved_sections": list(set(
                doc.metadata.get("section", "Unknown") for doc in documents
            )),
            "retrieved_chunks": [
                {
                    "content": doc.page_content[:200] + "...",
                    "section": doc.metadata.get("section", "Unknown"),
                    "score": score,
                }
                for doc, score in zip(documents, scores)
            ],
        }

        # Check if it's a refusal
        if "could not find that information" in answer.lower():
            result["is_refusal"] = True
        else:
            result["is_refusal"] = False

        logger.info(f"Query completed in {latency:.2f}s with {len(documents)} chunks")
        return result

    @staticmethod
    def extract_citations(answer: str) -> List[str]:
        """
        Extract citation section names from an answer.

        Args:
            answer: The generated answer text.

        Returns:
            List of cited section names.
        """
        import re
        citations = re.findall(r'\*\*\[(.*?)\]\*\*', answer)
        return citations


class LLMJudge:
    """
    LLM-based evaluator for RAG responses.
    """

    def __init__(self) -> None:
        """Initialize the LLM Judge with OpenRouter client."""
        self.llm_client: OpenAI = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
        self.llm_model: str = LLM_MODEL

    def evaluate(
        self,
        question: str,
        expected_answer: str,
        actual_answer: str,
        dimension: str,
        retrieved_chunks: str,
    ) -> Dict[str, Any]:
        """
        Evaluate a test case using the LLM Judge.

        Args:
            question: The test question.
            expected_answer: The expected answer.
            actual_answer: The actual answer from the RAG system.
            dimension: The evaluation dimension.
            retrieved_chunks: The retrieved context chunks.

        Returns:
            Dictionary with pass, reason, and score.
        """
        from prompts import format_judge_prompt

        prompt = format_judge_prompt(
            question, expected_answer, actual_answer, dimension, retrieved_chunks
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            result_text = response.choices[0].message.content.strip()

            # Parse JSON from response
            result = self._parse_json(result_text)
            return result
        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            return {
                "pass": False,
                "reason": f"Judge evaluation error: {str(e)}",
                "score": 0,
            }

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response, handling potential markdown code blocks.

        Args:
            text: Text potentially containing JSON.

        Returns:
            Parsed JSON dictionary.
        """
        import re

        # Try to find JSON in code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        # Clean up
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to fix common issues
            text = text.replace("'", '"')
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {
                    "pass": False,
                    "reason": f"Failed to parse judge response: {text[:200]}",
                    "score": 0,
                }