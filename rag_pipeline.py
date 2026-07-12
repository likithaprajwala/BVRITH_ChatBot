"""
RAG Pipeline for BVRIT College Chatbot.

Handles query intent detection, section-aware retrieval, chunk merging,
prompt formatting, and LLM interaction.
"""

import os
import re
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

# ── Intent Detection ──────────────────────────────────────────────────────────
# Maps detected intent → AVAILABLE_SECTIONS value
_INTENT_MAP: Dict[str, str] = {
    "about":        "About",
    "departments":  "Departments",
    "admissions":   "Admissions",
    "fee":          "Fee Structure",
    "placements":   "Placements",
    "facilities":   "Facilities",
    "faculty":      "Faculty",
    "contact":      "Contact",
}

# Keyword patterns per intent (checked against lower-cased query)
_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "about": [
        "about bvrit", "about the college", "overview", "history",
        "established", "founded", "vision", "mission", "introduction",
        "what is bvrit", "tell me about bvrit",
    ],
    "departments": [
        "department", "departments", "branch", "branches", "program",
        "programmes", "courses offered", "b.tech", "btech", "m.tech",
        "mtech", "cse", "ece", "eee", "it ", " it,", "civil", "mba",
        "what do you offer", "what programs",
    ],
    "admissions": [
        "admission", "admissions", "how to apply", "apply", "eligibility",
        "entrance", "eamcet", "tseamcet", "jee", "seat", "intake",
        "application process", "admission process",
    ],
    "fee": [
        "fee", "fees", "fee structure", "tuition", "cost", "how much",
        "payment", "charges", "annual fee", "semester fee",
    ],
    "placements": [
        "placement", "placements", "job", "recruit", "company", "companies",
        "package", "salary", "lpa", "ctc", "campus", "placed", "hiring",
        "career", "offer", "highest package",
    ],
    "facilities": [
        "facilit", "library", "lab", "hostel", "cafeteria", "canteen",
        "transport", "sports", "gym", "wifi", "internet", "infrastructure",
        "yoga", "wellness", "amenities",
    ],
    "faculty": [
        "faculty", "professor", "teacher", "staff", "lecturer",
        "phd", "principal", "hod", "head of department",
    ],
    "contact": [
        "contact", "address", "phone", "email", "website", "location",
        "how to reach", "where is bvrit", "map",
    ],
}

# Queries that are clearly broad (need all chunks from a section)
_BROAD_QUERY_PATTERNS = re.compile(
    r"\b(all|list|every|complete|full|details?|tell me about|what are|"
    r"what is the|explain|describe|overview|summary)\b",
    re.IGNORECASE,
)


def detect_intent(query: str) -> Optional[str]:
    """
    Detect which BVRIT section the query is about.

    Returns the AVAILABLE_SECTIONS value (e.g. 'Departments') or None.
    """
    q = query.lower()
    for intent, keywords in _INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                return _INTENT_MAP[intent]
    return None


def is_broad_query(query: str) -> bool:
    """Return True if the query is broad enough to warrant fetching more chunks."""
    return bool(_BROAD_QUERY_PATTERNS.search(query)) or len(query.split()) <= 6


# ── RAG Pipeline ─────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for BVRIT College Q&A.
    Includes query intent detection, section-aware retrieval, and chunk merging.
    """

    def __init__(self, top_k: int = TOP_K) -> None:
        self.top_k: int = top_k
        self.vector_store: Chroma = get_vector_store()
        self.llm_client: OpenAI = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
        self.llm_model: str = LLM_MODEL
        logger.info(f"RAGPipeline initialized with top_k={top_k}, model={LLM_MODEL}")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        section: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[List[Any], List[Dict[str, Any]], List[float]]:
        """
        Smart retrieval with intent detection and section-aware fetching.

        1. Detect query intent → infer section if not provided by user.
        2. If broad query about a section, fetch all chunks from that section.
        3. Merge duplicate/near-duplicate content from the same section.
        4. Fall back to standard semantic search if no section detected.
        """
        k = top_k if top_k is not None else self.top_k

        # Step 1 — Resolve section: explicit filter > intent detection
        detected_section = section if (section and section in AVAILABLE_SECTIONS) else detect_intent(query)
        broad = is_broad_query(query)

        logger.info(
            f"Retrieving: section={detected_section}, broad={broad}, top_k={k}"
        )

        results: List[Tuple[Any, float]] = []

        if detected_section:
            # Increase k for broad questions about a specific section
            fetch_k = max(k * 2, 8) if broad else k + 2
            try:
                results = self.vector_store.similarity_search_with_score(
                    query,
                    k=fetch_k,
                    filter={"section": detected_section},
                )
                logger.info(
                    f"Section-filtered retrieval [{detected_section}]: {len(results)} chunks"
                )
            except Exception as e:
                logger.warning(f"Section filter failed ({e}), falling back to unfiltered")
                results = []

            # If section filter returns too few results, supplement with unfiltered
            if len(results) < 3:
                extra = self.vector_store.similarity_search_with_score(query, k=k)
                # Merge: avoid duplicates by page_content
                seen = {doc.page_content for doc, _ in results}
                for doc, score in extra:
                    if doc.page_content not in seen:
                        results.append((doc, score))
                        seen.add(doc.page_content)
        else:
            # No section detected → standard semantic search
            results = self.vector_store.similarity_search_with_score(query, k=k)

        # Deduplicate by content prefix (first 100 chars)
        seen_prefixes: set = set()
        deduped: List[Tuple[Any, float]] = []
        for doc, score in results:
            prefix = doc.page_content[:100].strip()
            if prefix not in seen_prefixes:
                deduped.append((doc, score))
                seen_prefixes.add(prefix)

        # Cap at reasonable limit to avoid token overflow
        max_chunks = min(len(deduped), 8 if broad else k)
        deduped = deduped[:max_chunks]

        documents = [doc for doc, _ in deduped]
        metadata_list = [doc.metadata for doc, _ in deduped]
        scores = [float(s) for _, s in deduped]

        logger.info(f"Final retrieved chunks: {len(documents)}")
        return documents, metadata_list, scores

    # ── Context Formatting ────────────────────────────────────────────────────

    def format_context(self, documents: List[Any]) -> str:
        """
        Format retrieved docs into context string.
        Groups chunks by section and merges them for clarity.
        """
        # Group by section
        sections: Dict[str, List[str]] = {}
        for doc in documents:
            sec = doc.metadata.get("section", "General")
            sections.setdefault(sec, []).append(doc.page_content.strip())

        parts = []
        for sec, contents in sections.items():
            merged = "\n\n".join(contents)
            parts.append(f"[Section: {sec}]\n{merged}")

        return "\n\n---\n\n".join(parts)

    def format_chat_history(self, history: List[Tuple[str, str]]) -> str:
        """Format last 5 conversation turns."""
        if not history:
            return "No previous conversation."
        lines = []
        for user_msg, assistant_msg in history[-5:]:
            lines.append(f"User: {user_msg}")
            lines.append(f"Assistant: {assistant_msg}")
        return "\n".join(lines)

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(
        self,
        question: str,
        documents: List[Any],
        chat_history: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        """Generate answer from LLM using retrieved context."""
        context = self.format_context(documents)
        history_str = self.format_chat_history(chat_history or [])
        prompt = format_system_prompt(context, history_str, question)

        logger.info(f"Generating answer for: '{question[:60]}'")
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=600,
            )
            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated {len(answer)} chars")
            return answer
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return (
                f"I encountered an error while generating a response. "
                f"Please try again later. (Error: {str(e)})"
            )

    # ── Full Query ────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        section: Optional[str] = None,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Complete RAG query: detect intent → retrieve → generate."""
        start_time = time.time()

        documents, metadata_list, scores = self.retrieve(question, section, top_k)
        answer = self.generate(question, documents, chat_history)

        citations = self.extract_citations(answer)
        if not citations and documents:
            citations = list({doc.metadata.get("section", "Unknown") for doc in documents})

        latency = time.time() - start_time

        result: Dict[str, Any] = {
            "answer": answer,
            "citations": list(set(citations)),
            "retrieved_chunk_count": len(documents),
            "latency": round(latency, 3),
            "retrieved_sections": list({
                doc.metadata.get("section", "Unknown") for doc in documents
            }),
            "retrieved_chunks": [
                {
                    "content": doc.page_content[:200] + "...",
                    "section": doc.metadata.get("section", "Unknown"),
                    "score": score,
                }
                for doc, score in zip(documents, scores)
            ],
            "is_refusal": (
                "couldn't find" in answer.lower()
                or "could not find" in answer.lower()
                or "i don't have" in answer.lower()
            ),
        }

        logger.info(f"Query done in {latency:.2f}s, {len(documents)} chunks")
        return result

    @staticmethod
    def extract_citations(answer: str) -> List[str]:
        """Extract **[Section]** and [Section | Page] style citations."""
        hits = re.findall(r'\*\*\[([^\]]+)\]\*\*', answer)
        hits += re.findall(r'\[([^\]|]+?)(?:\s*\|\s*[^\]]+)?\]', answer)
        # Filter out numeric-only matches (page refs)
        return [h.strip() for h in hits if not h.strip().isdigit()]


# ── LLM Judge ─────────────────────────────────────────────────────────────────

class LLMJudge:
    """LLM-based evaluator for RAG responses."""

    def __init__(self) -> None:
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
                max_tokens=400,
            )
            return self._parse_json(response.choices[0].message.content.strip())
        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            return {"pass": False, "reason": f"Judge error: {str(e)}", "score": 0}

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if m:
            text = m.group(1)
        m2 = re.search(r'(\{.*\})', text, re.DOTALL)
        if m2:
            text = m2.group(1)
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            text = re.sub(r',\s*([}\]])', r'\1', text.replace("'", '"'))
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {
                    "pass": False,
                    "reason": f"Parse error: {text[:200]}",
                    "score": 0,
                }
