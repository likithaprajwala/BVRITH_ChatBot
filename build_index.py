"""
Document ingestion and vector database builder for BVRIT RAG Chatbot.

Handles loading documents, chunking, embedding, and storing in ChromaDB.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
import pymupdf  # PyMuPDF for image-based PDFs
import easyocr
import numpy as np
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()

# Load environment variables from the script directory
load_dotenv(SCRIPT_DIR / ".env")

# --- Configuration Constants ---
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", str(SCRIPT_DIR / "chroma_db"))
# Resolve document path relative to script directory if not absolute
_env_doc_path = os.getenv("DOCUMENT_PATH", str(SCRIPT_DIR / "data" / "bvrit_info.pdf"))
if not os.path.isabs(_env_doc_path):
    _env_doc_path = str(SCRIPT_DIR / _env_doc_path)
DOCUMENT_PATH: str = _env_doc_path
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K: int = int(os.getenv("TOP_K", "5"))
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"


# Available sections in the document
AVAILABLE_SECTIONS: List[str] = [
    "About",
    "Departments",
    "Admissions",
    "Fee Structure",
    "Placements",
    "Facilities",
    "Faculty",
    "Contact",
]


def create_embeddings() -> OpenAIEmbeddings:
    """
    Create OpenAI embeddings instance configured to use OpenRouter.

    Returns:
        OpenAIEmbeddings: Configured embeddings instance.
    """
    return OpenAIEmbeddings(
        openai_api_key=OPENROUTER_API_KEY,
        model=EMBEDDING_MODEL,
        openai_api_base=OPENROUTER_BASE_URL,
    )


# Lazy-loaded OCR reader (initialized once)
_ocr_reader = None


def get_ocr_reader():
    """Get or create the EasyOCR reader (singleton)."""
    global _ocr_reader
    if _ocr_reader is None:
        logger.info("Initializing EasyOCR reader (CPU mode)...")
        _ocr_reader = easyocr.Reader(['en'], gpu=False)
    return _ocr_reader


def load_pdf_with_ocr(file_path: str) -> List[Document]:
    """
    Load a PDF using OCR for image-based/scanned PDFs.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of Document objects with OCR-extracted text.
    """
    logger.info(f"Loading PDF with OCR from: {file_path}")
    reader = get_ocr_reader()
    doc = pymupdf.open(file_path)
    documents = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        # First try extracting text directly
        text = page.get_text().strip()

        if len(text) < 20:  # If little to no text, use OCR
            logger.info(f"Page {page_num + 1} has minimal text ({len(text)} chars), using OCR...")
            pix = page.get_pixmap(dpi=300)
            # Convert to numpy array (EasyOCR expects numpy array, RGB format)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            # If RGBA (4 channels), convert to RGB (3 channels)
            if pix.n == 4:
                img_array = img_array[:, :, :3]
            elif pix.n == 1:
                # Grayscale, convert to RGB by stacking
                img_array = np.stack([img_array.squeeze()] * 3, axis=-1)
            results = reader.readtext(img_array, paragraph=True)
            text = "\n".join([r[1] for r in results])
            logger.info(f"OCR extracted {len(text)} chars from page {page_num + 1}")

        doc_obj = Document(
            page_content=text,
            metadata={
                "source": os.path.basename(file_path),
                "page": page_num + 1,
            }
        )
        documents.append(doc_obj)

    doc.close()
    logger.info(f"Loaded {len(documents)} page(s) via OCR")
    return documents


def load_document(file_path: str) -> List[Document]:
    """
    Load a document using the appropriate loader based on file extension.

    Supports .docx (via Docx2txtLoader) and .pdf (via PyPDFLoader with OCR fallback).

    Args:
        file_path: Path to the document file.

    Returns:
        List of Document objects.

    Raises:
        FileNotFoundError: If the document does not exist.
        ValueError: If the file format is not supported.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document not found at: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".docx":
        logger.info(f"Loading .docx document from: {file_path}")
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
    elif ext == ".pdf":
        # Try standard PDF loader first
        try:
            logger.info(f"Loading .pdf document from: {file_path}")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            # Check if text was actually extracted
            total_chars = sum(len(d.page_content) for d in documents)
            if total_chars < 20:
                logger.info(f"Standard PDF loader extracted only {total_chars} chars, falling back to OCR...")
                documents = load_pdf_with_ocr(file_path)
        except Exception as e:
            logger.warning(f"Standard PDF loader failed: {e}, falling back to OCR...")
            documents = load_pdf_with_ocr(file_path)
    else:
        raise ValueError(
            f"Unsupported file format '{ext}'. Please provide a .docx or .pdf file."
        )

    logger.info(f"Loaded {len(documents)} document(s)")
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.

    Args:
        documents: List of Document objects to split.

    Returns:
        List of chunked Document objects with metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks")

    # Add metadata to each chunk
    for i, chunk in enumerate(chunks):
        # Detect section from content keywords
        section = detect_section(chunk.page_content)
        chunk.metadata["section"] = section
        chunk.metadata["chunk_id"] = f"chunk_{i:04d}"
        chunk.metadata["source"] = os.path.basename(DOCUMENT_PATH)

    return chunks


def detect_section(text: str) -> str:
    """
    Detect which section the text belongs to based on keywords.

    Args:
        text: The text content to analyze.

    Returns:
        The detected section name.
    """
    text_lower = text.lower()

    section_keywords = {
        "About": ["about bvrit", "about the college", "vision", "mission", "introduction", "overview"],
        "Departments": ["department", "cse", "ece", "eee", "mech", "civil", "engineering"],
        "Admissions": ["admission", "eligibility", "entrance", "jee", "eamcet", "apply", "seat"],
        "Fee Structure": ["fee", "tuition", "cost", "payment", "scholarship"],
        "Placements": ["placement", "recruit", "company", "offer", "package", "career"],
        "Facilities": ["facility", "library", "lab", "hostel", "cafeteria", "transport", "sports"],
        "Faculty": ["faculty", "professor", "teacher", "staff", "phd"],
        "Contact": ["contact", "email", "phone", "address", "website"],
    }

    for section, keywords in section_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                return section

    return "General"


def build_vector_store(chunks: List[Document], persist_directory: str) -> Chroma:
    """
    Build a ChromaDB vector store from document chunks.

    Args:
        chunks: List of chunked Document objects.
        persist_directory: Directory to persist the ChromaDB database.

    Returns:
        Chroma: The built vector store.
    """
    embeddings = create_embeddings()

    logger.info(f"Building vector store at: {persist_directory}")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    vector_store.persist()
    logger.info("Vector store built and persisted successfully")

    return vector_store


def load_existing_vector_store(persist_directory: str) -> Optional[Chroma]:
    """
    Load an existing ChromaDB vector store if it exists.

    Args:
        persist_directory: Path to the persisted ChromaDB database.

    Returns:
        Chroma vector store if exists, None otherwise.
    """
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        try:
            embeddings = create_embeddings()
            logger.info(f"Loading existing vector store from: {persist_directory}")
            vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
            )
            logger.info("Existing vector store loaded successfully")
            return vector_store
        except Exception as e:
            logger.warning(f"Failed to load existing vector store: {e}")
            return None
    return None


def get_vector_store() -> Chroma:
    """
    Get or create the vector store.

    Returns:
        Chroma vector store.
    """
    persist_directory = CHROMA_DB_PATH

    # Try to load existing vector store
    existing_store = load_existing_vector_store(persist_directory)
    if existing_store is not None:
        collection_size = len(existing_store.get()["ids"])
        logger.info(f"Using existing vector store with {collection_size} chunks")
        logger.info(f"Embedding model: {EMBEDDING_MODEL}")
        logger.info(f"Persistence path: {os.path.abspath(persist_directory)}")
        return existing_store

    # Build new vector store
    logger.info("No existing vector store found. Building new one...")
    documents = load_document(DOCUMENT_PATH)
    chunks = split_documents(documents)
    vector_store = build_vector_store(chunks, persist_directory)

    # Print summary
    logger.info(f"Total chunks: {len(chunks)}")
    logger.info(f"Embedding model: {EMBEDDING_MODEL}")
    logger.info(f"Persistence path: {os.path.abspath(persist_directory)}")

    return vector_store


def main() -> None:
    """Main entry point for building the index."""
    print("=" * 60)
    print("BVRIT RAG Chatbot - Index Builder")
    print("=" * 60)

    try:
        vector_store = get_vector_store()
        collection = vector_store.get()
        print(f"\n✅ Index ready!")
        print(f"   Total chunks: {len(collection['ids'])}")
        print(f"   Embedding model: {EMBEDDING_MODEL}")
        print(f"   Persistence path: {os.path.abspath(CHROMA_DB_PATH)}")
    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()