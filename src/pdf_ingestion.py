"""PDF ingestion module for RAG chatbot.

Handles PDF upload, text extraction, and chunking using LangChain components.
"""

import os
import tempfile
from typing import List
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class PDFIngestionError(Exception):
    """Raised when PDF ingestion fails."""
    pass


def load_pdf(file_path: str) -> List[Document]:
    """Load a PDF file and extract text using PyPDFLoader.

    Args:
        file_path: Path to the PDF file

    Returns:
        List of Document objects, one per PDF page

    Raises:
        PDFIngestionError: If PDF loading fails
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        if not documents:
            raise PDFIngestionError(f"No content extracted from PDF: {file_path}")

        # Guard against scanned/image-only PDFs. PyPDFLoader returns empty or
        # near-empty page_content for scans — the index would be built from
        # meaningless vectors and all answers would be hallucinated.
        total_chars = sum(len(doc.page_content.strip()) for doc in documents)
        if total_chars < 100:
            raise PDFIngestionError(
                f"This PDF appears to be scanned or image-based "
                f"({total_chars} characters extracted). "
                "Only text-based PDFs are supported."
            )

        return documents
    except PDFIngestionError:
        raise
    except Exception as e:
        raise PDFIngestionError(f"Failed to load PDF: {str(e)}") from e


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """Split documents into smaller chunks for embedding.

    Args:
        documents: List of Document objects to split
        chunk_size: Maximum size of each chunk (default: 1000)
        chunk_overlap: Number of characters to overlap between chunks (default: 200)

    Returns:
        List of Document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


def save_uploaded_file(uploaded_file) -> str:
    """Save a Streamlit uploaded file to a temporary location.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        Path to the saved temporary file

    Raises:
        PDFIngestionError: If file save fails
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        raise PDFIngestionError(f"Failed to save uploaded file: {str(e)}") from e


def cleanup_temp_file(file_path: str) -> None:
    """Remove a temporary file.

    Args:
        file_path: Path to the temporary file to remove
    """
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"Warning: Failed to cleanup temp file {file_path}: {str(e)}")


def ingest_pdf(uploaded_file, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Complete PDF ingestion pipeline: upload -> load -> split.

    Args:
        uploaded_file: Streamlit UploadedFile object
        chunk_size: Maximum size of each chunk (default: 1000)
        chunk_overlap: Number of characters to overlap between chunks (default: 200)

    Returns:
        List of Document chunks ready for embedding

    Raises:
        PDFIngestionError: If any step of ingestion fails
    """
    temp_path = None
    try:
        temp_path = save_uploaded_file(uploaded_file)
        documents = load_pdf(temp_path)
        chunks = split_documents(documents, chunk_size, chunk_overlap)
        return chunks
    finally:
        if temp_path:
            cleanup_temp_file(temp_path)
