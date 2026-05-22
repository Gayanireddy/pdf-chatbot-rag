"""Vector store module for RAG chatbot.

Handles embedding generation and FAISS vector store operations.
"""

import os
from typing import List, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""
    pass


def create_embeddings_model(
    model_name: str = "models/gemini-embedding-001",
    task_type: str = "retrieval_document",
    google_api_key: Optional[str] = None
) -> GoogleGenerativeAIEmbeddings:
    """Create a Google Generative AI embeddings model.

    Args:
        model_name: Name of the embedding model (default: models/gemini-embedding-001)
        task_type: Gemini task type — use "retrieval_document" when embedding PDF chunks
                   for indexing, and "retrieval_query" when embedding user questions.
        google_api_key: Google API key (if not provided, reads from GOOGLE_API_KEY env var)

    Returns:
        GoogleGenerativeAIEmbeddings instance

    Raises:
        VectorStoreError: If embeddings model creation fails
    """
    try:
        api_key = google_api_key or os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise VectorStoreError(
                "GOOGLE_API_KEY not found. Set it in .env file or pass as parameter."
            )

        embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name,
            task_type=task_type,
            google_api_key=api_key
        )

        return embeddings

    except Exception as e:
        raise VectorStoreError(f"Failed to create embeddings model: {str(e)}") from e


def create_vector_store(chunks) -> FAISS:
    """Create a FAISS vector store from document chunks.

    Args:
        chunks: List of Document chunks to embed and index

    Returns:
        FAISS vector store instance

    Raises:
        VectorStoreError: If vector store creation fails
    """
    try:
        embeddings = create_embeddings_model(task_type="retrieval_document")
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store
    except VectorStoreError:
        raise
    except Exception as e:
        raise VectorStoreError(f"Failed to create vector store: {str(e)}") from e


def save_vector_store(vector_store: FAISS, save_path: str = "faiss_index") -> None:
    """Save FAISS vector store to disk.

    Args:
        vector_store: FAISS vector store instance to save
        save_path: Directory path to save the index (default: faiss_index)

    Raises:
        VectorStoreError: If save operation fails
    """
    try:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        vector_store.save_local(save_path)
    except Exception as e:
        raise VectorStoreError(f"Failed to save vector store: {str(e)}") from e


def load_vector_store(
    load_path: str = "faiss_index",
    embeddings: Optional[GoogleGenerativeAIEmbeddings] = None
) -> FAISS:
    """Load FAISS vector store from disk.

    Args:
        load_path: Directory path to load the index from (default: faiss_index)
        embeddings: Embeddings model (if not provided, creates default model)

    Returns:
        FAISS vector store instance

    Raises:
        VectorStoreError: If load operation fails
    """
    try:
        if not os.path.exists(load_path):
            raise VectorStoreError(f"Vector store path does not exist: {load_path}")

        if embeddings is None:
            embeddings = create_embeddings_model()

        vector_store = FAISS.load_local(
            load_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        return vector_store

    except VectorStoreError:
        raise
    except Exception as e:
        raise VectorStoreError(f"Failed to load vector store: {str(e)}") from e


def similarity_search(vector_store: FAISS, query: str, k: int = 4) -> List[Document]:
    """Perform similarity search on the vector store.

    Args:
        vector_store: FAISS vector store instance
        query: Query string to search for
        k: Number of top results to return (default: 4)

    Returns:
        List of most similar Document objects

    Raises:
        VectorStoreError: If search operation fails
    """
    try:
        results = vector_store.similarity_search(query, k=k)
        return results
    except Exception as e:
        raise VectorStoreError(f"Similarity search failed: {str(e)}") from e


def create_retriever(vector_store: FAISS, k: int = 4):
    """Create a retriever from the vector store.

    Args:
        vector_store: FAISS vector store instance
        k: Number of documents to retrieve (default: 4)

    Returns:
        VectorStoreRetriever instance
    """
    return vector_store.as_retriever(search_kwargs={"k": k})
