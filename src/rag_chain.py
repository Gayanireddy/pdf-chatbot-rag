"""RAG chain module for question answering.

Implements the retrieval-augmented generation chain using LangChain LCEL.
"""

from typing import Optional
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from src.vector_store import create_embeddings_model


class RAGChainError(Exception):
    """Raised when RAG chain operations fail."""
    pass


def create_llm(
    model_name: str = "gemini-2.0-flash",
    temperature: float = 0,
    google_api_key: Optional[str] = None
) -> ChatGoogleGenerativeAI:
    """Create a Google Gemini LLM instance.

    Args:
        model_name: Name of the Gemini model (default: gemini-2.0-flash)
        temperature: Temperature for generation (default: 0 for deterministic RAG)
        google_api_key: Google API key (if not provided, reads from GOOGLE_API_KEY env var)

    Returns:
        ChatGoogleGenerativeAI instance

    Raises:
        RAGChainError: If LLM creation fails
    """
    try:
        api_key = google_api_key or os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise RAGChainError(
                "GOOGLE_API_KEY not found. Set it in .env file or pass as parameter."
            )

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key
        )

        return llm

    except Exception as e:
        raise RAGChainError(f"Failed to create LLM: {str(e)}") from e


def create_rag_prompt() -> ChatPromptTemplate:
    """Create the RAG prompt template.

    Returns:
        ChatPromptTemplate instance
    """
    system_template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer based on the context, say that you don't know.
Keep the answer concise and accurate.

Context:
{context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "{input}")
    ])

    return prompt


def create_rag_chain(vector_store: FAISS, llm: Optional[ChatGoogleGenerativeAI] = None):
    """Create a complete RAG chain using LCEL.

    Args:
        vector_store: FAISS vector store instance
        llm: ChatGoogleGenerativeAI instance (if not provided, creates default)

    Returns:
        Retrieval chain that can be invoked with {"input": "question"}

    Raises:
        RAGChainError: If chain creation fails
    """
    try:
        if llm is None:
            llm = create_llm()

        # Override FAISS embedding function with retrieval_query task type.
        # The index was built with retrieval_document embeddings; at query time
        # Gemini expects retrieval_query so the two vectors align in the same space.
        query_embeddings = create_embeddings_model(task_type="retrieval_query")
        vector_store.embedding_function = query_embeddings.embed_query

        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        prompt = create_rag_prompt()

        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        return retrieval_chain

    except Exception as e:
        raise RAGChainError(f"Failed to create RAG chain: {str(e)}") from e


def ask_question(chain, question: str) -> dict:
    """Ask a question using the RAG chain.

    Args:
        chain: Retrieval chain created by create_rag_chain
        question: User question string

    Returns:
        Dictionary with keys:
        - 'answer': The generated answer
        - 'context': List of source documents used

    Raises:
        RAGChainError: If question answering fails
    """
    try:
        response = chain.invoke({"input": question})
        return response
    except Exception as e:
        raise RAGChainError(f"Failed to answer question: {str(e)}") from e
