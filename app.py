"""PDF Chatbot RAG - Streamlit Application"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.pdf_ingestion import ingest_pdf, PDFIngestionError
from src.vector_store import (
    create_vector_store,
    save_vector_store,
    load_vector_store,
    VectorStoreError,
)
from src.rag_chain import (
    create_rag_chain,
    create_rag_prompt,
    ask_question,
    RAGChainError,
)

st.set_page_config(page_title="PDF Chatbot RAG", page_icon="📚")

st.title("📚 PDF Chatbot RAG")
st.write("Upload a PDF and ask questions about its content")

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.get("pdf_name"):
        try:
            with st.spinner("Processing PDF..."):
                chunks = ingest_pdf(uploaded_file)
                st.success(f"✅ PDF processed successfully!")
                st.info(f"Created **{len(chunks)}** chunks from your document")

            with st.spinner("Creating embeddings and building vector store..."):
                vector_store = create_vector_store(chunks)
                save_vector_store(vector_store, save_path="faiss_index")
                st.success("✅ Vector store created and saved!")

                st.session_state.chunks = chunks
                st.session_state.vector_store = vector_store
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.rag_chain = create_rag_chain(vector_store)
                st.session_state.messages = []

        except PDFIngestionError as e:
            st.error(f"❌ Error processing PDF: {str(e)}")
        except VectorStoreError as e:
            st.error(f"❌ Error creating vector store: {str(e)}")
        except RAGChainError as e:
            st.error(f"❌ Error creating RAG chain: {str(e)}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

    if "chunks" in st.session_state:
        chunks = st.session_state.chunks
        pages = len(set(chunk.metadata.get('page', 0) for chunk in chunks))
        vectors = st.session_state.vector_store.index.ntotal if "vector_store" in st.session_state else len(chunks)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pages Extracted", pages)
        with col2:
            st.metric("Chunks Created", len(chunks))
        with col3:
            st.metric("Vectors Indexed", vectors)

        with st.expander("🔍 Preview First Chunk"):
            st.markdown("**Content:**")
            st.text(chunks[0].page_content[:500] + "..." if len(chunks[0].page_content) > 500 else chunks[0].page_content)
            st.markdown("**Metadata:**")
            st.json(chunks[0].metadata)

elif "chunks" in st.session_state:
    st.info(f"📄 Currently loaded: **{st.session_state.pdf_name}** ({len(st.session_state.chunks)} chunks)")

    if "vector_store" not in st.session_state:
        try:
            with st.spinner("Loading vector store..."):
                st.session_state.vector_store = load_vector_store()
            st.success("✅ Vector store loaded from disk")
        except VectorStoreError as e:
            st.warning(f"⚠️ Could not load vector store: {str(e)}")

    if "rag_chain" not in st.session_state and "vector_store" in st.session_state:
        try:
            st.session_state.rag_chain = create_rag_chain(st.session_state.vector_store)
        except RAGChainError as e:
            st.error(f"❌ Error creating RAG chain: {str(e)}")

    if st.button("Clear and upload new PDF"):
        for key in ["chunks", "vector_store", "pdf_name", "rag_chain", "messages"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

if "rag_chain" in st.session_state:
    st.markdown("---")
    st.subheader("💬 Ask Questions")

    with st.expander("🧠 View Prompt Template"):
        prompt = create_rag_prompt()
        for msg in prompt.messages:
            role = "System" if msg.__class__.__name__ == "SystemMessagePromptTemplate" else "Human"
            st.markdown(f"**{role}:**")
            st.code(msg.prompt.template, language="text")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📄 View Source Documents"):
                    for i, doc in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}** (Page {doc.metadata.get('page', 'N/A')}):")
                        st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                        st.markdown("---")

    if question := st.chat_input("Ask a question about your PDF"):
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = ask_question(st.session_state.rag_chain, question)
                    answer = response["answer"]
                    context_docs = response.get("context", [])

                    st.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": context_docs
                    })

                    if context_docs:
                        with st.expander("📄 View Source Documents"):
                            for i, doc in enumerate(context_docs, 1):
                                st.markdown(f"**Source {i}** (Page {doc.metadata.get('page', 'N/A')}):")
                                st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                                st.markdown("---")

                except RAGChainError as e:
                    error_msg = f"❌ Error generating answer: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
