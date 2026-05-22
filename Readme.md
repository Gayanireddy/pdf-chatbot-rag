# PDF Chatbot RAG

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload any PDF and ask natural language questions about its content. Built with Python, LangChain, FAISS, Streamlit, and Google Gemini API.

---

## How It Works

1. You upload a PDF
2. The app splits it into small text chunks and converts them into numerical vectors (embeddings) using Gemini
3. Those vectors are stored in a local FAISS index
4. When you ask a question, your question is also converted to a vector, and the 4 most relevant chunks are retrieved
5. Those chunks + your question are sent to Gemini, which generates a grounded answer

---

## Prerequisites

- Python 3.12
- A free Google Gemini API key — get one at [aistudio.google.com](https://aistudio.google.com)

---

## Setup & Run

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd pdf-chatbot-rag
```

### 2. Create and activate a virtual environment

```bash
# Linux / Mac
python3.12 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your actual key:

```
GOOGLE_API_KEY=your_actual_key_here
```

### 5. Run the app

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## Usage

1. Click **Browse files** and upload a text-based PDF (scanned/image PDFs are not supported)
2. Wait for the processing spinner — the app embeds all chunks and saves the FAISS index to `faiss_index/`
3. Type a question in the chat box at the bottom
4. The answer will appear along with the source pages it was drawn from

---

## Project Structure

```
pdf-chatbot-rag/
├── app.py                  # Streamlit UI — entry point
├── src/
│   ├── pdf_ingestion.py    # PDF loading and text chunking
│   ├── vector_store.py     # Embeddings and FAISS index management
│   └── rag_chain.py        # RAG chain — retrieval + LLM answering
├── faiss_index/            # Saved vector index (auto-generated, gitignored)
├── requirements.txt
└── .env.example
```

---

## Tech Stack

| Component | Library |
|-----------|---------|
| UI | Streamlit 1.57 |
| LLM | Google Gemini 1.5 Flash via `langchain-google-genai` |
| Embeddings | `models/gemini-embedding-001` |
| Vector Store | FAISS (local, no server needed) |
| PDF Parsing | PyPDF via `langchain-community` |
| Orchestration | LangChain 1.x (LCEL) |

---

## Notes

- Only **text-based PDFs** work. Scanned / image-only PDFs will be rejected.
- The Gemini free tier allows ~15 requests/min and 1500/day — sufficient for single-user testing.
- The FAISS index is saved to `faiss_index/` after the first upload, so re-opening the app does not re-embed the same PDF.
- `temperature=0` is used for the LLM to keep answers deterministic and grounded in the retrieved context.
