# 📚 Multi-PDF RAG App with Sources

A single-file Retrieval-Augmented Generation (RAG) web application built with **Streamlit**, **LangChain**, and **FAISS**. Upload multiple PDFs at once, index them into a unified vector store, and ask questions with exact **file name and page number** citations.

## ✨ Key Features
- **Multi-PDF Batch Upload:** Index multiple PDF documents into a single FAISS vector database stored in `st.session_state`.
- **Exact Source Attribution:** Every answer displays deduplicated `filename - page X` badges plus an expandable excerpt inspector for verification.
- **Conflict-Aware Prompting:** Explicitly surfaces disagreements when two documents state conflicting facts instead of silently picking one.
- **Dual Provider Support (Cost-Optimized):**
  - **Ollama (Free Local):** `llama3.2` + `nomic-embed-text` for $0 development and testing.
  - **OpenAI (Low Cost):** `gpt-4o-mini` / `gpt-6-luna` + `text-embedding-3-small` for fast, low-cost production queries.
- **Tunable Chunking & MMR Retrieval:** Uses `RecursiveCharacterTextSplitter` (default `1000` chunk size, `150` overlap) and Maximal Marginal Relevance (MMR) to avoid fragmented, repetitive citations.

## 🛠️ Tech Stack
- Python, Streamlit
- LangChain (`langchain-openai`, `langchain-ollama`, `langchain-community`, `langchain-text-splitters`)
- FAISS (`faiss-cpu`)
- PyPDF (`pypdf`)

## 🚀 Getting Started

### 1. Clone & Set Up Environment
```bash
git clone [https://github.com/](https://github.com/)<your-username>/multi-rag.git
cd multi-rag
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt