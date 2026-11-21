# 📚 Multi-PDF RAG App with Sources

A single-file Retrieval-Augmented Generation (RAG) web application built with **Streamlit**, **LangChain**, and **FAISS**. Upload multiple PDFs at once, index them into one unified vector database, and ask questions with exact **file name and page number** citations.

**🌐 Live App:** [Launch Multi-PDF RAG App](https://multi-pdf-rag-app-nslb9ngrg2pkk8daxcmnwu.streamlit.app/) *(Bring Your Own OpenAI API Key)*

---

## ✨ Key Features

- **Multi-PDF Batch Indexing:** Upload multiple PDF documents at once and embed them into a single in-memory **FAISS** vector store stored in `st.session_state`, with live chunk-count reporting.
- **Verifiable Source Citations:** Every answer displays deduplicated `filename - page X` badges (1-indexed) alongside an expandable **Inspect retrieved excerpts** panel.
- **MMR Retrieval:** Uses **Maximal Marginal Relevance (`search_type="mmr"`)** instead of basic similarity search to retrieve diverse, non-redundant chunks across documents.
- **Conflict-Aware Prompting:** Explicitly surfaces disagreements when two uploaded PDFs state conflicting facts—quoting what each file and page says rather than silently picking one.
- **Dual-Provider, Cost-Conscious Design:**
  - **OpenAI (Live Web App — BYOK):** Visitors enter their own OpenAI API key in the masked sidebar input (kept only in their browser session). Uses low-cost `gpt-4o-mini` + `text-embedding-3-small`.
  - **Ollama (100% Free Local Mode):** Automatically detects a local Ollama instance, checks for required models (`llama3.2`, `qwen2.5:3b`, `mistral`, and `nomic-embed-text`), and provides a **one-click in-app model downloader** with a live progress bar.

---

## 📸 Demo

### How It Works
![How Multi-PDF RAG Works](assets/rag-architecture.jpg)

---

## 🏗️ Architecture

| Stage | Implementation |
| :--- | :--- |
| **Document Loader** | `PyPDFLoader` (preserves real uploaded filename & 1-indexed page metadata) |
| **Chunking** | `RecursiveCharacterTextSplitter` (default `chunk_size=1000`, `chunk_overlap=150`, tunable in UI) |
| **Embeddings** | `text-embedding-3-small` (OpenAI) or `nomic-embed-text` (Ollama) |
| **Vector Store** | `FAISS` (`faiss-cpu`) stored in `st.session_state` |
| **Retrieval** | Dense Retrieval with **MMR** (`k=4`, tunable in UI) |
| **Generation** | Grounded, conflict-aware `ChatPromptTemplate` with `ChatOpenAI` or `ChatOllama` |

---

## 🚀 Run Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/udhayakumarpanneerselvamuk-eng/multi-pdf-rag-app.git](https://github.com/udhayakumarpanneerselvamuk-eng/multi-pdf-rag-app.git)
cd multi-pdf-rag-app
```

### 2. Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run rag_app.py
```

### 4. Choose Your Provider in the Sidebar
- **Option A — Free Local (Ollama):** Install [Ollama](https://ollama.com/download), start it, select **Ollama (Free Local)** in the sidebar, and click **⬇️ Download Missing Models** directly in the UI.
- **Option B — OpenAI (Low Cost):** Paste your OpenAI API key into the sidebar input box (or save `OPENAI_API_KEY=sk-...` in a local `.env` file).

---

## 🧠 Design Decisions ("Think It Through")

- **Handling Contradictions Across PDFs:** When multiple documents live in one vector store, older and newer policies can conflict. The system prompt strictly forbids silently choosing one answer and instructs the LLM to report both claims with their exact file names and page numbers.
- **Preventing Fragmented Citations:** Using `chunk_size=1000`, `chunk_overlap=150`, and MMR retrieval keeps paragraphs intact so answers cite 1–2 focused pages instead of scattering across 5 tiny fragments.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Orchestration:** LangChain (`langchain`, `langchain-openai`, `langchain-ollama`, `langchain-community`, `langchain-text-splitters`)
- **Vector Database:** FAISS (`faiss-cpu`)
- **PDF Parsing:** PyPDF (`pypdf`)