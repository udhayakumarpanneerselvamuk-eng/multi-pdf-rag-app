import os
import tempfile
from collections import OrderedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Multi-PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

# --- Custom Styling ---
st.markdown(
    """
    <style>
    .source-badge {
        display: inline-block;
        background-color: #f0f2f6;
        color: #1f2937;
        border: 1px solid #d1d5db;
        border-radius: 14px;
        padding: 3px 10px;
        margin: 4px 6px 4px 0;
        font-size: 0.82rem;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Prompt Designed to Surface Conflicts & Stay Grounded ---
SYSTEM_PROMPT = """You are a precise document research assistant.
Answer the user's question using ONLY the provided context excerpts.

Rules:
1. Cite specific facts clearly and concisely.
2. CONFLICT DETECTION: If two or more excerpts contradict or disagree with each other,
   DO NOT silently pick one. Explicitly state that the documents disagree and present
   what each document says along with its file name and page number.
3. If the answer is not in the provided context, say:
   "I could not find that information in the uploaded documents."

Context:
{context}
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


# --- Helper Functions ---
def get_embeddings(provider: str):
    if provider == "OpenAI (Low Cost)":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OPENAI_API_KEY not found in environment variables.")
            st.stop()
        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    return OllamaEmbeddings(model="nomic-embed-text")


def get_llm(provider: str, model_name: str):
    if provider == "OpenAI (Low Cost)":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OPENAI_API_KEY not found in environment variables.")
            st.stop()
        return ChatOpenAI(model=model_name, temperature=0, api_key=api_key)
    return ChatOllama(model=model_name, temperature=0)


def build_vector_store(uploaded_files, provider: str, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    file_names = []

    for uploaded_file in uploaded_files:
        file_names.append(uploaded_file.name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            for doc in pages:
                # Replace temp path with real filename and make page 1-indexed
                doc.metadata["source"] = uploaded_file.name
                doc.metadata["page"] = int(doc.metadata.get("page", 0)) + 1

            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)
        finally:
            os.remove(tmp_path)

    embeddings = get_embeddings(provider)
    vector_store = FAISS.from_documents(all_chunks, embeddings)
    return vector_store, len(all_chunks), file_names


def format_context(docs) -> str:
    formatted = []
    for idx, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Excerpt {idx} | File: {src}, Page: {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def extract_unique_sources(docs) -> list[str]:
    unique = OrderedDict()
    for doc in docs:
        src = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        label = f"{src} - page {page}"
        unique[label] = True
    return list(unique.keys())


# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_provider" not in st.session_state:
    st.session_state.active_provider = None

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")

    provider = st.radio(
        "Model Provider",
        options=["Ollama (Free Local)", "OpenAI (Low Cost)"],
        index=0,
        help="Use Ollama for $0 local testing. Switch to OpenAI for final verification.",
    )

    if provider == "Ollama (Free Local)":
        model_name = st.selectbox("Ollama Chat Model", ["llama3.2", "qwen2.5:3b", "mistral"])
        st.caption("Embeddings: `nomic-embed-text` (local)")
    else:
        model_name = st.selectbox("OpenAI Chat Model", ["gpt-4o-mini", "gpt-6-luna"])
        st.caption("Embeddings: `text-embedding-3-small` ($0.02/1M tokens)")
        if not os.getenv("OPENAI_API_KEY"):
            st.warning("Set `OPENAI_API_KEY` in `.env` or your terminal before indexing.")

    # Reset FAISS store if provider changes (different embedding dimensions)
    if st.session_state.active_provider and st.session_state.active_provider != provider:
        st.session_state.pop("store", None)
        st.session_state.pop("chunk_count", None)
        st.session_state.pop("file_names", None)
        st.info("Provider changed. Please re-index your PDFs.")
    st.session_state.active_provider = provider

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload PDFs (Multiple Allowed)",
        type="pdf",
        accept_multiple_files=True,
    )

    with st.expander("Chunking & Retrieval Settings"):
        chunk_size = st.slider("Chunk Size", 400, 1500, 1000, step=100)
        chunk_overlap = st.slider("Chunk Overlap", 50, 300, 150, step=25)
        top_k = st.slider("Retrieved Chunks (k)", 2, 8, 4)

    index_btn = st.button("🚀 Process & Index PDFs", use_container_width=True, type="primary")

    if index_btn:
        if not uploaded_files:
            st.warning("Please upload at least one PDF file first.")
        else:
            with st.spinner(f"Chunking & embedding with {provider}..."):
                store, chunk_count, file_names = build_vector_store(
                    uploaded_files, provider, chunk_size, chunk_overlap
                )
                st.session_state.store = store
                st.session_state.chunk_count = chunk_count
                st.session_state.file_names = file_names
                st.session_state.messages = []
            st.success(f"Indexed {chunk_count} chunks across {len(file_names)} file(s).")

    if "store" in st.session_state:
        st.divider()
        st.metric("Indexed Chunks", st.session_state.chunk_count)
        st.caption("**Uploaded:** " + ", ".join(st.session_state.file_names))
        if st.button("🗑️ Clear Chat & Index", use_container_width=True):
            for key in ["store", "chunk_count", "file_names", "messages"]:
                st.session_state.pop(key, None)
            st.rerun()

# --- Main Interface ---
st.title("📚 Multi-PDF RAG App with Sources")
st.caption(
    "Upload multiple PDFs, index them into a single FAISS database, and ask questions with exact file & page citations."
)

if "store" not in st.session_state:
    st.info("👈 Upload one or more PDFs in the sidebar and click **Process & Index PDFs** to get started.")
else:
    st.success(
        f"**Ready:** {', '.join(st.session_state.file_names)} — "
        f"**Indexed {st.session_state.chunk_count} chunks** using `{provider}`."
    )

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                badges = "".join(
                    f'<span class="source-badge">📄 {src}</span>' for src in msg["sources"]
                )
                st.markdown(f"**Sources:** {badges}", unsafe_allow_html=True)
                with st.expander("Inspect retrieved excerpts"):
                    for i, excerpt in enumerate(msg["excerpts"], start=1):
                        st.markdown(f"**{i}. {excerpt['source']} — Page {excerpt['page']}**")
                        st.caption(excerpt["text"])

    # Chat input
    if question := st.chat_input("Ask a question across all uploaded PDFs..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                retriever = st.session_state.store.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": top_k, "fetch_k": max(10, top_k * 3)},
                )
                docs = retriever.invoke(question)

                context = format_context(docs)
                llm = get_llm(provider, model_name)
                chain = PROMPT_TEMPLATE | llm
                response = chain.invoke({"context": context, "question": question})
                answer_text = response.content

                sources = extract_unique_sources(docs)
                excerpts = [
                    {
                        "source": d.metadata.get("source", "Unknown"),
                        "page": d.metadata.get("page", "?"),
                        "text": d.page_content,
                    }
                    for d in docs
                ]

            st.markdown(answer_text)
            badges = "".join(f'<span class="source-badge">📄 {src}</span>' for src in sources)
            st.markdown(f"**Sources:** {badges}", unsafe_allow_html=True)

            with st.expander("Inspect retrieved excerpts"):
                for i, excerpt in enumerate(excerpts, start=1):
                    st.markdown(f"**{i}. {excerpt['source']} — Page {excerpt['page']}**")
                    st.caption(excerpt["text"])

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer_text,
                "sources": sources,
                "excerpts": excerpts,
            }
        )