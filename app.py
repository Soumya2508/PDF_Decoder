"""Streamlit UI for PDF_Decoder.

Kept intentionally thin: parse user actions, call the pipeline, render the
result. All RAG logic lives in `src/rag/`.
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from src.rag.pipeline import RAGPipeline

load_dotenv()

st.set_page_config(page_title="PDF_Decoder", page_icon=":page_facing_up:", layout="centered")


@st.cache_resource(show_spinner=False)
def get_pipeline(api_key: str) -> RAGPipeline:
    return RAGPipeline(api_key=api_key)


# --- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("Setup")

    env_key = os.environ.get("GROQ_API_KEY", "")
    api_key = st.text_input(
        "Groq API key",
        value=env_key,
        type="password",
        help="Get a free key at console.groq.com. Or set GROQ_API_KEY in .env.",
    )

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    build_clicked = st.button("Build index", type="primary", use_container_width=True)

    st.divider()
    st.caption("Pipeline: PDF -> chunks -> MiniLM embeddings -> FAISS -> Groq LLM")


# --- Main ------------------------------------------------------------------
st.title("PDF_Decoder")
st.write("Ask grounded questions about an annual report (or any PDF).")

if "ingest_stats" not in st.session_state:
    st.session_state.ingest_stats = None

if build_clicked:
    if not api_key:
        st.error("Please paste your Groq API key in the sidebar.")
    elif uploaded is None:
        st.error("Please upload a PDF first.")
    else:
        with st.spinner("Reading PDF, chunking, embedding, and indexing..."):
            pipeline = get_pipeline(api_key)
            try:
                stats = pipeline.ingest(uploaded.getvalue())
                st.session_state.ingest_stats = stats
            except Exception as exc:  # surfaced to the user
                st.session_state.ingest_stats = None
                st.error(f"Ingest failed: {exc}")

if st.session_state.ingest_stats:
    s = st.session_state.ingest_stats
    st.success(f"Indexed {s.num_chunks} chunks from {s.num_pages} pages.")

question = st.text_input("Your question", placeholder="e.g. What was total revenue?")
ask_clicked = st.button("Ask")

if ask_clicked:
    if not st.session_state.ingest_stats:
        st.warning("Build the index first (sidebar).")
    elif not question.strip():
        st.warning("Type a question.")
    else:
        with st.spinner("Retrieving context and asking the model..."):
            pipeline = get_pipeline(api_key)
            try:
                result = pipeline.ask(question)
                st.markdown("### Answer")
                st.markdown(result.answer.text)
                st.caption(f"Model: {result.answer.model}")
            except Exception as exc:
                st.error(f"Query failed: {exc}")
