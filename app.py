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

    env_key = os.environ.get("GROQ_API_KEY", "").strip()
    if env_key:
        # Key already configured server-side (e.g. via .env or Render env var).
        # Do NOT echo it to the UI — anyone visiting the deployed URL could
        # otherwise reveal it by clicking the password-field eye icon.
        api_key = env_key
        st.success("Groq API key loaded from environment.")
    else:
        api_key = st.text_input(
            "Groq API key",
            value="",
            type="password",
            help="Paste your key for this session only. Get one free at console.groq.com.",
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

                with st.expander(f"Sources ({len(result.hits)} chunks retrieved)"):
                    for h in result.hits:
                        st.markdown(
                            f"**Page {h.chunk.page_number}** "
                            f"&nbsp; _chunk {h.chunk.chunk_id}, distance {h.score:.3f}_"
                        )
                        snippet = h.chunk.text
                        if len(snippet) > 600:
                            snippet = snippet[:600] + "..."
                        st.write(snippet)
                        st.divider()
            except Exception as exc:
                st.error(f"Query failed: {exc}")
