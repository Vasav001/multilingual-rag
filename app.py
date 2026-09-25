"""
Streamlit UI.
"""
import streamlit as st

from config import settings
from ingestion.pipeline import run_ingestion
from retrieval.embedder import embed_query
from retrieval.generator import generate_answer
from retrieval.language import code_from_dropdown, detect_language, dropdown_options, language_name
from retrieval.qdrant_store import ensure_collection, search
from retrieval.reranker import rerank

st.set_page_config(page_title="Multilingual RAG", page_icon="🌐", layout="centered")
st.title("🌐 Multilingual RAG")
st.caption("Ask a question in any language; the app retrieves across all ingested documents regardless of language.")

with st.sidebar:
    st.header("Document ingestion")
    st.write(f"Drop `.pdf`, `.txt`, `.docx` files into `{settings.RAW_DOCS_DIR}/`, then click ingest.")
    if st.button("Run ingestion", use_container_width=True):
        with st.spinner("Ingesting documents..."):
            log_lines: list[str] = []
            results = run_ingestion(log=lambda m: log_lines.append(m))
        if not results:
            st.warning("No supported files found.")
        else:
            for r in results:
                icon = {"ingested": "✅", "skipped": "⏭️", "empty": "⚠️"}[r["status"]]
                st.write(f"{icon} {r['filename']} — {r['status']} ({r['chunks']} chunks)")

    st.divider()
    st.header("Settings")
    st.write(f"Candidates retrieved: **{settings.TOP_K_CANDIDATES}**")
    st.write(f"Chunks sent to LLM: **{settings.TOP_N_FINAL}**")
    st.write(f"Embedding model: `{settings.EMBEDDING_MODEL}` ({settings.EMBEDDING_DIM}d)")
    st.write(f"Rerank model: `{settings.RERANK_MODEL}`")
    st.write(f"LLM: `{settings.GROQ_MODEL}`")

query = st.text_area("Your question (any language)", height=100, placeholder="e.g. What does the contract say about termination? / ¿Qué dice el documento sobre...?")
lang_choice = st.selectbox("Answer language", dropdown_options(), index=0)

if st.button("Ask", type="primary", disabled=not query.strip()):
    ensure_collection()
    with st.spinner("Retrieving..."):
        query_vector = embed_query(query)
        candidates = search(query_vector, limit=settings.TOP_K_CANDIDATES)
        top_chunks = rerank(query, candidates, top_n=settings.TOP_N_FINAL)

    output_lang = code_from_dropdown(lang_choice) or detect_language(query)

    with st.spinner(f"Generating answer in {language_name(output_lang)}..."):
        answer = generate_answer(query, output_lang, top_chunks)

    st.subheader("Answer")
    st.write(answer)
    st.caption(f"Answered in: {language_name(output_lang)}")

    if top_chunks:
        with st.expander(f"Sources ({len(top_chunks)} chunks used)"):
            for i, c in enumerate(top_chunks, start=1):
                loc = f", page {c['page']}" if c.get("page") else ""
                st.markdown(f"**[{i}] {c['filename']}{loc}** — relevance {c.get('rerank_score', 0):.2f}, language: {language_name(c.get('language', ''))}")
                st.text(c["text"][:500] + ("..." if len(c["text"]) > 500 else ""))
    else:
        st.info("No relevant chunks were found in the knowledge base for this question.")
