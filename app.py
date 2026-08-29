import hashlib

import streamlit as st
from dotenv import load_dotenv

from generate import create_llm, generate_answer
from rag_utils import (
    create_embeddings,
    create_vector_store,
    create_bm25_documents,
    create_bm25_index,
    create_reranker,
    extract_text_from_docx,
    extract_text_from_pdf,
    extract_text_from_txt,
    extract_text_from_web,
    rerank_documents,
    search_multiple_documents,
    split_text,
)


load_dotenv()


# Page settings
st.set_page_config(
    page_title="Smart Research Assistant",
    page_icon="📚",
    layout="wide"
)


# Cache embedding model
@st.cache_resource
def get_embeddings():
    return create_embeddings()


# Cache cross-encoder reranker
@st.cache_resource
def get_reranker():
    return create_reranker()


# Cache language model
@st.cache_resource
def create_language_model():
    return create_llm()


# Generate source ID
def get_source_id(source):
    return hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()


# Generate file ID
def get_file_id(uploaded_file):
    return hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()


# App header
st.title("📚 Smart Research Assistant")

st.markdown(
    "Ask questions about research papers, text files, "
    "Word documents, and web content using a RAG-based AI assistant."
)

st.divider()


# Sidebar inputs
with st.sidebar:

    st.header("📚 Documents")

    uploaded_files = st.file_uploader(
        "Drag & drop your files here",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="Upload PDF, TXT, or DOCX documents."
    )

    st.divider()

    st.subheader("🌐 Add Web Content")

    web_url = st.text_input(
        "Enter a webpage URL",
        placeholder="https://example.com/article"
    )

    st.divider()

    st.caption(
        "Powered by RAG • Hybrid Retrieval • Reranking • LLM"
    )


# Prepare sources
sources_to_process = []


# Add uploaded files
if uploaded_files:

    for uploaded_file in uploaded_files:

        sources_to_process.append({
            "name": uploaded_file.name,
            "type": uploaded_file.type,
            "file": uploaded_file
        })


# Add webpage URL
if web_url.strip():

    sources_to_process.append({
        "name": web_url.strip(),
        "type": "web",
        "url": web_url.strip()
    })


# Process sources
if sources_to_process:

    source_ids = []

    for source in sources_to_process:

        if "file" in source:

            source_ids.append(
                get_file_id(source["file"])
            )

        else:

            source_ids.append(
                get_source_id(source["name"])
            )


    # Reprocess only when sources change
    if st.session_state.get("source_ids") != source_ids:

        # Clear old data
        st.session_state.pop(
            "vector_stores",
            None
        )

        st.session_state.pop(
            "bm25_stores",
            None
        )

        st.session_state.pop(
            "documents",
            None
        )

        vector_stores = []
        bm25_stores = []
        documents = []


        with st.spinner(
            "Processing documents and building retrieval indexes..."
        ):

            try:

                embedding_model = get_embeddings()

                for source in sources_to_process:

                    source_name = source["name"]


                    # Extract source text
                    if source["type"] == "application/pdf":

                        pages = extract_text_from_pdf(
                            source["file"]
                        )

                    elif source["type"] == "text/plain":

                        pages = extract_text_from_txt(
                            source["file"]
                        )

                    elif source["type"] == (
                        "application/vnd.openxmlformats-officedocument."
                        "wordprocessingml.document"
                    ):

                        pages = extract_text_from_docx(
                            source["file"]
                        )

                    else:

                        try:

                            pages = extract_text_from_web(
                                source["url"]
                            )

                        except Exception as e:

                            st.warning(
                                f"Could not read "
                                f"{source_name}: {str(e)}"
                            )

                            continue


                    # Validate extracted content
                    if not pages:

                        st.warning(
                            f"No readable text found in "
                            f"{source_name}."
                        )

                        continue


                    # Create chunks
                    chunks = split_text(pages)

                    if not chunks:

                        st.warning(
                            f"No text chunks were created "
                            f"for {source_name}."
                        )

                        continue


                    # Generate collection ID
                    if "file" in source:

                        collection_id = get_file_id(
                            source["file"]
                        )

                    else:

                        collection_id = get_source_id(
                            source_name
                        )


                    # Create ChromaDB vector store
                    vector_store = create_vector_store(
                        chunks,
                        embedding_model,
                        source_name,
                        collection_id
                    )

                    vector_stores.append(
                        vector_store
                    )


                    # Create BM25 documents and index
                    bm25_documents = create_bm25_documents(
                        chunks,
                        source_name
                    )

                    bm25_index = create_bm25_index(
                        bm25_documents
                    )

                    bm25_stores.append({
                        "index": bm25_index,
                        "documents": bm25_documents
                    })


                    # Store document information
                    documents.append({
                        "name": source_name,
                        "pages": pages,
                        "chunks": chunks,
                        "type": source["type"]
                    })


                # Save processed sources
                if documents:

                    st.session_state.source_ids = (
                        source_ids
                    )

                    st.session_state.vector_stores = (
                        vector_stores
                    )

                    st.session_state.bm25_stores = (
                        bm25_stores
                    )

                    st.session_state.documents = (
                        documents
                    )

                else:

                    st.warning(
                        "No readable content was found."
                    )


            except Exception as e:

                st.error(
                    f"Could not process the content: "
                    f"{str(e)}"
                )


# Clear old documents when nothing is selected
elif "vector_stores" in st.session_state:

    st.session_state.pop(
        "vector_stores",
        None
    )

    st.session_state.pop(
        "bm25_stores",
        None
    )

    st.session_state.pop(
        "documents",
        None
    )

    st.session_state.pop(
        "source_ids",
        None
    )


# Show document information
if "vector_stores" in st.session_state:

    vector_stores = (
        st.session_state.vector_stores
    )

    bm25_stores = (
        st.session_state.bm25_stores
    )

    documents = (
        st.session_state.documents
    )


    # Success message
    st.success(
        f"📚 {len(documents)} source(s) ready"
    )


    # Calculate document statistics
    total_pages = sum(
        len(document["pages"])
        for document in documents
    )

    total_chunks = sum(
        len(document["chunks"])
        for document in documents
    )

    total_characters = sum(
        len(page["text"])
        for document in documents
        for page in document["pages"]
    )


    # Show metrics
    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "📚 Sources",
            len(documents)
        )

    with col2:

        st.metric(
            "📑 Pages",
            total_pages
        )

    with col3:

        st.metric(
            "🧩 Chunks",
            total_chunks
        )


    st.caption(
        f"🔤 {total_characters:,} characters processed"
    )

    st.caption(
        "✅ Dense and BM25 retrieval indexes are ready."
    )

    st.divider()


    # Show uploaded sources
    st.subheader("📚 Sources")

    for document in documents:

        if document["type"] == "web":

            icon = "🌐"

        elif document["type"] == "text/plain":

            icon = "📝"

        elif document["type"] == (
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ):

            icon = "📝"

        else:

            icon = "📄"


        st.write(
            f"{icon} **{document['name']}**"
        )


    st.divider()


    # Question section
    st.subheader("💬 Ask a Question")

    query = st.text_input(
        "Enter your question",
        placeholder="e.g. What are the main findings?"
    )


    if query.strip():

        with st.spinner(
            "Retrieving, reranking, and generating an answer..."
        ):

            try:

                # Hybrid retrieval:
                # ChromaDB semantic retrieval + BM25 keyword retrieval
                candidate_chunks = (
                    search_multiple_documents(
                        vector_stores,
                        bm25_stores,
                        query,
                        top_k=3,
                        candidate_k=8
                    )
                )


                # Cross-encoder reranking
                reranker = get_reranker()

                relevant_chunks = rerank_documents(
                    reranker,
                    query,
                    candidate_chunks,
                    top_k=3
                )


                if not relevant_chunks:

                    st.warning(
                        "No relevant information was found."
                    )

                else:

                    # Generate answer using final Top-3 chunks
                    llm = create_language_model()

                    answer = generate_answer(
                        llm,
                        query,
                        relevant_chunks
                    )


                    # Display answer
                    st.subheader("🤖 Answer")

                    st.info(answer)

                    st.divider()


                    # Show supporting sources
                    st.subheader(
                        "📚 Supporting Sources"
                    )

                    sources = {}

                    for document in relevant_chunks:

                        source_name = (
                            document.metadata.get(
                                "source",
                                "Unknown"
                            )
                        )

                        page_number = (
                            document.metadata.get(
                                "page",
                                "Unknown"
                            )
                        )

                        if source_name not in sources:

                            sources[source_name] = set()

                        sources[source_name].add(
                            page_number
                        )


                    for (
                        source_name,
                        source_pages
                    ) in sources.items():

                        page_list = ", ".join(
                            str(page)
                            for page in sorted(
                                source_pages,
                                key=str
                            )
                        )

                        st.write(
                            f"📄 **{source_name}** — "
                            f"Pages {page_list}"
                        )


                    # Show retrieved chunks
                    with st.expander(
                        "🔍 View Retrieved Chunks"
                    ):

                        for (
                            index,
                            document
                        ) in enumerate(
                            relevant_chunks,
                            start=1
                        ):

                            source_name = (
                                document.metadata.get(
                                    "source",
                                    "Unknown"
                                )
                            )

                            page_number = (
                                document.metadata.get(
                                    "page",
                                    "Unknown"
                                )
                            )

                            st.markdown(
                                f"**Chunk {index} — "
                                f"{source_name} — "
                                f"Page {page_number}**"
                            )

                            st.write(
                                document.page_content
                            )

                            if (
                                index
                                < len(relevant_chunks)
                            ):

                                st.divider()


            except Exception as e:

                st.error(
                    "Something went wrong while "
                    f"answering the question: {str(e)}"
                )


# RAGAS evaluation
st.divider()

st.subheader("📊 RAG Evaluation")

st.caption(
    "RAG pipeline evaluation using Ragas."
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Faithfulness",
        "1.00"
    )


with col2:

    st.metric(
        "Answer Relevancy",
        "0.86"
    )


with col3:

    st.metric(
        "Context Precision",
        "0.90"
    )


with col4:

    st.metric(
        "Context Recall",
        "0.80"
    )


st.caption(
    "Scores obtained from the Ragas evaluation test set."
)


# Show start message
if "vector_stores" not in st.session_state:

    st.info(
        "👈 Upload PDFs/TXT/DOCX files or add a "
        "webpage URL from the sidebar to get started."
    )