import hashlib

import requests
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from docx import Document


# Create embedding model
def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):
        page_text = page.extract_text()

        if page_text:
            pages.append({
                "text": page_text,
                "page": page_number
            })

    return pages


# Extract text from TXT
def extract_text_from_txt(txt_file):
    text = txt_file.getvalue().decode(
        "utf-8",
        errors="ignore"
    )

    if not text.strip():
        return []

    return [{
        "text": text,
        "page": 1
    }]

# Extract text from DOCX
def extract_text_from_docx(docx_file):
    document = Document(docx_file)

    text = "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )

    if not text.strip():
        return []

    return [{
        "text": text,
        "page": 1
    }]


# Extract text from webpage
def extract_text_from_web(url):
    response = requests.get(
        url,
        timeout=15,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    for element in soup(
        ["script", "style", "noscript"]
    ):
        element.decompose()

    text = soup.get_text(
        separator=" ",
        strip=True
    )

    if not text:
        return []

    return [{
        "text": text,
        "page": 1
    }]


# Split text into chunks
def split_text(pages):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = []

    for page in pages:

        page_chunks = text_splitter.split_text(
            page["text"]
        )

        for chunk in page_chunks:

            chunks.append({
                "text": chunk,
                "page": page["page"]
            })

    return chunks


# Create a unique collection ID
def create_collection_id(source_name):
    return hashlib.sha256(
        source_name.encode("utf-8")
    ).hexdigest()


# Create Chroma vector store
def create_vector_store(
    chunks,
    embedding_model,
    source_name,
    collection_id
):
    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "page": chunk["page"],
            "source": source_name
        }
        for chunk in chunks
    ]

    return Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas,
        collection_name=f"research_{collection_id}"
    )


# Search across multiple documents
def search_multiple_documents(
    vector_stores,
    query,
    top_k=3
):
    all_results = []

    for vector_store in vector_stores:

        results = (
            vector_store
            .similarity_search_with_score(
                query,
                k=top_k
            )
        )

        all_results.extend(results)

    all_results.sort(
        key=lambda x: x[1]
    )

    return [
        document
        for document, score in all_results[:top_k]
    ]