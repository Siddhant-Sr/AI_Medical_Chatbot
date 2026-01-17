import logging
import hashlib
from typing import List

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec

from app.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ------------------------------------------------------------------
# Load PDFs
# ------------------------------------------------------------------
def load_pdf_files(data_dir: str) -> List[Document]:
    loader = DirectoryLoader(
        data_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    return loader.load()

# ------------------------------------------------------------------
# Clean + filter documents
# ------------------------------------------------------------------
def filter_documents(
    documents: List[Document],
    min_length: int = 1000,
) -> List[Document]:
    cleaned_docs: List[Document] = []

    for doc in documents:
        if len(doc.page_content.strip()) < min_length:
            continue

        cleaned_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                },
            )
        )

    return cleaned_docs

# ------------------------------------------------------------------
# Chunking
# ------------------------------------------------------------------
def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)

# ------------------------------------------------------------------
# Pinecone setup
# ------------------------------------------------------------------
def get_or_create_pinecone_index(embeddings: HuggingFaceEmbeddings):
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if not pc.has_index(PINECONE_INDEX_NAME):
        logging.info("Creating Pinecone index...")

        dimension = len(embeddings.embed_query("dimension_check"))

        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

    return pc.Index(PINECONE_INDEX_NAME)

# ------------------------------------------------------------------
# Helper: deterministic chunk IDs
# ------------------------------------------------------------------
def generate_chunk_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# ------------------------------------------------------------------
# Upload embeddings (batched + idempotent)
# ------------------------------------------------------------------
def upload_embeddings(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    batch_size: int = 64,
):
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
    )

    total = len(chunks)
    logging.info(f"Uploading {total} chunks in batches...")

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]

        ids = [
            generate_chunk_id(doc.page_content)
            for doc in batch
        ]

        vectorstore.add_documents(
            documents=batch,
            ids=ids,
        )

        logging.info(
            f"Uploaded batch {i // batch_size + 1} "
            f"({min(i + batch_size, total)}/{total})"
        )

# ------------------------------------------------------------------
# Main ingestion pipeline
# ------------------------------------------------------------------
def ingest_docs(data_dir: str):
    try:
        logging.info("Starting ingestion pipeline")

        raw_docs = load_pdf_files(data_dir)
        logging.info(f"Loaded {len(raw_docs)} raw documents")

        cleaned_docs = filter_documents(raw_docs)
        logging.info(f"{len(cleaned_docs)} documents after cleaning")

        chunks = split_documents(cleaned_docs)
        logging.info(f"Created {len(chunks)} chunks")

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME
        )

        get_or_create_pinecone_index(embeddings)

        upload_embeddings(
            chunks=chunks,
            embeddings=embeddings,
        )

        logging.info("Ingestion completed successfully")

    except Exception as e:
        logging.exception("Ingestion failed")
        raise
