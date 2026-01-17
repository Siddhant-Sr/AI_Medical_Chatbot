import logging
import hashlib
from typing import List, Dict

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
)
from app.utils import logger




# ---------------------------
# Initialize Vector Store
# ---------------------------
def get_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    return PineconeVectorStore(
        index=index,
        embedding=embeddings
    )


# ---------------------------
# Retrieval Function
# ---------------------------
def retrieve_medical_context(
    query: str,
    k: int = 3
) -> Dict:
    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    docs = retriever.invoke(query)

    context = "\n\n".join(doc.page_content for doc in docs)
    sources = list(
        set(doc.metadata.get("source", "unknown") for doc in docs)
    )

    logger(
        event_type="retrieval",
        payload={
            "query": query,
            "k": k,
            "num_docs": len(docs),
            "sources": sources
        }
    )

    return {
        "context": context,
        "sources": sources,
        "documents": docs
    }
