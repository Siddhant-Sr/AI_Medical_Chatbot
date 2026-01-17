import os
from dotenv import load_dotenv

load_dotenv()

# -------- API KEYS --------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not set")

# -------- RAG CONFIG --------
PINECONE_INDEX_NAME = "medical-chatbot"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
STT_MODEL = "whisper-large-v3-turbo"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# Constants file name

AUDIO_PATH= "patient_voice.wav"
IMAGE_PATH="med.jpg"
