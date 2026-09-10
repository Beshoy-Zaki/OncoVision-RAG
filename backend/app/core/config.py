import os
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import List

# Calculate project root (3 levels up from backend/app/core/config.py)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=os.path.join(PROJECT_ROOT, ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    PROJECT_NAME: str = "Multimodal Medical RAG Assistant"
    API_V1_STR: str = "/api/v1"
    
    # LLM Settings
    GEMINI_API_KEY: str = ""
    LLM_PROVIDER: str = "gemini"
    LLM_MODEL: str = "gemini-3.5-flash-lite"
    
    # Vector DB & Embeddings
    CHROMA_PERSIST_DIR: str = os.path.join(PROJECT_ROOT, "data", "vector_store")
    COLLECTION_NAME: str = "neuro_oncology_rag"
    TOP_K_DEFAULT: int = 4
    MAX_RELEVANCE_DISTANCE: float = 0.58
    WEB_SEARCH_MODEL: str = "gemma-4-26b-a4b-it"

    # Hybrid Retrieval & Reranker Settings
    ENABLE_HYBRID_SEARCH: bool = True
    ENABLE_RERANKER: bool = True
    RERANKER_MODEL: str = "ms-marco-MiniLM-L-12-v2"
    RERANK_TOP_K: int = 4
    HYBRID_FETCH_K: int = 20
    MIN_RERANK_SCORE: float = 0.05
    HIGH_RELEVANCE_THRESHOLD: float = 0.10
    BM25_CORPUS_FILE: str = "bm25_corpus.pkl"
    
    # Computer Vision (YOLO)
    YOLO_MODEL_PATH: str = os.path.join(PROJECT_ROOT, "data", "models", "brain_tumor_yolo.onnx")
    CONF_THRESHOLD: float = 0.25
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "*"
    ]

settings = Settings()
