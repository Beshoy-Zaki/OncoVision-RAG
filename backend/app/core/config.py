import os
from pathlib import Path
from typing import List
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


def resolve_project_root() -> str:
    """Resolve project root reliably across local dev, pytest, and Docker containers."""
    current = Path(__file__).resolve().parent
    for parent in [current.parents[1], current.parents[2], current.parents[0]]:
        if (parent / "data").exists() or (parent / ".env").exists():
            return str(parent)
    return str(current.parents[2])


PROJECT_ROOT = resolve_project_root()


class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=os.path.join(PROJECT_ROOT, ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PROJECT_NAME: str = "OncoVision-RAG: Multimodal Neuro-Oncology Assistant"
    API_V1_STR: str = "/api/v1"

    # LLM Settings
    GEMINI_API_KEY: str = ""
    LLM_PROVIDER: str = "gemini"
    LLM_MODEL: str = "gemini-2.5-flash"
    WEB_SEARCH_MODEL: str = "gemini-2.5-flash"
    LLM_TEMPERATURE: float = 0.1

    # Vector DB & Embeddings
    CHROMA_PERSIST_DIR: str = os.path.join(PROJECT_ROOT, "data", "vector_store")
    COLLECTION_NAME: str = "neuro_oncology_rag"
    TOP_K_DEFAULT: int = 4
    MAX_RELEVANCE_DISTANCE: float = 0.58

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
    MAX_IMAGE_SIZE_MB: int = 15

    # CORS (Avoid combining wildcard '*' with allow_credentials=True)
    CORS_ORIGINS: List[str] = [
        "http://localhost",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ]


settings = Settings()
