from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, description="Clinical question for the RAG assistant")
    chat_history: Optional[List[ChatTurn]] = Field(default_factory=list)


class VisionFinding(BaseModel):
    detected_class: str
    confidence: float
    probabilities: Dict[str, float]
    bbox: Optional[List[int]] = None  # [x1, y1, x2, y2] if detection model or ROI is extracted


class ChunkEvidence(BaseModel):
    source: str
    page: int
    text: str
    hierarchy: Optional[str] = None
    rerank_score: Optional[float] = None
    bbox: Optional[List[float]] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    chunks: List[ChunkEvidence] = Field(default_factory=list)
    vision_finding: Optional[VisionFinding] = None
    expanded_query: Optional[str] = None
    is_web_search: bool = False
    is_hybrid_web: bool = False
    low_relevance_warning: Optional[str] = None
    web_sources: List[Dict[str, str]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    total_chunks: int
    bm25_loaded: bool
    reranker_loaded: bool
    yolo_model_loaded: bool
    llm_model: str
