from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, description="The user question to ask the RAG assistant")

class VisionFinding(BaseModel):
    detected_class: str
    confidence: float
    probabilities: Dict[str, float]

class ChunkEvidence(BaseModel):
    source: str
    page: int
    text: str
    hierarchy: Optional[str] = None
    rerank_score: Optional[float] = None
    bbox: Optional[List[float]] = None  # [x0, y0, x1, y1] from ingestion-time PyMuPDF block extraction

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    chunks: Optional[List[ChunkEvidence]] = []
    vision_finding: Optional[VisionFinding] = None
    is_web_search: bool = False
    is_hybrid_web: bool = False
    low_relevance_warning: Optional[str] = None
    web_sources: Optional[List[Dict[str, str]]] = []

class HealthResponse(BaseModel):
    status: str
    total_chunks: int
    yolo_model_loaded: bool
    llm_model: str
