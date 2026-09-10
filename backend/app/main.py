from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes.query import router as query_router
from app.services.retrieval import vector_store_service
from app.services.vision import vision_service
from app.services.generation import generation_service
from app.utils.logging_config import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager:
    Loads vector store, YOLO vision model, and LLM connections ONCE at startup.
    """
    logger.info("Starting up Multimodal Medical RAG Backend...")
    try:
        vector_store_service.initialize()
        vision_service.initialize()
        generation_service.initialize()
        logger.info("All backend services initialized successfully.")
    except Exception as e:
        logger.error(f"Error during backend startup initialization: {e}")
        raise e
        
    yield
    
    logger.info("Shutting down Multimodal Medical RAG Backend...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Multimodal Neuro-Oncology RAG Assistant powered by YOLOv8, ChromaDB, and Google GenAI.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routes
app.include_router(query_router, tags=["RAG Query & Health"])

@app.get("/")
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "docs_url": "/docs",
        "health_url": "/health",
        "query_url": "/query"
    }
