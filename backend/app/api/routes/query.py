import asyncio
import json
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from app.core.config import settings
from app.schemas.query import (
    ChunkEvidence,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    VisionFinding,
)
from app.services.generation import NO_EVIDENCE_SENTENCE, generation_service
from app.services.retrieval import vector_store_service
from app.services.vision import vision_service
from app.utils.logging_config import logger

router = APIRouter()

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint reporting vector store, BM25, reranker, YOLO, and LLM status."""
    return HealthResponse(
        status="healthy",
        total_chunks=vector_store_service.total_chunks,
        bm25_loaded=vector_store_service.bm25_model is not None,
        reranker_loaded=vector_store_service.ranker is not None,
        yolo_model_loaded=vision_service.is_loaded,
        llm_model=settings.LLM_MODEL,
    )


@router.post("/query", response_model=QueryResponse)
async def process_query(
    question: str = Form(..., min_length=2, description="Clinical question to answer"),
    chat_history_json: Optional[str] = Form(None, description="JSON-encoded list of recent chat turns"),
    file: Optional[UploadFile] = File(None, description="Optional Brain MRI image file (JPEG/PNG)"),
):
    vision_finding = None
    search_query = question.strip()

    # Parse optional chat history
    chat_history = []
    if chat_history_json:
        try:
            chat_history = json.loads(chat_history_json)
        except Exception:
            logger.warning("Could not parse chat_history_json; proceeding without history.")

    # 1. Process optional MRI scan
    if file and file.filename:
        if file.content_type and file.content_type.lower() not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type '{file.content_type}'. Please upload a JPEG or PNG MRI scan.",
            )
        try:
            image_bytes = await file.read()
            max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
            if len(image_bytes) > max_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Image exceeds maximum allowed size of {settings.MAX_IMAGE_SIZE_MB} MB.",
                )

            pred = await asyncio.to_thread(vision_service.predict_from_bytes, image_bytes)
            vision_finding = VisionFinding(**pred)

            det_cls = vision_finding.detected_class
            logger.info(f"YOLO detection completed: {det_cls} ({vision_finding.confidence*100:.1f}%)")

            lower_q = question.lower()
            generic_triggers = [
                "mri", "scan", "image", "this", "tumor", "finding",
                "diagnosis", "management", "patient", "pathology", "lesion", "treatment",
            ]
            if any(t in lower_q for t in generic_triggers) and det_cls.lower() not in lower_q:
                if det_cls != "notumor":
                    search_query = f"{question} {det_cls}"
                else:
                    search_query = f"{question} normal brain MRI"
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to process uploaded image: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error processing image: {str(e)}",
            )

    # 2. Retrieve & Generate
    try:
        retrieved_chunks = await asyncio.to_thread(
            vector_store_service.retrieve, search_query, settings.TOP_K_DEFAULT
        )
        vf_dict = vision_finding.model_dump() if vision_finding else None

        top_score = 0.0
        if retrieved_chunks:
            top_score = retrieved_chunks[0].get("rerank_score")
            if top_score is None:
                top_score = max(0.0, 1.0 - retrieved_chunks[0].get("distance", 1.0))

        chunks_data = [
            ChunkEvidence(
                source=c["source"],
                page=c["page"],
                text=c["text"],
                hierarchy=c.get("hierarchy", ""),
                rerank_score=c.get("rerank_score"),
                bbox=c.get("bbox"),
            )
            for c in retrieved_chunks
        ]

        # Tier 3: Relevance < 5% -> Pure Web Search (drop irrelevant local chunks)
        if top_score < settings.MIN_RERANK_SCORE or not retrieved_chunks:
            logger.info(f"Query '{question}' relevance ({top_score:.3f}) < 5%: performing web search only.")
            answer, web_sources, sources_text = await asyncio.to_thread(
                generation_service.generate_web_search_answer,
                question,
                vf_dict,
                None,
                chat_history,
            )
            return QueryResponse(
                answer=answer,
                sources=sources_text,
                chunks=[],
                vision_finding=vision_finding,
                expanded_query=search_query,
                is_web_search=True,
                is_hybrid_web=False,
                low_relevance_warning=None,
                web_sources=web_sources,
            )

        # Tier 2: Relevance 5% - 10% -> True Hybrid (pass both local chunks and Web Search to LLM)
        elif top_score < settings.HIGH_RELEVANCE_THRESHOLD:
            logger.info(f"Query '{question}' relevance ({top_score:.3f}) in [5%, 10%): hybrid web + local chunks.")
            answer, web_sources, sources_text = await asyncio.to_thread(
                generation_service.generate_web_search_answer,
                question,
                vf_dict,
                retrieved_chunks,
                chat_history,
            )
            local_source_labels = [f"{c['source']} (Page {c['page']})" for c in retrieved_chunks]
            combined_sources = list(dict.fromkeys(sources_text + local_source_labels))
            warning_msg = (
                f"⚠️ Low Local Relevance Notice ({top_score*100:.1f}%): The closest passages in the indexed medical "
                "literature showed marginal relevance. A hybrid synthesis combining real-time web search and "
                "the closest local textbook excerpts was performed."
            )
            return QueryResponse(
                answer=answer,
                sources=combined_sources,
                chunks=chunks_data,
                vision_finding=vision_finding,
                expanded_query=search_query,
                is_web_search=True,
                is_hybrid_web=True,
                low_relevance_warning=warning_msg,
                web_sources=web_sources,
            )

        # Tier 1: High Relevance >= 10% -> Local Vector Store + BM25
        else:
            logger.info(f"Query '{question}' relevance ({top_score:.3f}) >= 10%: using local database chunks.")
            answer, sources = await asyncio.to_thread(
                generation_service.generate_answer,
                question,
                retrieved_chunks,
                vf_dict,
                chat_history,
            )

            insufficient_patterns = [
                NO_EVIDENCE_SENTENCE.lower(),
                "does not contain sufficient clinical evidence",
                "no mention or description",
                "does not contain sufficient evidence",
            ]
            if any(p in answer.lower() for p in insufficient_patterns):
                logger.info(
                    f"LLM indicated insufficient local evidence for '{question}'. Supplementing with hybrid web search..."
                )
                web_ans, web_sources, sources_text = await asyncio.to_thread(
                    generation_service.generate_web_search_answer,
                    question,
                    vf_dict,
                    retrieved_chunks,
                    chat_history,
                )
                combined_sources = list(dict.fromkeys(sources_text + sources))
                warning_msg = (
                    "⚠️ Partial Local Evidence: Local literature lacked specific details to fully answer the question. "
                    "A real-time web search was performed alongside the local excerpts below."
                )
                return QueryResponse(
                    answer=web_ans,
                    sources=combined_sources,
                    chunks=chunks_data,
                    vision_finding=vision_finding,
                    expanded_query=search_query,
                    is_web_search=True,
                    is_hybrid_web=True,
                    low_relevance_warning=warning_msg,
                    web_sources=web_sources,
                )

            return QueryResponse(
                answer=answer,
                sources=sources,
                chunks=chunks_data,
                vision_finding=vision_finding,
                expanded_query=search_query,
                is_web_search=False,
                is_hybrid_web=False,
                low_relevance_warning=None,
                web_sources=[],
            )
    except Exception as e:
        logger.error(f"Generation pipeline failure: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}",
        )


@router.post("/query/json", response_model=QueryResponse)
async def process_query_json(request: QueryRequest):
    """Programmatic JSON-only text query endpoint."""
    history_json = (
        json.dumps([t.model_dump() for t in request.chat_history])
        if request.chat_history
        else None
    )
    return await process_query(question=request.question, chat_history_json=history_json, file=None)
