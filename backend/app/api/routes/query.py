import asyncio
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status
from typing import Optional
from app.schemas.query import QueryRequest, QueryResponse, HealthResponse, VisionFinding, ChunkEvidence
from app.services.retrieval import vector_store_service
from app.services.vision import vision_service
from app.services.generation import generation_service, NO_EVIDENCE_SENTENCE
from app.core.config import settings
from app.utils.logging_config import logger

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint reporting vector store, YOLO, and LLM status."""
    return HealthResponse(
        status="healthy",
        total_chunks=vector_store_service.total_chunks,
        yolo_model_loaded=vision_service.is_loaded,
        llm_model=settings.LLM_MODEL
    )

@router.post("/query", response_model=QueryResponse)
async def process_query(
    question: str = Form(..., min_length=2, description="Clinical question to answer"),
    file: Optional[UploadFile] = File(None, description="Optional Brain MRI image file (JPEG/PNG)")
):
    """
    Multimodal RAG query endpoint:
    - Accepts question and optional MRI image scan (multipart/form-data)
    - Runs YOLO inference if image is uploaded
    - Expands vector retrieval query with visual findings
    - Generates cited, grounded medical answer
    - Returns exact chunk evidence for visual bounding box display
    """
    vision_finding = None
    search_query = question
    
    if file and file.filename:
        try:
            image_bytes = await file.read()
            pred = await asyncio.to_thread(vision_service.predict_from_bytes, image_bytes)
            vision_finding = VisionFinding(**pred)
            
            det_cls = vision_finding.detected_class
            logger.info(f"YOLO detection completed: {det_cls} ({vision_finding.confidence*100:.1f}%)")
            
            # If the user question refers to the scan/image/patient, augment query with detected tumor class
            lower_q = question.lower()
            generic_triggers = ["mri", "scan", "image", "this", "tumor", "finding", "diagnosis", "management", "patient", "pathology"]
            if any(t in lower_q for t in generic_triggers):
                if det_cls != "notumor":
                    search_query = f"{question} {det_cls}"
                else:
                    search_query = f"{question} normal brain"
            else:
                # For specific domain questions (e.g. Simpson grade, prolactinoma), keep query targeted
                search_query = question
        except Exception as e:
            logger.error(f"Failed to process uploaded image: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error processing image: {str(e)}"
            )

    try:
        retrieved_chunks = await asyncio.to_thread(vector_store_service.retrieve, search_query, settings.TOP_K_DEFAULT)
        vf_dict = vision_finding.model_dump() if vision_finding else None
        
        # Determine top relevance score
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
                rerank_score=c.get("rerank_score", None),
                bbox=c.get("bbox", None)
            )
            for c in retrieved_chunks
        ]

        # -------------------------------------------------------------
        # Tier 3: Relevance < 5% (< 0.05) -> Only Web Search (drop chunks)
        # -------------------------------------------------------------
        if top_score < settings.MIN_RERANK_SCORE or not retrieved_chunks:
            logger.info(f"Query '{question}' relevance ({top_score:.3f}) < 5%: performing web search only.")
            answer, web_sources, sources_text = await asyncio.to_thread(
                generation_service.generate_web_search_answer,
                question,
                vf_dict
            )
            return QueryResponse(
                answer=answer,
                sources=sources_text,
                chunks=[],
                vision_finding=vision_finding,
                is_web_search=True,
                is_hybrid_web=False,
                low_relevance_warning=None,
                web_sources=web_sources
            )

        # -------------------------------------------------------------
        # Tier 2: Relevance between 5% and 10% (0.05 <= top_score < 0.10)
        # -> Hybrid: Retrieve low relevance chunks AND perform grounded web search
        # -------------------------------------------------------------
        elif top_score < settings.HIGH_RELEVANCE_THRESHOLD:
            logger.info(f"Query '{question}' relevance ({top_score:.3f}) between 5% and 10%: performing hybrid web search + chunk retrieval.")
            answer, web_sources, sources_text = await asyncio.to_thread(
                generation_service.generate_web_search_answer,
                question,
                vf_dict
            )
            combined_sources = sources_text + [f"{c['source']} (Page {c['page']})" for c in retrieved_chunks]
            warning_msg = (
                f"⚠️ Low Local Relevance Notice ({top_score*100:.1f}%): The closest passages in the indexed medical literature "
                "showed marginal relevance. A real-time web search was conducted to provide a complete grounded answer, "
                "and the low-relevance local document excerpts are also displayed below for your clinical reference."
            )
            return QueryResponse(
                answer=answer,
                sources=combined_sources,
                chunks=chunks_data,
                vision_finding=vision_finding,
                is_web_search=True,
                is_hybrid_web=True,
                low_relevance_warning=warning_msg,
                web_sources=web_sources
            )

        # -------------------------------------------------------------
        # Tier 1: High Relevance >= 10% (top_score >= 0.10) -> Local Database
        # -------------------------------------------------------------
        else:
            logger.info(f"Query '{question}' relevance ({top_score:.3f}) >= 10%: using local database chunks.")
            answer, sources = await asyncio.to_thread(
                generation_service.generate_answer,
                question,
                retrieved_chunks,
                vf_dict
            )

            # Check if LLM indicates the local documentation lacked sufficient clinical evidence
            insufficient_patterns = [
                NO_EVIDENCE_SENTENCE.lower(),
                "does not contain sufficient clinical evidence",
                "no mention or description",
                "does not contain sufficient evidence"
            ]
            if any(p in answer.lower() for p in insufficient_patterns):
                logger.info(f"LLM indicated insufficient local evidence for '{question}'. Supplementing with web search...")
                web_ans, web_sources, sources_text = await asyncio.to_thread(
                    generation_service.generate_web_search_answer,
                    question,
                    vf_dict
                )
                combined_sources = sources_text + [f"{c['source']} (Page {c['page']})" for c in retrieved_chunks]
                warning_msg = (
                    "⚠️ Partial Local Evidence: Local literature lacked specific details to fully answer the clinical question. "
                    "A real-time web search was performed to supplement the response, while the closest local excerpts are provided below."
                )
                return QueryResponse(
                    answer=web_ans,
                    sources=combined_sources,
                    chunks=chunks_data,
                    vision_finding=vision_finding,
                    is_web_search=True,
                    is_hybrid_web=True,
                    low_relevance_warning=warning_msg,
                    web_sources=web_sources
                )

            return QueryResponse(
                answer=answer,
                sources=sources,
                chunks=chunks_data,
                vision_finding=vision_finding,
                is_web_search=False,
                is_hybrid_web=False,
                low_relevance_warning=None,
                web_sources=[]
            )
    except Exception as e:
        logger.error(f"Generation pipeline failure: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}"
        )

@router.post("/query/json", response_model=QueryResponse)
async def process_query_json(request: QueryRequest):
    """Programmatic JSON-only text query endpoint."""
    return await process_query(question=request.question, file=None)
