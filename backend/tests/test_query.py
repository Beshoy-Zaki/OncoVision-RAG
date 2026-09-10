import os
import glob
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="module")
def client():
    """Context manager fixture that triggers FastAPI lifespan loading."""
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    """Verify GET /health returns 200 and reports loaded components."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["total_chunks"] > 0
    assert data["yolo_model_loaded"] is True
    assert "llm_model" in data

def test_query_happy_path(client):
    """Verify POST /query/json returns a grounded answer with cited sources."""
    payload = {"question": "What is the Stupp protocol for Glioblastoma?"}
    response = client.post("/query/json", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 10
    assert "sources" in data
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) > 0

def test_query_invalid_input_422(client):
    """Verify sending invalid/missing input returns HTTP 422 Unprocessable Entity."""
    # Empty payload
    response = client.post("/query/json", json={})
    assert response.status_code == 422

    # Empty string question (min_length=2)
    response2 = client.post("/query/json", json={"question": "a"})
    assert response2.status_code == 422

def test_query_multimodal_with_image(client):
    """Verify POST /query with multipart form data (image + question)."""
    img_candidates = glob.glob("data/images/meningioma/*.jpg") or glob.glob("../data/images/meningioma/*.jpg")
    assert len(img_candidates) > 0, "No sample test image found for multimodal test"
    
    img_path = img_candidates[0]
    with open(img_path, "rb") as f:
        img_bytes = f.read()

    response = client.post(
        "/query",
        data={"question": "What pathology is suspected on this MRI scan?"},
        files={"file": ("mri_scan.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "vision_finding" in data
    assert data["vision_finding"] is not None
    assert "detected_class" in data["vision_finding"]
    assert "confidence" in data["vision_finding"]
    assert len(data["sources"]) > 0

def test_clean_chunk_text():
    """Verify clean_chunk_text fixes line-break hyphens and chopped word boundaries."""
    from app.services.retrieval import vector_store_service
    
    raw = ", in 20-30% of patients, pathological contrast enhancement due to irradia- tion and necrosis"
    cleaned = vector_store_service.clean_chunk_text(raw)
    assert cleaned.startswith("In 20-30%")
    assert "irradiation" in cleaned
    assert "irradia- tion" not in cleaned
    
    chopped = "s, or earlier upon clinical deterioration"
    cleaned_chopped = vector_store_service.clean_chunk_text(chopped)
    assert cleaned_chopped.startswith("Or earlier")

def test_query_web_search_fallback_on_unrelated_query(client):
    """Verify unrelated questions suppress database chunks and trigger web search."""
    payload = {"question": "What is the primary treatment for celiac disease in pediatric patients?"}
    response = client.post("/query/json", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_web_search"] is True
    # Crucial requirement: database chunks MUST be empty when database lacks the answer
    assert len(data.get("chunks", [])) == 0
    assert len(data.get("answer", "")) > 10
    assert len(data.get("web_sources", [])) > 0 or len(data.get("sources", [])) > 0

def test_hybrid_search_and_flashrank_reranker(client):
    """Verify hybrid search and FlashRank reranker prioritize gold-standard passages with high confidence."""
    from app.services.retrieval import vector_store_service
    
    query = "What are the clinical symptoms and locations of cerebral gliomas?"
    chunks = vector_store_service.retrieve(query, top_k=4)
    
    assert len(chunks) > 0
    # Top chunk must have a high rerank score
    top_chunk = chunks[0]
    assert top_chunk.get("rerank_score") is not None
    assert top_chunk["rerank_score"] > 0.85
    # Gold standard: Bookshelf_NBK469998.pdf Page 284 or Osborns Brain Astrocytomas
    assert "Bookshelf_NBK469998.pdf" in top_chunk["source"] or "Osborns_Brain.pdf" in top_chunk["source"]
    # No diagram / pie chart pages in retrieved chunks
    for c in chunks:
        assert "(16-9)" not in c["text"]
        assert "pie chart" not in c["text"].lower()


