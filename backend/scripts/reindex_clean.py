import os
import sys
import glob
import time
import re
import json
import pickle
import pymupdf
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DOCS_DIR = os.path.join(PROJECT_ROOT, "data", "raw_docs")
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "data", "vector_store")
COLLECTION_NAME = "neuro_oncology_rag"
BM25_FILE = os.path.join(VECTOR_STORE_DIR, "bm25_corpus.pkl")

def get_page_hierarchy_map(doc) -> dict:
    """Builds a mapping from page number (1-indexed) to hierarchical breadcrumb string using PDF TOC."""
    toc = doc.get_toc()
    if not toc:
        return {}
        
    page_map = {}
    current_levels = {}
    
    # Sort TOC by page number
    sorted_toc = sorted(toc, key=lambda x: (x[2], x[0]))
    toc_idx = 0
    num_toc = len(sorted_toc)
    
    for p in range(1, len(doc) + 1):
        while toc_idx < num_toc and sorted_toc[toc_idx][2] <= p:
            level, title, _ = sorted_toc[toc_idx]
            clean_title = re.sub(r'\s+', ' ', title).strip()
            current_levels[level] = clean_title
            # Clear deeper levels when a higher level is updated
            for k in list(current_levels.keys()):
                if k > level:
                    del current_levels[k]
            toc_idx += 1
            
        if current_levels:
            # Join up to level 3 (e.g. Section > Chapter > Subheading)
            active = [current_levels[k] for k in sorted(current_levels.keys()) if k <= 3]
            page_map[p] = " > ".join(active)
            
    return page_map

def is_pure_reference_page(text: str) -> bool:
    """Detects pages containing only bibliographies, references, or publication notices."""
    lower_t = text.lower()
    first_200 = lower_t[:200]
    if "references\n" in first_200 or "bibliography\n" in first_200:
        ref_items = len(re.findall(r'^\s*\d+[\.\\)]\s+[A-Z]', text, re.M))
        if ref_items >= 4:
            return True
            
    doi_count = lower_t.count("doi.org") + lower_t.count("http://dx.doi")
    if doi_count >= 4:
        return True
        
    return False

def clean_block_text(raw: str) -> str:
    """Clean text from a single block while preserving its searchability against the page."""
    # 1. Rejoin hyphenated words across line wraps
    text = re.sub(r'(\b[a-zA-Z]{2,})-\s*\n\s*([a-zA-Z]{2,}\b)', r'\1\2', raw)
    text = re.sub(r'(\b[a-zA-Z]{2,})-\s+([a-zA-Z]{2,}\b)', r'\1\2', text)
    
    # 2. Clean running headers/footers
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        l_str = line.strip()
        if re.match(r'^(Chapter \d+\s+\d+\s+|[A-Z][a-z]+ \d{2,4}\s+|[A-Z][\w\s\-]+ et al\.\s+\d+\s+Introduction|\d{1,4}\s*$)', l_str):
            continue
        cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)
    
    # 3. Normalize whitespace (but keep newlines for paragraph structure)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def is_diagram_block(block_text: str) -> bool:
    """Identifies text blocks that are figure captions, diagram labels, or pie chart annotations."""
    cleaned = block_text.strip().lower()
    if len(cleaned) < 15:
        return True  # Very short blocks are typically labels/captions
    
    # Diagram/figure caption patterns
    diag_terms = ["pie chart", "diagram shows relative", "schematic illustration", 
                  "relative frequency", "graphic abstract"]
    if any(term in cleaned for term in diag_terms) and len(cleaned) < 300:
        return True
    
    return False

def extract_text_blocks_with_bbox(page) -> list:
    """Extract text blocks from a page using PyMuPDF's layout-aware dict extraction.
    
    Returns a list of dicts, each with:
      - "text": the cleaned block text
      - "bbox": [x0, y0, x1, y1] coordinates (rounded to 1 decimal)
    
    Image blocks (type == 1) and diagram captions are filtered out at block level.
    """
    page_dict = page.get_text("dict")
    text_blocks = []
    
    for block in page_dict.get("blocks", []):
        # Skip image blocks entirely
        if block.get("type", 0) != 0:
            continue
        
        # Extract all text from the block's lines/spans
        block_text_parts = []
        for line in block.get("lines", []):
            line_text_parts = []
            for span in line.get("spans", []):
                span_text = span.get("text", "")
                if span_text.strip():
                    line_text_parts.append(span_text)
            if line_text_parts:
                block_text_parts.append("".join(line_text_parts))
        
        block_text = "\n".join(block_text_parts).strip()
        if not block_text or len(block_text) < 5:
            continue
        
        # Filter out diagram/caption blocks
        if is_diagram_block(block_text):
            continue
        
        bbox = block.get("bbox", [0, 0, 0, 0])
        text_blocks.append({
            "text": block_text,
            "bbox": [round(bbox[0], 1), round(bbox[1], 1), round(bbox[2], 1), round(bbox[3], 1)]
        })
    
    return text_blocks

def merge_blocks_into_chunks(text_blocks: list, chunk_size: int = 800, chunk_overlap: int = 120) -> list:
    """Merge consecutive text blocks into chunks of approximately chunk_size characters.
    
    Each chunk carries the union bounding box of all blocks that contributed to it.
    """
    if not text_blocks:
        return []
    
    chunks = []
    current_text_parts = []
    current_char_count = 0
    # Track the union bbox as [x0, y0, x1, y1]
    current_bbox = None
    
    def union_bbox(a, b):
        """Compute the union bounding box of two [x0,y0,x1,y1] rects."""
        return [
            min(a[0], b[0]),  # x0
            min(a[1], b[1]),  # y0
            max(a[2], b[2]),  # x1
            max(a[3], b[3])   # y1
        ]
    
    def finalize_chunk():
        """Flush current accumulated blocks into a chunk."""
        if not current_text_parts:
            return None
        text = "\n".join(current_text_parts).strip()
        if len(text) < 50:
            return None
        return {
            "text": text,
            "bbox": [round(v, 1) for v in current_bbox] if current_bbox else None
        }
    
    for b_idx, block in enumerate(text_blocks):
        block_text = block["text"]
        block_len = len(block_text)

        if current_char_count > 0 and current_char_count + block_len > chunk_size:
            chunk = finalize_chunk()
            if chunk:
                chunks.append(chunk)

            # Overlap using exact slice index b_idx (fixes list.index() duplicate block bug)
            overlap_text_parts = []
            overlap_chars = 0
            overlap_bbox = None
            for prev_block in reversed(text_blocks[:b_idx]):
                if overlap_chars + len(prev_block["text"]) > chunk_overlap:
                    break
                overlap_text_parts.insert(0, prev_block["text"])
                overlap_chars += len(prev_block["text"])
                overlap_bbox = (
                    prev_block["bbox"][:]
                    if overlap_bbox is None
                    else union_bbox(overlap_bbox, prev_block["bbox"])
                )

            current_text_parts = overlap_text_parts
            current_char_count = overlap_chars
            current_bbox = overlap_bbox

        current_text_parts.append(block_text)
        current_char_count += block_len
        current_bbox = (
            block["bbox"][:]
            if current_bbox is None
            else union_bbox(current_bbox, block["bbox"])
        )
    
    # Flush the last chunk
    chunk = finalize_chunk()
    if chunk:
        chunks.append(chunk)
    
    return chunks

def tokenize_for_bm25(text: str):
    """Simple regex tokenizer for BM25 lexical index."""
    return re.findall(r'\b[a-zA-Z0-9_\-]{2,}\b', text.lower())

def main():
    print(f"================================================================")
    print(f"Starting Production Re-Indexing: Block-Aware Bbox + Hybrid BM25")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"================================================================")
    
    pdf_files = sorted(glob.glob(os.path.join(RAW_DOCS_DIR, "*.pdf")))
    if not pdf_files:
        print(f"ERROR: No PDFs found in {RAW_DOCS_DIR}")
        return
        
    t_start = time.time()
    all_chunks = []
    tokenized_corpus = []
    
    for pdf_path in pdf_files:
        fname = os.path.basename(pdf_path)
        print(f"\nProcessing: {fname}...")
        doc = pymupdf.open(pdf_path)
        hierarchy_map = get_page_hierarchy_map(doc)
        if hierarchy_map:
            print(f"  Loaded Table of Contents hierarchy ({len(hierarchy_map)} pages mapped)")
        else:
            print(f"  No embedded TOC found; relying on heading analysis")
            
        doc_chunk_count = 0
        skipped_ref_pages = 0
        skipped_image_blocks = 0
        
        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            page = doc[page_idx]
            
            # Quick pre-check with flat text for pure reference pages
            raw_flat = page.get_text("text") or ""
            if is_pure_reference_page(raw_flat):
                skipped_ref_pages += 1
                continue
            
            # Extract text blocks with layout awareness (image blocks filtered at source)
            text_blocks = extract_text_blocks_with_bbox(page)
            if not text_blocks:
                continue
            
            # Clean each block's text
            for tb in text_blocks:
                tb["text"] = clean_block_text(tb["text"])
            
            # Remove blocks that are now too short after cleaning
            text_blocks = [tb for tb in text_blocks if len(tb["text"].strip()) >= 10]
            if not text_blocks:
                continue
            
            # Merge blocks into chunks with bbox
            page_hierarchy = hierarchy_map.get(page_num, "")
            page_chunks = merge_blocks_into_chunks(text_blocks, chunk_size=800, chunk_overlap=120)
            
            for c_idx, ch in enumerate(page_chunks):
                if page_hierarchy:
                    enriched_text = f"[Document: {fname} | Section: {page_hierarchy} | Page: {page_num}]\n{ch['text']}"
                else:
                    enriched_text = f"[Document: {fname} | Page: {page_num}]\n{ch['text']}"
                    
                chunk_id = f"{fname}_p{page_num}_c{c_idx}"
                
                # Store bbox as a JSON-serializable list in metadata
                # ChromaDB metadata values must be str, int, float, or bool — store bbox as JSON string
                bbox_val = ch["bbox"]
                meta = {
                    "source": fname,
                    "page": page_num,
                    "chunk_index": c_idx,
                    "hierarchy": page_hierarchy,
                    "bbox_json": json.dumps(bbox_val) if bbox_val else ""
                }
                
                all_chunks.append({
                    "id": chunk_id,
                    "text": enriched_text,
                    "clean_text": ch["text"],
                    "bbox": bbox_val,
                    "metadata": meta
                })
                
                # Tokenize for BM25 (exclude Document/Page prefixes to avoid skewing IDF)
                tokenized_corpus.append(tokenize_for_bm25(f"{page_hierarchy} {ch['text']}"))
                doc_chunk_count += 1
                
        print(f"  Completed {fname}: {len(doc)} pages -> {doc_chunk_count} clean chunks (filtered {skipped_ref_pages} ref pages)")
        
    print(f"\n================================================================")
    print(f"Block-aware extraction completed in {time.time() - t_start:.2f}s!")
    print(f"Total Enriched Chunks: {len(all_chunks):,}")
    print(f"================================================================")
    
    # Verify bbox coverage
    bbox_count = sum(1 for c in all_chunks if c["bbox"])
    print(f"Chunks with native bounding boxes: {bbox_count:,} / {len(all_chunks):,} ({bbox_count/max(len(all_chunks),1)*100:.1f}%)")
    
    # 1. Build and persist BM25 Index
    print(f"\nBuilding BM25 lexical index with {len(tokenized_corpus):,} documents...")
    t_bm25 = time.time()
    bm25_model = BM25Okapi(tokenized_corpus)
    
    bm25_payload = {
        "bm25": bm25_model,
        "chunks": [
            {
                "id": c["id"],
                "text": c["text"],
                "clean_text": c["clean_text"],
                "source": c["metadata"]["source"],
                "page": c["metadata"]["page"],
                "hierarchy": c["metadata"]["hierarchy"],
                "bbox": c["bbox"]
            }
            for c in all_chunks
        ]
    }
    
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25_payload, f)
    print(f"BM25 index persisted to {BM25_FILE} in {time.time() - t_bm25:.2f}s!")
    
    # 2. Re-create ChromaDB Vector Store
    print(f"\nConnecting to ChromaDB at: {VECTOR_STORE_DIR}")
    chroma_client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"Deleted old ChromaDB collection '{COLLECTION_NAME}'")
    except Exception:
        pass
        
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    print(f"Embedding and storing {len(all_chunks):,} chunks in ChromaDB...")
    t_embed = time.time()
    batch_size = 500
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        collection.add(
            ids=[item["id"] for item in batch],
            documents=[item["text"] for item in batch],
            metadatas=[item["metadata"] for item in batch]
        )
        print(f"  Indexed {min(i+batch_size, len(all_chunks)):,} / {len(all_chunks):,} chunks...")
        
    print(f"\nChromaDB indexing completed in {time.time() - t_embed:.2f}s!")
    print(f"Final Count in ChromaDB: {collection.count():,}")
    print(f"Final Count in BM25: {len(bm25_payload['chunks']):,}")
    print(f"All indexes successfully built and persisted!")

if __name__ == "__main__":
    main()
