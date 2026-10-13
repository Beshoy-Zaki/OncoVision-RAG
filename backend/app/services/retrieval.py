import json
import os
import pickle
import re
import chromadb
import numpy as np
from chromadb.utils import embedding_functions
from app.core.config import settings
from app.utils.logging_config import logger

TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9_\-]{2,}\b")
HYPHEN_WRAP_PATTERN = re.compile(r"(\b[a-zA-Z]{2,})-\s+([a-zA-Z]{2,}\b)")
LEADING_PUNCT_PATTERN = re.compile(r"^[\s,;:\.\-\)\]\}]+")

VALID_SHORT_TOKENS = {
    "a", "an", "in", "on", "at", "to", "by", "of", "or", "as", "is", "it",
    "if", "we", "the", "for", "not", "and", "but", "no", "so", "up", "vs",
    "mri", "ct", "pet", "dna", "rna", "who", "cns", "idh", "gy", "mg", "ml",
    "cm", "mm", "iv", "ii", "iii", "vi",
}


class VectorStoreService:
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_fn = None
        self.bm25_model = None
        self.bm25_chunks = []
        self.ranker = None

    def initialize(self):
        persist_dir = settings.CHROMA_PERSIST_DIR
        if not os.path.exists(persist_dir):
            alt_dir = os.path.abspath(os.path.join("..", persist_dir))
            if os.path.exists(alt_dir):
                persist_dir = alt_dir

        os.makedirs(persist_dir, exist_ok=True)
        logger.info(f"Loading ChromaDB persistent vector store from: {persist_dir}")
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        self.collection = self.client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"Vector store initialized successfully. Total chunks: {self.collection.count():,}"
        )

        # Load BM25 Lexical Index
        bm25_path = os.path.join(persist_dir, settings.BM25_CORPUS_FILE)
        if os.path.exists(bm25_path):
            try:
                logger.info(f"Loading BM25 lexical corpus from: {bm25_path}")
                with open(bm25_path, "rb") as f:
                    data = pickle.load(f)
                    self.bm25_model = data.get("bm25")
                    self.bm25_chunks = data.get("chunks", [])
                logger.info(f"BM25 index loaded with {len(self.bm25_chunks):,} documents.")
            except Exception as e:
                logger.warning(f"Failed to load BM25 corpus: {e}. Falling back to dense search.")
                self.bm25_model = None
        else:
            logger.warning(f"BM25 corpus not found at {bm25_path}. Running dense search only.")

        # Initialize FlashRank Cross-Encoder Reranker
        if settings.ENABLE_RERANKER:
            try:
                from flashrank import Ranker
                logger.info(f"Initializing FlashRank Reranker ({settings.RERANKER_MODEL})...")
                self.ranker = Ranker(model_name=settings.RERANKER_MODEL)
                logger.info("FlashRank Reranker initialized successfully.")
            except Exception as e:
                logger.warning(f"Failed to initialize FlashRank: {e}. Running without reranker.")
                self.ranker = None

    @staticmethod
    def _is_reference_or_boilerplate(text: str) -> bool:
        """Filters out bibliography, references, and author contact blocks."""
        lower_t = text.lower()
        if lower_t.count("doi.org") + lower_t.count("http://dx.doi") >= 1:
            return True
        if "author for correspondence:" in lower_t or "email: " in lower_t:
            return True
        if "references\n" in lower_t or "references \n" in lower_t:
            return True
        if "pmid:" in lower_t and ("et al." in lower_t or "vol." in lower_t):
            return True
        return False

    @staticmethod
    def _extract_clean_body(text: str) -> str:
        """Strips any [Document: ... | Page: ...] metadata header from the passage body."""
        if text.startswith("[Document:") and "]" in text:
            return text.split("]", 1)[-1].strip()
        return text.strip()

    @staticmethod
    def clean_chunk_text(text: str) -> str:
        """Fixes broken hyphenations and trims fragmented boundary words without deleting clinical units."""
        prefix = ""
        body = text
        if body.startswith("[Document:") and "]" in body:
            parts = body.split("]", 1)
            prefix = parts[0] + "]\n"
            body = parts[1].strip()

        # 1. Re-join words broken across line wraps
        body = HYPHEN_WRAP_PATTERN.sub(r"\1\2", body)

        # 2. Strip leading orphan punctuation
        body = LEADING_PUNCT_PATTERN.sub("", body)

        # 3. Clean leading incomplete word fragment (preserve uppercase, digits, and valid short clinical terms)
        first_space = body.find(" ")
        if 0 < first_space <= 4:
            first_word = body[:first_space].strip()
            if (
                first_word.lower() not in VALID_SHORT_TOKENS
                and not first_word[0].isupper()
                and not any(ch.isdigit() for ch in first_word)
            ):
                body = body[first_space + 1 :].lstrip()

        # 4. Only strip trailing dangling hyphens (never strip valid Roman numerals or units like 'I', 'II', 'Gy')
        if body.endswith("-"):
            body = re.sub(r"\s+\w+-$", "", body).rstrip("-")

        # 5. Capitalize the first letter for clean presentation
        if body and body[0].islower():
            body = body[0].upper() + body[1:]

        return (prefix + body).strip()

    def is_relevant_match(self, chunks: list, max_distance: float | None = None) -> bool:
        threshold = max_distance if max_distance is not None else settings.MAX_RELEVANCE_DISTANCE
        if not chunks:
            return False
        top_chunk = chunks[0]
        if top_chunk.get("rerank_score") is not None:
            return top_chunk["rerank_score"] >= settings.MIN_RERANK_SCORE
        return top_chunk.get("distance", 1.0) <= threshold

    def retrieve(
        self, query: str, top_k: int | None = None, filter_irrelevant: bool = False
    ) -> list:
        if self.collection is None:
            raise RuntimeError("VectorStoreService is not initialized.")

        total_docs = self.collection.count()
        if total_docs == 0:
            return []

        k = top_k or settings.TOP_K_DEFAULT
        fetch_limit = min(max(k * 4, settings.HYBRID_FETCH_K), total_docs)

        # Stage 1: Dense Vector Retrieval (ChromaDB)
        dense_results = self.collection.query(query_texts=[query], n_results=fetch_limit)
        dense_candidates = {}
        if dense_results and dense_results.get("documents") and dense_results["documents"][0]:
            c_ids = dense_results["ids"][0]
            c_docs = dense_results["documents"][0]
            c_metas = dense_results["metadatas"][0]
            c_dists = dense_results["distances"][0]
            for r_idx, (cid, doc, meta, dist) in enumerate(
                zip(c_ids, c_docs, c_metas, c_dists), 1
            ):
                bbox_val = None
                bbox_json = meta.get("bbox_json", "")
                if bbox_json:
                    try:
                        bbox_val = json.loads(bbox_json)
                    except (json.JSONDecodeError, TypeError):
                        pass
                dense_candidates[cid] = {
                    "id": cid,
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page", 1),
                    "hierarchy": meta.get("hierarchy", ""),
                    "bbox": bbox_val,
                    "distance": float(dist),
                    "dense_rank": r_idx,
                }

        # Stage 2: Lexical Retrieval (BM25)
        bm25_candidates = {}
        if settings.ENABLE_HYBRID_SEARCH and self.bm25_model and self.bm25_chunks:
            query_tokens = TOKEN_PATTERN.findall(query.lower())
            if query_tokens:
                bm25_scores = self.bm25_model.get_scores(query_tokens)
                top_bm25_indices = np.argsort(bm25_scores)[-fetch_limit:][::-1]
                for r_idx, idx in enumerate(top_bm25_indices, 1):
                    score = float(bm25_scores[idx])
                    if score <= 0.0:
                        break
                    chunk = self.bm25_chunks[idx]
                    cid = chunk["id"]
                    bm25_candidates[cid] = {
                        "id": cid,
                        "text": chunk["text"],
                        "source": chunk["source"],
                        "page": chunk["page"],
                        "hierarchy": chunk.get("hierarchy", ""),
                        "bbox": chunk.get("bbox"),
                        "bm25_score": score,
                        "bm25_rank": r_idx,
                    }

        # Stage 3: Reciprocal Rank Fusion (RRF, k=60)
        all_candidate_ids = set(dense_candidates.keys()) | set(bm25_candidates.keys())
        rrf_scores = {}
        for cid in all_candidate_ids:
            score = 0.0
            if cid in dense_candidates:
                score += 1.0 / (60.0 + dense_candidates[cid]["dense_rank"])
            if cid in bm25_candidates:
                score += 1.0 / (60.0 + bm25_candidates[cid]["bm25_rank"])
            rrf_scores[cid] = score

        sorted_cids = sorted(all_candidate_ids, key=lambda cid: rrf_scores[cid], reverse=True)[
            :fetch_limit
        ]

        merged_candidates = []
        for cid in sorted_cids:
            if cid in dense_candidates:
                c_item = dense_candidates[cid]
            else:
                c_item = bm25_candidates[cid]
                c_item["distance"] = 0.50

            if self._is_reference_or_boilerplate(c_item["text"]):
                continue
            merged_candidates.append(c_item)

        if not merged_candidates:
            return []

        # Stage 4: Cross-Encoder Re-Ranking (FlashRank)
        if settings.ENABLE_RERANKER and self.ranker and merged_candidates:
            from flashrank import RerankRequest

            passages = [
                {"id": c["id"], "text": self._extract_clean_body(c["text"])}
                for c in merged_candidates
            ]
            try:
                reranked_results = self.ranker.rerank(
                    RerankRequest(query=query, passages=passages)
                )
                score_map = {r["id"]: float(r["score"]) for r in reranked_results}

                for c in merged_candidates:
                    c["rerank_score"] = score_map.get(c["id"], 0.0)

                merged_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

                above_threshold = [
                    c for c in merged_candidates if c["rerank_score"] >= settings.MIN_RERANK_SCORE
                ]
                if above_threshold:
                    merged_candidates = above_threshold
                elif filter_irrelevant:
                    return []
                else:
                    # Keep top candidates sorted so the router can inspect top_score and route to Tier 3
                    merged_candidates = merged_candidates[:k]
            except Exception as e:
                logger.warning(f"Error during FlashRank reranking: {e}. Falling back to RRF.")

        final_chunks = [
            {
                "text": self.clean_chunk_text(c["text"]),
                "source": c["source"],
                "page": c["page"],
                "hierarchy": c.get("hierarchy", ""),
                "bbox": c.get("bbox"),
                "distance": c.get("distance", 0.5),
                "rerank_score": c.get("rerank_score", None),
            }
            for c in merged_candidates[:k]
        ]

        if filter_irrelevant and not self.is_relevant_match(final_chunks):
            logger.info("Retrieved chunks suppressed: top chunk below relevance threshold.")
            return []

        return final_chunks

    @property
    def total_chunks(self) -> int:
        return self.collection.count() if self.collection else 0


vector_store_service = VectorStoreService()
