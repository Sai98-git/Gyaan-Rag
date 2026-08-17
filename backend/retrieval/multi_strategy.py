import os
import logging
from typing import List, Dict, Any, Optional
from backend.retrieval.bm25 import BM25Retriever
from backend.retrieval.vector_store import NumpyVectorStore
from backend.retrieval.normalizer import normalize_query_text

logger = logging.getLogger(__name__)

STRATEGIES = ["semantic", "sliding_window", "passage"]


class MultiStrategyRetriever:
    """
    True Dataset-Grounded Hybrid Retriever:
    Combines Multilingual-E5 Dense Semantic Embeddings + Multi-Strategy BM25
    across all offline chunking strategies (semantic, sliding-window, passage).
    
    Zero hardcoded knowledge, zero manual transliteration maps.
    Uses Multilingual-E5 representation for cross-lingual English/Hindi/Hinglish retrieval.
    """
    
    def __init__(self, index_base_dir: str):
        self.index_base_dir = index_base_dir
        self.bm25_retrievers: Dict[str, BM25Retriever] = {}
        self.vector_stores: Dict[str, NumpyVectorStore] = {}
        self.embedding_generator = None
        self.loaded = False

    def load(self, load_dense: bool = True) -> bool:
        """Loads BM25 and Dense vector indexes across all chunking strategies."""
        bm25_count = 0
        dense_count = 0

        # 1. Load BM25 across strategies
        for strategy in STRATEGIES:
            bm25_dir = os.path.join(self.index_base_dir, strategy, "bm25")
            if os.path.isdir(bm25_dir):
                r = BM25Retriever()
                if r.load(bm25_dir):
                    self.bm25_retrievers[strategy] = r
                    bm25_count += 1

        # 2. Load Dense Vector Stores (Local / Dedicated runtime only, skip on Vercel lambda cold start)
        is_serverless = bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
        if load_dense and not is_serverless:
            for strategy in STRATEGIES:
                dense_dir = os.path.join(self.index_base_dir, strategy, "dense")
                if os.path.isdir(dense_dir):
                    vs = NumpyVectorStore()
                    if vs.load(dense_dir):
                        self.vector_stores[strategy] = vs
                        dense_count += 1

            # Try loading embedding generator for query encoding if dependencies allow
            try:
                from backend.retrieval.embeddings import get_embedding_generator
                self.embedding_generator = get_embedding_generator()
                logger.info("Dense embedding generator attached to MultiStrategyRetriever.")
            except Exception as e:
                logger.warning(f"Dense embedding generator could not be loaded: {e}. Running in pure BM25 multi-strategy mode.")

        self.loaded = (bm25_count > 0 or dense_count > 0)
        logger.info(
            f"MultiStrategyRetriever loaded: BM25 strategies={list(self.bm25_retrievers.keys())}, "
            f"Dense strategies={list(self.vector_stores.keys())}, Total BM25 chunks={self.total_bm25_chunks}"
        )
        return self.loaded

    @property
    def total_chunks(self) -> int:
        return self.total_bm25_chunks

    @property
    def total_bm25_chunks(self) -> int:
        return sum(len(r.chunks) for r in self.bm25_retrievers.values())

    @property
    def total_dense_chunks(self) -> int:
        return sum(len(vs.chunks_metadata) for vs in self.vector_stores.values())

    def search(
        self, 
        query: str, 
        top_k: int = 10, 
        rrf_k: int = 60,
        dense_weight: float = 1.2,
        bm25_weight: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid semantic vector + multi-strategy lexical retrieval:
        1. Normalizes query text (unicode compose, clean whitespace).
        2. Computes multilingual E5 query embedding (with 'query: ' prefix).
        3. Retrieves semantic candidates from Dense Vector Stores.
        4. Retrieves keyword candidates from BM25 indexes across strategies.
        5. Fuses candidate lists using Reciprocal Rank Fusion (RRF).
        6. Returns top_k deduplicated evidence passages with exact scoring diagnostics.
        """
        if not self.loaded:
            return []

        norm_query = normalize_query_text(query)
        if not norm_query:
            return []

        candidates_pool: Dict[str, Dict[str, Any]] = {}
        fused_scores: Dict[str, float] = {}
        dense_scores_map: Dict[str, float] = {}
        bm25_scores_map: Dict[str, float] = {}
        strategy_sources_map: Dict[str, List[str]] = {}

        # ── 1. Dense Semantic Retrieval via Multilingual-E5 ──
        if self.vector_stores and self.embedding_generator:
            try:
                q_emb = self.embedding_generator.embed_query(norm_query)
                for strategy, vs in self.vector_stores.items():
                    dense_hits = vs.search(q_emb, top_k=top_k * 2)
                    for rank, hit in enumerate(dense_hits):
                        cid = hit["chunk_id"]
                        d_score = hit.get("score", 0.0)
                        dense_scores_map[cid] = max(dense_scores_map.get(cid, 0.0), d_score)
                        
                        # Dense RRF contribution
                        dense_rrf = dense_weight / (rrf_k + rank + 1)
                        fused_scores[cid] = fused_scores.get(cid, 0.0) + dense_rrf
                        
                        if cid not in candidates_pool:
                            candidates_pool[cid] = hit
                            strategy_sources_map[cid] = []
                        if f"dense_{strategy}" not in strategy_sources_map[cid]:
                            strategy_sources_map[cid].append(f"dense_{strategy}")
            except Exception as e:
                logger.error(f"[MultiStrategy] Dense search error: {e}")

        # ── 2. Multi-Strategy BM25 Retrieval ──
        strategy_weights = {"semantic": 1.2, "sliding_window": 1.0, "passage": 1.0}
        for strategy, retriever in self.bm25_retrievers.items():
            bm25_hits = retriever.search(norm_query, top_k=top_k * 2)
            strat_w = strategy_weights.get(strategy, 1.0) * bm25_weight
            for rank, hit in enumerate(bm25_hits):
                cid = hit["chunk_id"]
                b_score = hit.get("score", 0.0)
                bm25_scores_map[cid] = max(bm25_scores_map.get(cid, 0.0), b_score)
                
                # BM25 RRF contribution
                bm25_rrf = strat_w / (rrf_k + rank + 1)
                fused_scores[cid] = fused_scores.get(cid, 0.0) + bm25_rrf
                
                if cid not in candidates_pool:
                    candidates_pool[cid] = hit
                    strategy_sources_map[cid] = []
                if f"bm25_{strategy}" not in strategy_sources_map[cid]:
                    strategy_sources_map[cid].append(f"bm25_{strategy}")

        if not fused_scores:
            return []

        # Sort candidates by combined fused RRF score
        sorted_cids = sorted(fused_scores.keys(), key=lambda c: fused_scores[c], reverse=True)[:top_k]
        max_rrf = max(fused_scores.values()) if fused_scores else 1.0

        fused_results = []
        for cid in sorted_cids:
            chunk = candidates_pool[cid]
            norm_score = fused_scores[cid] / max_rrf if max_rrf > 0 else 0.0
            d_sc = dense_scores_map.get(cid, 0.0)
            b_sc = bm25_scores_map.get(cid, 0.0)
            
            fused_results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": round(norm_score, 4),
                "rrf_score": round(fused_scores[cid], 5),
                "dense_score": round(d_sc, 4),
                "bm25_score": round(b_sc, 4),
                "retrieval_method": "hybrid_dense_bm25_rrf",
                "strategy_hits": strategy_sources_map.get(cid, []),
                "metadata": {
                    **chunk.get("metadata", {}),
                    "dense_similarity": round(d_sc, 4),
                    "bm25_relevance": round(b_sc, 4),
                    "strategy_sources": strategy_sources_map.get(cid, [])
                }
            })

        logger.info(
            f"[MultiStrategy] Query='{query}' -> {len(fused_results)} fused chunks. "
            f"Top RRF={fused_results[0]['rrf_score']:.4f}, Top Dense={fused_results[0]['dense_score']:.4f}, "
            f"Top BM25={fused_results[0]['bm25_score']:.4f}"
        )
        return fused_results
