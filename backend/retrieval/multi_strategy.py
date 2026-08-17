import os
import logging
from typing import List, Dict, Any, Optional
from backend.retrieval.bm25 import BM25Retriever
from backend.retrieval.normalizer import expand_indic_query

logger = logging.getLogger(__name__)

STRATEGIES = ["semantic", "sliding_window", "passage"]


class MultiStrategyRetriever:
    """
    Orchestrates multi-strategy retrieval across precomputed offline indexes:
    - Semantic sentence/paragraph chunks
    - Sliding-window chunks with overlap
    - Natural passage/document chunks
    
    Applies multilingual Indic-English query expansion and fuses results using
    Reciprocal Rank Fusion (RRF) with strategy-weighted scoring.
    """
    
    def __init__(self, index_base_dir: str):
        self.index_base_dir = index_base_dir
        self.retrievers: Dict[str, BM25Retriever] = {}
        self.loaded = False

    def load(self) -> bool:
        """Loads BM25 indexes for all precomputed chunking strategies."""
        loaded_count = 0
        for strategy in STRATEGIES:
            bm25_dir = os.path.join(self.index_base_dir, strategy, "bm25")
            if os.path.isdir(bm25_dir):
                r = BM25Retriever()
                if r.load(bm25_dir):
                    self.retrievers[strategy] = r
                    loaded_count += 1
                    logger.info(f"Loaded multi-strategy BM25 index for '{strategy}' ({len(r.chunks)} chunks).")
                    
        self.loaded = loaded_count > 0
        return self.loaded

    @property
    def total_chunks(self) -> int:
        return sum(len(r.chunks) for r in self.retrievers.values())

    def search(self, query: str, top_k: int = 10, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """
        Executes multi-strategy retrieval:
        1. Expands query with multilingual cross-lingual synonyms.
        2. Retrieves candidate passages from each chunking strategy.
        3. Fuses rankings using Reciprocal Rank Fusion (RRF).
        4. Returns top_k deduplicated, score-normalized evidence passages.
        """
        if not self.loaded or not self.retrievers:
            return []

        expanded_query = expand_indic_query(query)
        logger.debug(f"[MultiStrategy] Query='{query}' -> Expanded='{expanded_query}'")

        # Gather ranked candidates from each strategy
        strategy_candidates: Dict[str, List[Dict[str, Any]]] = {}
        for strategy, retriever in self.retrievers.items():
            # Query with expanded tokens, falling back to raw query if needed
            res = retriever.search(expanded_query, top_k=top_k * 2)
            if not res:
                res = retriever.search(query, top_k=top_k * 2)
            strategy_candidates[strategy] = res

        # Reciprocal Rank Fusion (RRF)
        # Weights: semantic (1.2), sliding_window (1.0), passage (1.0)
        weights = {"semantic": 1.2, "sliding_window": 1.0, "passage": 1.0}
        fused_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}
        strategy_hits: Dict[str, List[str]] = {}

        for strategy, candidates in strategy_candidates.items():
            w = weights.get(strategy, 1.0)
            for rank, item in enumerate(candidates):
                cid = item["chunk_id"]
                rrf_score = w / (rrf_k + rank + 1)
                fused_scores[cid] = fused_scores.get(cid, 0.0) + rrf_score
                
                if cid not in chunk_map:
                    chunk_map[cid] = item
                    strategy_hits[cid] = []
                strategy_hits[cid].append(strategy)

        if not fused_scores:
            return []

        # Sort by fused RRF score
        sorted_cids = sorted(fused_scores.keys(), key=lambda c: fused_scores[c], reverse=True)[:top_k]

        # Normalize top score
        max_rrf = max(fused_scores.values()) if fused_scores else 1.0
        
        fused_results = []
        for cid in sorted_cids:
            chunk = chunk_map[cid]
            norm_score = fused_scores[cid] / max_rrf if max_rrf > 0 else 0.0
            
            fused_results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": round(norm_score, 4),
                "raw_rrf_score": round(fused_scores[cid], 5),
                "retrieval_method": "multi_strategy_bm25_rrf",
                "strategy_hits": strategy_hits.get(cid, []),
                "metadata": {
                    **chunk.get("metadata", {}),
                    "chunk_strategy_sources": strategy_hits.get(cid, [])
                }
            })

        logger.info(
            f"[MultiStrategy] Retrieved {len(fused_results)} fused chunks. "
            f"Top score={fused_results[0]['score']:.4f}, Strategies active={list(self.retrievers.keys())}"
        )
        return fused_results
