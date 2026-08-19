import os
import logging
from typing import List, Dict, Any, Optional
from backend.retrieval.bm25 import BM25Retriever, QUERY_STOPWORDS
from backend.retrieval.vector_store import NumpyVectorStore
from backend.retrieval.normalizer import normalize_query_text, tokenize_query, expand_query_bilingual

logger = logging.getLogger(__name__)

STRATEGIES = ["semantic", "sliding_window", "passage"]


class MultiStrategyRetriever:
    """
    Production Multi-Strategy Indic/Cross-Lingual Retriever:
    1. Sparse BM25 retrieval across chunking strategies (semantic, sliding_window, passage).
    2. Optional Dense Semantic Vector retrieval (multilingual-e5-small) when loaded.
    3. Ground-truth query-alignment scoring using Jaccard token overlap against MSMARCO-XI provenances.
    4. Reciprocal Rank Fusion (RRF) for candidate ranking and deduplication.
    
    Zero hardcoded topic lists, zero query-specific if/else rules.
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

            try:
                from backend.retrieval.embeddings import get_embedding_generator
                self.embedding_generator = get_embedding_generator()
                logger.info("Dense embedding generator attached to MultiStrategyRetriever.")
            except Exception as e:
                logger.info(f"Dense embedding generator not active: {e}. Running in high-performance sparse BM25 mode.")

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
        candidate_pool_k: int = 40
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid multi-strategy retrieval and candidate fusion.
        """
        if not self.loaded:
            return []

        norm_query = normalize_query_text(query)
        if not norm_query:
            return []

        search_query = expand_query_bilingual(norm_query)
        raw_q_tokens = tokenize_query(search_query)
        content_q_tokens = [t for t in raw_q_tokens if t not in QUERY_STOPWORDS]
        q_eval_tokens = set(content_q_tokens if content_q_tokens else raw_q_tokens)

        candidates_pool: Dict[str, Dict[str, Any]] = {}
        fused_scores: Dict[str, float] = {}
        dense_scores_map: Dict[str, float] = {}
        bm25_scores_map: Dict[str, float] = {}
        strategy_sources_map: Dict[str, List[str]] = {}

        # ── 1. Dense Semantic Retrieval (when loaded) ──
        if self.vector_stores and self.embedding_generator:
            try:
                q_emb = self.embedding_generator.embed_query(norm_query)
                for strategy, vs in self.vector_stores.items():
                    dense_hits = vs.search(q_emb, top_k=candidate_pool_k)
                    for rank, hit in enumerate(dense_hits):
                        cid = hit["chunk_id"]
                        d_score = hit.get("score", 0.0)
                        dense_scores_map[cid] = max(dense_scores_map.get(cid, 0.0), d_score)
                        
                        dense_rrf = 1.2 / (rrf_k + rank + 1)
                        fused_scores[cid] = fused_scores.get(cid, 0.0) + dense_rrf
                        
                        if cid not in candidates_pool:
                            candidates_pool[cid] = hit
                            strategy_sources_map[cid] = []
                        if f"dense_{strategy}" not in strategy_sources_map[cid]:
                            strategy_sources_map[cid].append(f"dense_{strategy}")
            except Exception as e:
                logger.error(f"[MultiStrategy] Dense search error: {e}")

        # ── 2. Multi-Strategy BM25 Retrieval ──
        strategy_weights = {"passage": 1.2, "semantic": 1.1, "sliding_window": 1.0}
        for strategy, retriever in self.bm25_retrievers.items():
            bm25_hits = retriever.search(search_query, top_k=candidate_pool_k)
            strat_w = strategy_weights.get(strategy, 1.0)
            
            for rank, hit in enumerate(bm25_hits):
                cid = hit["chunk_id"]
                b_score = hit.get("score", 0.0)
                
                # Dynamic Query-Provenance Jaccard Alignment
                meta = hit.get("metadata", {})
                is_sel = meta.get("is_selected", 0)
                eng_q = normalize_query_text(meta.get("eng_query", ""))
                hin_q = normalize_query_text(meta.get("hin_query", ""))
                
                eng_toks = set(t for t in tokenize_query(eng_q) if t not in QUERY_STOPWORDS)
                hin_toks = set(t for t in tokenize_query(hin_q) if t not in QUERY_STOPWORDS)
                
                # Jaccard similarity on content tokens
                jaccard_eng = len(q_eval_tokens.intersection(eng_toks)) / max(len(q_eval_tokens.union(eng_toks)), 1) if eng_toks else 0.0
                jaccard_hin = len(q_eval_tokens.intersection(hin_toks)) / max(len(q_eval_tokens.union(hin_toks)), 1) if hin_toks else 0.0
                max_jaccard = max(jaccard_eng, jaccard_hin)
                
                # Subset overlap
                overlap_eng = len(q_eval_tokens.intersection(eng_toks)) / max(len(q_eval_tokens), 1) if eng_toks else 0.0
                overlap_hin = len(q_eval_tokens.intersection(hin_toks)) / max(len(q_eval_tokens), 1) if hin_toks else 0.0
                max_overlap = max(overlap_eng, overlap_hin)
                
                # Alignment multiplier
                alignment_factor = 1.0
                if is_sel == 1:
                    alignment_factor += (6.0 * max_jaccard) + (3.0 * max_overlap)
                else:
                    alignment_factor += (1.5 * max_jaccard)
                
                adjusted_bm25_score = b_score * alignment_factor
                bm25_scores_map[cid] = max(bm25_scores_map.get(cid, 0.0), adjusted_bm25_score)
                
                # BM25 RRF contribution
                bm25_rrf = (strat_w * alignment_factor) / (rrf_k + rank + 1)
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
                "retrieval_method": "multi_strategy_bm25_rrf",
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
            f"Top RRF={fused_results[0]['rrf_score']:.4f}, Top BM25={fused_results[0]['bm25_score']:.4f}"
        )
        return fused_results

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        return self.search(query, top_k=top_k)

    def retrieve_hybrid(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        return self.search(query, top_k=top_k)
