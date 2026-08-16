import logging
from typing import List, Dict, Any
from backend.core.config import settings
from backend.retrieval.embeddings import get_embedding_generator
from backend.retrieval.vector_store import NumpyVectorStore
from backend.retrieval.bm25 import BM25Retriever

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    A unified Hybrid Retriever that combines dense vector retrieval and 
    BM25 lexical search using Reciprocal Rank Fusion (RRF).
    
    Provides adjustable weighting for dense and sparse scores, and 
    can load index files persisted locally.
    """
    
    def __init__(self, vector_store: NumpyVectorStore, bm25_retriever: BM25Retriever):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.embedding_gen = get_embedding_generator()

    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Executes a hybrid search querying both dense and sparse indexes, 
        combining candidate ranks using Weighted Reciprocal Rank Fusion (RRF).
        """
        top_k = top_k or settings.RETRIEVAL_TOP_K
        
        # We query the individual retrievers for a larger candidate pool (e.g. 100 docs)
        # to ensure good rank combination depth
        candidate_depth = max(100, top_k * 2)
        
        # 1. Dense Search
        query_emb = self.embedding_gen.embed_query(query)
        dense_results = self.vector_store.search(query_emb, top_k=candidate_depth)
        
        # 2. Sparse Search
        sparse_results = self.bm25_retriever.search(query, top_k=candidate_depth)
        
        # 3. Reciprocal Rank Fusion
        rrf_scores: Dict[str, Dict[str, Any]] = {}
        rrf_k = settings.RRF_K
        
        # Process dense ranks
        for rank, res in enumerate(dense_results):
            chunk_id = res["chunk_id"]
            # rrf_k + rank (1-indexed)
            rrf_score = (1.0 / (rrf_k + (rank + 1))) * settings.DENSE_WEIGHT
            
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {
                    "res": res,
                    "score": 0.0
                }
            rrf_scores[chunk_id]["score"] += rrf_score
            
        # Process sparse ranks
        for rank, res in enumerate(sparse_results):
            chunk_id = res["chunk_id"]
            rrf_score = (1.0 / (rrf_k + (rank + 1))) * settings.SPARSE_WEIGHT
            
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {
                    "res": res,
                    "score": 0.0
                }
            rrf_scores[chunk_id]["score"] += rrf_score
            
        # Sort candidates descending by combined RRF score
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        
        # Slice top_k
        results = []
        for chunk_id, info in sorted_candidates[:top_k]:
            original_res = info["res"]
            results.append({
                "chunk_id": chunk_id,
                "text": original_res["text"],
                "score": info["score"],
                "retrieval_method": "hybrid",
                "metadata": original_res["metadata"]
            })
            
        return results
