import sys
import os
import time
import logging
import statistics
import numpy as np
from typing import List, Dict, Any, Set

# Ensure UTF-8 output on Windows terminal
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("evaluate_retrieval")

from backend.core.config import settings
# Force CORPUS_MODE to sample for consistent benchmarking
settings.CORPUS_MODE = "sample"
settings.MAX_RECORDS = 200

from backend.ingestion.dataset_loader import iterate_records
from backend.retrieval.embeddings import get_embedding_generator
from backend.retrieval.vector_store import NumpyVectorStore
from backend.retrieval.bm25 import BM25Retriever
from backend.retrieval.hybrid import HybridRetriever

def get_dir_size(directory: str) -> float:
    """Returns size of a directory in MB."""
    total_size = 0
    if not os.path.exists(directory):
        return 0.0
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

def evaluate_configuration(strategy: str, records: List[Any], n_queries: int = 200) -> Dict[str, Any]:
    print(f"\nEvaluating configuration: Strategy={strategy} ...")
    
    dense_dir = f"data/indexes/{strategy}/dense"
    bm25_dir = f"data/indexes/{strategy}/bm25"
    
    # 1. Load Vector Store and BM25 index
    vector_store = NumpyVectorStore()
    if not vector_store.load(dense_dir):
        print(f"Error: Vector store index not found for {strategy}. Build index first!")
        return {}
        
    bm25_retriever = BM25Retriever()
    if not bm25_retriever.load(bm25_dir):
        print(f"Error: BM25 index not found for {strategy}. Build index first!")
        return {}
        
    hybrid_retriever = HybridRetriever(vector_store, bm25_retriever)
    
    # 2. Get list of all chunks in index to find relevant chunk mappings
    all_chunks = vector_store.chunks_metadata
    
    # Map query_id to relevant chunk IDs
    relevance_map: Dict[int, Set[str]] = {}
    for chunk in all_chunks:
        qid = chunk["query_id"]
        is_sel = chunk["metadata"].get("is_selected", 0)
        if is_sel == 1:
            if qid not in relevance_map:
                relevance_map[qid] = set()
            relevance_map[qid].add(chunk["chunk_id"])
            
    # 3. Filter test queries (records where we have at least one relevant chunk in the index)
    eval_queries = []
    for r in records:
        if r.query_id in relevance_map:
            eval_queries.append(r)
            if len(eval_queries) >= n_queries:
                break
                
    print(f"Selected {len(eval_queries)} queries for evaluation.")
    
    methods = ["dense", "bm25", "hybrid"]
    metrics = {m: {"r1": [], "r5": [], "r10": [], "mrr10": [], "latencies": []} for m in methods}
    
    # Initialize embedding generator singleton once to avoid model reload time
    embedding_gen = get_embedding_generator()
    
    # Warmup
    embedding_gen.embed_query("कॉर्पोरेशन क्या है?")
    
    for idx, r in enumerate(eval_queries):
        query = r.query
        relevant_ids = relevance_map[r.query_id]
        
        # A. Dense Search
        t0 = time.perf_counter()
        query_emb = embedding_gen.embed_query(query)
        dense_res = vector_store.search(query_emb, top_k=10)
        metrics["dense"]["latencies"].append(time.perf_counter() - t0)
        
        # B. BM25 Search
        t0 = time.perf_counter()
        bm25_res = bm25_retriever.search(query, top_k=10)
        metrics["bm25"]["latencies"].append(time.perf_counter() - t0)
        
        # C. Hybrid Search
        t0 = time.perf_counter()
        hybrid_res = hybrid_retriever.search(query, top_k=10)
        metrics["hybrid"]["latencies"].append(time.perf_counter() - t0)
        
        # Calculate metrics for each method
        results_map = {
            "dense": dense_res,
            "bm25": bm25_res,
            "hybrid": hybrid_res
        }
        
        for m, res_list in results_map.items():
            res_ids = [res["chunk_id"] for res in res_list]
            
            # Recall@K
            metrics[m]["r1"].append(1.0 if any(rid in relevant_ids for rid in res_ids[:1]) else 0.0)
            metrics[m]["r5"].append(1.0 if any(rid in relevant_ids for rid in res_ids[:5]) else 0.0)
            metrics[m]["r10"].append(1.0 if any(rid in relevant_ids for rid in res_ids[:10]) else 0.0)
            
            # MRR@10
            mrr = 0.0
            for rank, rid in enumerate(res_ids[:10]):
                if rid in relevant_ids:
                    mrr = 1.0 / (rank + 1)
                    break
            metrics[m]["mrr10"].append(mrr)
            
    # Calculate averages
    summary = {}
    for m in methods:
        lats = [l * 1000 for l in metrics[m]["latencies"]]  # convert to ms
        summary[m] = {
            "r1": sum(metrics[m]["r1"]) / len(eval_queries),
            "r5": sum(metrics[m]["r5"]) / len(eval_queries),
            "r10": sum(metrics[m]["r10"]) / len(eval_queries),
            "mrr10": sum(metrics[m]["mrr10"]) / len(eval_queries),
            "mean_latency": sum(lats) / len(lats),
            "median_latency": statistics.median(lats),
            "p95_latency": np.percentile(lats, 95)
        }
        
    index_size = get_dir_size(dense_dir) + get_dir_size(bm25_dir)
    summary["index_size_mb"] = index_size
    summary["total_chunks"] = len(all_chunks)
    
    return summary

def main():
    print("=== Starting Retrieval Evaluation Subsystem ===")
    
    # 1. Load records once
    print("Loading test dataset records...")
    try:
        record_gen = iterate_records()
        records = list(record_gen)
    except Exception as e:
        print(f"Error loading records: {e}")
        sys.exit(1)
        
    print(f"Loaded {len(records)} records.")
    
    strategies = ["passage", "sliding_window", "semantic"]
    all_results = {}
    
    for strategy in strategies:
        res = evaluate_configuration(strategy, records, n_queries=200)
        if res:
            all_results[strategy] = res
            
    # Output markdown formatted comparisons
    print("\n" + "="*95)
    print(f"{'Strategy':<14} | {'Method':<7} | {'R@1':<5} | {'R@5':<5} | {'R@10':<5} | {'MRR@10':<6} | {'Idx (MB)':<8} | {'Lat (ms)':<8}")
    print("-"*95)
    for strategy in strategies:
        if strategy not in all_results:
            continue
        idx_size = all_results[strategy]["index_size_mb"]
        for m in ["dense", "bm25", "hybrid"]:
            stats = all_results[strategy][m]
            print(f"{strategy:<14} | {m:<7} | {stats['r1']:<5.3f} | {stats['r5']:<5.3f} | {stats['r10']:<5.3f} | {stats['mrr10']:<6.3f} | {idx_size:<8.2f} | {stats['mean_latency']:<8.2f}")
        print("-"*95)
    print("="*95)

if __name__ == "__main__":
    main()
