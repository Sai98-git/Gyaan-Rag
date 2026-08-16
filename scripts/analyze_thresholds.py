import sys
import os
import logging
import statistics
import numpy as np

# Ensure UTF-8 output on Windows terminal
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

logging.basicConfig(level=logging.WARNING)

from backend.core.config import settings
settings.CORPUS_MODE = "sample"
settings.MAX_RECORDS = 200

from backend.ingestion.dataset_loader import iterate_records
from backend.retrieval.embeddings import get_embedding_generator
from backend.retrieval.vector_store import NumpyVectorStore

def main():
    print("=== Grounding Threshold Detailed Performance Analysis ===")
    
    strategy = "semantic"
    dense_dir = f"data/indexes/{strategy}/dense"
    
    vector_store = NumpyVectorStore()
    if not vector_store.load(dense_dir):
        print(f"Error: Vector index for {strategy} not found. Run scripts.build_index first!")
        sys.exit(1)
        
    # Map query_id to relevant chunk IDs
    relevance_map = {}
    for chunk in vector_store.chunks_metadata:
        qid = chunk["query_id"]
        is_sel = chunk["metadata"].get("is_selected", 0)
        if is_sel == 1:
            if qid not in relevance_map:
                relevance_map[qid] = set()
            relevance_map[qid].add(chunk["chunk_id"])
            
    # Load records
    try:
        record_gen = iterate_records()
        records = list(record_gen)
    except Exception as e:
        print(f"Error loading records: {e}")
        sys.exit(1)
        
    eval_queries = [r for r in records if r.query_id in relevance_map][:100]
    print(f"Loaded {len(eval_queries)} queries with relevant context.")
    
    embedding_gen = get_embedding_generator()
    
    pos_scores = []
    neg_scores = []
    
    print("Running vector searches and collecting similarities...")
    for idx, r in enumerate(eval_queries):
        query = r.query
        relevant_ids = relevance_map[r.query_id]
        
        query_emb = embedding_gen.embed_query(query)
        dense_res = vector_store.search(query_emb, top_k=10)
        
        for res in dense_res:
            score = res["score"]
            if res["chunk_id"] in relevant_ids:
                pos_scores.append(score)
            else:
                neg_scores.append(score)
                
    total_pos = len(pos_scores)
    total_neg = len(neg_scores)
    
    print(f"\nCollected {total_pos} positive chunks and {total_neg} negative chunks.")
    
    candidate_thresholds = [0.70, 0.75, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90]
    
    print("\n" + "="*85)
    print(f"{'Threshold':<9} | {'TP Rate (Rec)':<13} | {'FP Rate':<9} | {'Precision':<9} | {'F1-Score':<9} | {'TP count':<8} | {'FP count':<8}")
    print("-"*85)
    
    for t in candidate_thresholds:
        tp = sum(1 for s in pos_scores if s >= t)
        fn = sum(1 for s in pos_scores if s < t)
        fp = sum(1 for s in neg_scores if s >= t)
        tn = sum(1 for s in neg_scores if s < t)
        
        tp_rate = tp / total_pos if total_pos > 0 else 0.0
        fp_rate = fp / total_neg if total_neg > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * tp_rate / (precision + tp_rate) if (precision + tp_rate) > 0 else 0.0
        
        print(f"{t:<9.2f} | {tp_rate:<13.4f} | {fp_rate:<9.4f} | {precision:<9.4f} | {f1:<9.4f} | {tp:<8} | {fp:<8}")
        
    print("="*85)

if __name__ == "__main__":
    main()
