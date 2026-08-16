import sys
import os
import time
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
    print("=== Starting Retrieval Score Calibration Experiment ===")
    
    # 1. Load vector store for the selected semantic strategy
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
                
    print("\n--- Calibration Results ---")
    if pos_scores:
        print(f"Positive Chunks (Relevant - is_selected=1) [Total: {len(pos_scores)}]:")
        print(f"  Min Score   : {min(pos_scores):.4f}")
        print(f"  Max Score   : {max(pos_scores):.4f}")
        print(f"  Mean Score  : {sum(pos_scores)/len(pos_scores):.4f}")
        print(f"  Median Score: {statistics.median(pos_scores):.4f}")
        print(f"  p5 Score    : {np.percentile(pos_scores, 5):.4f}")
    else:
        print("No positive chunks retrieved.")
        
    if neg_scores:
        print(f"Negative Chunks (Irrelevant - is_selected=0) [Total: {len(neg_scores)}]:")
        print(f"  Min Score   : {min(neg_scores):.4f}")
        print(f"  Max Score   : {max(neg_scores):.4f}")
        print(f"  Mean Score  : {sum(neg_scores)/len(neg_scores):.4f}")
        print(f"  Median Score: {statistics.median(neg_scores):.4f}")
        print(f"  p95 Score   : {np.percentile(neg_scores, 95):.4f}")
    else:
        print("No negative chunks retrieved.")
        
    # Analyze optimal threshold (separating positive from negative chunks)
    # Typically, we want to set threshold to p5 or p10 of positive scores, 
    # ensuring we don't reject too many valid matches while blocking low score negatives.
    if pos_scores and neg_scores:
        p5_pos = np.percentile(pos_scores, 5)
        p95_neg = np.percentile(neg_scores, 95)
        
        recommended_threshold = round((p5_pos + p95_neg) / 2.0, 2)
        # Safe bounds: usually around 0.70 for E5 models
        if recommended_threshold < 0.60:
            recommended_threshold = 0.65
        elif recommended_threshold > 0.80:
            recommended_threshold = 0.75
            
        print("\n--- Recommendation ---")
        print(f"  Calculated p5 (Positives): {p5_pos:.4f}")
        print(f"  Calculated p95 (Negatives): {p95_neg:.4f}")
        print(f"  Recommended Grounding Threshold (MIN_RETRIEVAL_SCORE): {recommended_threshold:.2f}")
        print(f"  (This threshold ensures that at least 95% of relevant matches are accepted,")
        print(f"  while minimizing hallucinations on irrelevant/low-confidence matches.)")

if __name__ == "__main__":
    main()
