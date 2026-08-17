"""
evaluate_retrieval.py: Dataset-Driven Benchmark & Metric Evaluation for Gyaan RAG.

Evaluates MultiStrategyRetriever against ground-truth relevance labels from `ai4bharat/MSMARCO-XI`.
Computes:
- Recall@1, Recall@5, Recall@10
- MRR (Mean Reciprocal Rank)
- NDCG@10
"""

import sys
import os
import math
import time
import logging
from typing import List, Dict, Any

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.WARNING)

from backend.ingestion.dataset_loader import iterate_records
from backend.retrieval.multi_strategy import MultiStrategyRetriever


def compute_dcg(relevances: List[int], k: int = 10) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], 1):
        if rel > 0:
            dcg += rel / math.log2(i + 1)
    return dcg


def evaluate_dataset_retrieval(sample_limit: int = 100):
    print("=" * 85)
    print(f"📊 EVALUATING DATASET RETRIEVAL PERFORMANCE (SAMPLE = {sample_limit} QUERIES)")
    print("=" * 85)

    indexes_base = os.path.join(PROJECT_ROOT, "data", "indexes")
    retriever = MultiStrategyRetriever(indexes_base)
    retriever.load(load_dense=True)

    records = list(iterate_records())
    test_records = records[:sample_limit]

    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    mrr_total = 0.0
    ndcg_total = 0.0

    eval_count = 0
    t0 = time.perf_counter()

    for r in test_records:
        qid = r.query_id
        eng_q = r.Eng_Query
        hi_q = r.query
        
        # Test query in both English and Hindi
        for q_text in [hi_q, eng_q]:
            if not q_text:
                continue

            eval_count += 1
            results = retriever.retrieve(q_text, top_k=10)

            # Check if any retrieved chunk belongs to this query_id
            hit_ranks = []
            relevances = []

            for rank, hit in enumerate(results, 1):
                cid = str(hit.get("chunk_id", ""))
                meta = hit.get("metadata", {})
                hit_qid = str(meta.get("query_id") or cid.split("_")[0])
                if hit_qid == str(qid):
                    hit_ranks.append(rank)
                    relevances.append(1)
                else:
                    relevances.append(0)

            # Recall@K
            if any(rk <= 1 for rk in hit_ranks):
                recall_at_1 += 1
            if any(rk <= 5 for rk in hit_ranks):
                recall_at_5 += 1
            if any(rk <= 10 for rk in hit_ranks):
                recall_at_10 += 1

            # MRR
            if hit_ranks:
                first_rank = hit_ranks[0]
                mrr_total += 1.0 / first_rank

            # NDCG@10
            actual_dcg = compute_dcg(relevances, k=10)
            ideal_relevances = sorted(relevances, reverse=True)
            ideal_dcg = compute_dcg(ideal_relevances, k=10)
            if ideal_dcg > 0:
                ndcg_total += actual_dcg / ideal_dcg
            elif hit_ranks:
                ndcg_total += 1.0

    dt = time.perf_counter() - t0
    r1 = recall_at_1 / eval_count if eval_count else 0.0
    r5 = recall_at_5 / eval_count if eval_count else 0.0
    r10 = recall_at_10 / eval_count if eval_count else 0.0
    mrr = mrr_total / eval_count if eval_count else 0.0
    ndcg = ndcg_total / eval_count if eval_count else 0.0

    print(f"Total Evaluated Queries : {eval_count}")
    print(f"Evaluation Runtime      : {dt:.2f}s ({dt/eval_count*1000:.2f}ms/query)")
    print("-" * 85)
    print(f"Recall@1                : {r1*100:.2f}%")
    print(f"Recall@5                : {r5*100:.2f}%")
    print(f"Recall@10               : {r10*100:.2f}%")
    print(f"MRR                     : {mrr:.4f}")
    print(f"NDCG@10                 : {ndcg:.4f}")
    print("=" * 85)

    return {
        "eval_count": eval_count,
        "recall_at_1": r1,
        "recall_at_5": r5,
        "recall_at_10": r10,
        "mrr": mrr,
        "ndcg_10": ndcg
    }


if __name__ == "__main__":
    evaluate_dataset_retrieval(sample_limit=50)
