import sys
import os
import time
import json
import logging
from typing import List, Dict, Any
import numpy as np

# Ensure UTF-8 output on Windows terminal
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.core.config import settings
from backend.api.app import _execute_rag_pipeline, init_rag_resources
from backend.voice.cleaner import normalize_voice_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("benchmark")

# Benchmark Test Query Set covering Hindi, English, Hinglish, and out-of-domain
BENCHMARK_QUERIES = [
    ("hi_in_domain_1", "कॉर्पोरेशन क्या है?"),
    ("hi_in_domain_2", "व्यापार क्या होता है?"),
    ("hi_in_domain_3", "बैंक का कार्य क्या है?"),
    ("en_in_domain_1", "what is a corporation"),
    ("en_in_domain_2", "how does a partnership business operate"),
    ("en_in_domain_3", "what is the purpose of an organization"),
    ("hinglish_1", "corporation kya hota hai?"),
    ("hinglish_2", "business ke types kya hain?"),
    ("out_of_domain_1", "what is the capital of Mars?"),
    ("out_of_domain_2", "quantum gravity temporal displacement field theory"),
    ("hi_in_domain_4", "कंपनी की परिभाषा क्या है?"),
    ("en_in_domain_4", "definition of a joint stock company"),
    ("hi_in_domain_5", "निगम के प्रकार क्या हैं?"),
    ("en_in_domain_5", "what are the key characteristics of corporate structure"),
    ("out_of_domain_3", "xyzzy frobnicator banana recipe"),
]

def calculate_percentiles(values: List[float]) -> Dict[str, float]:
    """Calculates P50, P70, P90, P100, mean, min, max for a list of latency numbers."""
    if not values:
        return {"p50": 0.0, "p70": 0.0, "p90": 0.0, "p100": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
    
    arr = np.array(values)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p70": float(np.percentile(arr, 70)),
        "p90": float(np.percentile(arr, 90)),
        "p100": float(np.percentile(arr, 100)),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr))
    }

def run_benchmark():
    logger.info("=" * 70)
    logger.info("⚡ GYAAN RAG — LATENCY & PERFORMANCE BENCHMARK HARNESS ⚡")
    logger.info("=" * 70)
    
    init_rag_resources()
    logger.info(f"Chunking Strategy : '{settings.CHUNK_STRATEGY}'")
    logger.info(f"Generation Provider: '{settings.GENERATION_PROVIDER}'")
    logger.info(f"Embedding Model    : '{settings.EMBEDDING_MODEL_NAME}'")
    logger.info(f"Top-K Chunks       : {settings.RETRIEVAL_TOP_K}")
    logger.info(f"Total Test Queries : {len(BENCHMARK_QUERIES)}")
    logger.info("-" * 70)

    retrieval_latencies: List[float] = []
    generation_latencies: List[float] = []
    total_rag_latencies: List[float] = []
    cleaner_latencies: List[float] = []

    # Warmup run to avoid initialization cold-start bias
    logger.info("Executing warmup query...")
    _execute_rag_pipeline("warmup query")
    logger.info("Warmup complete. Commencing measurement...\n")

    for idx, (qid, query_text) in enumerate(BENCHMARK_QUERIES, 1):
        t_clean_0 = time.perf_counter()
        clean_q = normalize_voice_query(query_text)
        t_clean = (time.perf_counter() - t_clean_0) * 1000
        cleaner_latencies.append(t_clean)

        t_total_0 = time.perf_counter()
        result = _execute_rag_pipeline(clean_q)
        t_total = (time.perf_counter() - t_total_0) * 1000

        ret_ms = result["retrieval_latency"]
        gen_ms = result["generation_latency"]

        retrieval_latencies.append(ret_ms)
        generation_latencies.append(gen_ms)
        total_rag_latencies.append(t_total)

        status_str = "ABSTAINED" if result["guard_triggered"] else "GROUNDED"
        logger.info(
            f"[{idx:02d}/{len(BENCHMARK_QUERIES):02d}] {qid:<18} | "
            f"Ret={ret_ms:6.2f}ms | Gen={gen_ms:6.2f}ms | Total={t_total:6.2f}ms | "
            f"Status={status_str:<9} | Chunks={result['chunks_count']}"
        )

    # Compute Statistics
    ret_stats = calculate_percentiles(retrieval_latencies)
    gen_stats = calculate_percentiles(generation_latencies)
    tot_stats = calculate_percentiles(total_rag_latencies)
    clean_stats = calculate_percentiles(cleaner_latencies)

    print("\n" + "=" * 70)
    print("📊 BENCHMARK SUMMARY RESULTS")
    print("=" * 70)
    print(f"{'Metric':<24} | {'P50 (ms)':<10} | {'P70 (ms)':<10} | {'P100 (ms)':<10} | {'Mean (ms)':<10} | {'Min (ms)':<10}")
    print("-" * 70)
    print(f"{'1. Query Normalization':<24} | {clean_stats['p50']:<10.2f} | {clean_stats['p70']:<10.2f} | {clean_stats['p100']:<10.2f} | {clean_stats['mean']:<10.2f} | {clean_stats['min']:<10.2f}")
    print(f"{'2. Retrieval (E5/BM25)':<24} | {ret_stats['p50']:<10.2f} | {ret_stats['p70']:<10.2f} | {ret_stats['p100']:<10.2f} | {ret_stats['mean']:<10.2f} | {ret_stats['min']:<10.2f}")
    print(f"{'3. Generation (LLM)':<24} | {gen_stats['p50']:<10.2f} | {gen_stats['p70']:<10.2f} | {gen_stats['p100']:<10.2f} | {gen_stats['mean']:<10.2f} | {gen_stats['min']:<10.2f}")
    print(f"{'4. Total RAG Pipeline':<24} | {tot_stats['p50']:<10.2f} | {tot_stats['p70']:<10.2f} | {tot_stats['p100']:<10.2f} | {tot_stats['mean']:<10.2f} | {tot_stats['min']:<10.2f}")
    print("=" * 70)

    # Save benchmark report as JSON for traceability
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query_count": len(BENCHMARK_QUERIES),
        "chunk_strategy": settings.CHUNK_STRATEGY,
        "generation_provider": settings.GENERATION_PROVIDER,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "normalization_stats": clean_stats,
        "retrieval_stats": ret_stats,
        "generation_stats": gen_stats,
        "total_rag_stats": tot_stats
    }

    report_path = os.path.join(PROJECT_ROOT, "reports", "latency_benchmark.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Latency benchmark report saved to: {report_path}")

if __name__ == "__main__":
    run_benchmark()
