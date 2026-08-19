import os
import sys
import time
import json

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from backend.api.app import _execute_rag_pipeline

BASELINE_QUERIES = [
    ("What is democracy?", "en", "in_domain"),
    ("What is photosynthesis?", "en", "in_domain"),
    ("What is a corporation?", "en", "in_domain"),
    ("प्रकाश संश्लेषण क्या है?", "hi", "in_domain"),
    ("कॉर्पोरेशन क्या है?", "hi", "in_domain"),
    ("democracy kya hai?", "hinglish", "in_domain"),
    ("What is the capital of Mars?", "en", "out_of_domain"),
]

def run_baseline():
    print("=" * 80)
    print("BASELINE PROFILING RUN")
    print("=" * 80)
    
    results = []
    
    for query, lang, q_type in BASELINE_QUERIES:
        print(f"\nEvaluating: '{query}' [{lang} / {q_type}]")
        t0 = time.perf_counter()
        res = _execute_rag_pipeline(query)
        tot_ms = (time.perf_counter() - t0) * 1000
        
        entry = {
            "query": query,
            "lang": lang,
            "q_type": q_type,
            "retrieval_ms": res.get("retrieval_latency", 0.0),
            "generation_ms": res.get("generation_latency", 0.0),
            "total_ms": tot_ms,
            "sources_count": len(res.get("sources", [])),
            "top_chunk_id": res.get("sources", [{}])[0].get("chunk_id", "") if res.get("sources") else "",
            "top_score": res.get("sources", [{}])[0].get("score", 0.0) if res.get("sources") else 0.0,
            "guard_triggered": res.get("guard_triggered", False),
            "guard_reason": res.get("guard_reason"),
            "answer": res.get("answer", "")
        }
        results.append(entry)
        
        print(f"  Ans: {entry['answer'][:100]}...")
        print(f"  Lat: Ret={entry['retrieval_ms']:.2f}ms | Gen={entry['generation_ms']:.2f}ms | Tot={entry['total_ms']:.2f}ms")
        print(f"  Guard: triggered={entry['guard_triggered']}, reason={entry['guard_reason']}")
        print(f"  Sources: {entry['sources_count']} (Top score: {entry['top_score']})")

    # Save baseline to scratch / artifact
    baseline_path = os.path.join(PROJECT_ROOT, "scripts", "baseline_results.json")
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nBaseline saved to: {baseline_path}")

if __name__ == "__main__":
    run_baseline()
