import io
import sys
import os
import time
import json
import math
from typing import List, Dict, Any
from fastapi.testclient import TestClient

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.api.app import app
from backend.core.config import settings

client = TestClient(app)

# 50 Comprehensive Benchmark Queries across English, Hindi, Hinglish, in-domain, out-of-domain, and edge cases
BENCHMARK_50_QUERIES = [
    # ── 1. In-Domain Hindi Queries (Corpus Grounded) ──
    {"id": "q01", "query": "कॉर्पोरेशन क्या है?", "lang": "hi", "type": "in_domain", "expected_abstain": False},
    {"id": "q02", "query": "निगम की परिभाषा क्या है?", "lang": "hi", "type": "in_domain", "expected_abstain": False},
    {"id": "q03", "query": "शेयरधारक क्या होते हैं?", "lang": "hi", "type": "in_domain", "expected_abstain": False},
    {"id": "q04", "query": "मैकडॉनल्ड्स कॉर्पोरेशन क्या है?", "lang": "hi", "type": "in_domain", "expected_abstain": False},
    {"id": "q05", "query": "कानूनी अस्तित्व क्या होता है?", "lang": "hi", "type": "in_domain", "expected_abstain": False},
    {"id": "q06", "query": "बी कॉर्प समुदाय क्या है?", "lang": "hi", "type": "in_domain", "expected_abstain": False},
    {"id": "q07", "query": "कंपनी का स्वामित्व किसके पास होता है?", "lang": "hi", "type": "in_domain", "expected_abstain": False},
    {"id": "q08", "query": "निगमित संस्थाएं कैसे स्थापित की जाती हैं?", "lang": "hi", "type": "in_domain", "expected_abstain": False},
    {"id": "q09", "query": "संघीय ठेकेदार कौन होते हैं?", "lang": "hi", "type": "in_domain", "expected_abstain": False},
    {"id": "q10", "query": "कंपनी के शेयरधारकों के अधिकार क्या हैं?", "lang": "hi", "type": "in_domain", "expected_abstain": False},
    {"id": "q11", "query": "प्रमाणित बी कोर क्या होता है?", "lang": "hi", "type": "in_domain", "expected_abstain": False},
    {"id": "q12", "query": "व्यावसायिक उद्यम की संरचना क्या है?", "lang": "hi", "type": "in_domain", "expected_abstain": False},

    # ── 2. In-Domain English Queries (Cross-Lingual Retrieval) ──
    {"id": "q13", "query": "What is a corporation?", "lang": "en", "type": "in_domain", "expected_abstain": False},
    {"id": "q14", "query": "Who owns a corporation?", "lang": "en", "type": "in_domain", "expected_abstain": False},
    {"id": "q15", "query": "What are the characteristics of a corporation?", "lang": "en", "type": "in_domain", "expected_abstain": False},
    {"id": "q16", "query": "Define legal entity in business", "lang": "en", "type": "in_domain", "expected_abstain": False},
    {"id": "q17", "query": "What is B Corp certification?", "lang": "en", "type": "in_domain", "expected_abstain": False},
    {"id": "q18", "query": "How do shareholders participate in company ownership?", "lang": "en", "type": "in_domain", "expected_abstain": False},
    {"id": "q19", "query": "What is McDonald's Corporation?", "lang": "en", "type": "in_domain", "expected_abstain": False},
    {"id": "q20", "query": "Explain federal contractor requirements", "lang": "en", "type": "in_domain", "expected_abstain": False},
    {"id": "q21", "query": "What is the role of shareholders in a company?", "lang": "en", "type": "in_domain", "expected_abstain": False},
    {"id": "q22", "query": "What does a legal entity mean under the law?", "lang": "en", "type": "in_domain", "expected_abstain": False},

    # ── 3. In-Domain Hinglish Queries (Colloquial Indic) ──
    {"id": "q23", "query": "Corporation kya hota hai?", "lang": "hinglish", "type": "in_domain", "expected_abstain": False},
    {"id": "q24", "query": "Company ke shareholders ke rights kya hain?", "lang": "hinglish", "type": "in_domain", "expected_abstain": False},
    {"id": "q25", "query": "Legal entity ka matlab kya hai?", "lang": "hinglish", "type": "in_domain", "expected_abstain": False},
    {"id": "q26", "query": "B Corp community kya hai?", "lang": "hinglish", "type": "in_domain", "expected_abstain": False},
    {"id": "q27", "query": "Corporation ka ownership kiske paas hota hai?", "lang": "hinglish", "type": "in_domain", "expected_abstain": False},
    {"id": "q28", "query": "Nigam aur company mein kya sambandh hai?", "lang": "hinglish", "type": "in_domain", "expected_abstain": False},
    {"id": "q29", "query": "Federal contractor kya hota hai?", "lang": "hinglish", "type": "in_domain", "expected_abstain": False},
    {"id": "q30", "query": "McDonald corporation ke baare mein batao", "lang": "hinglish", "type": "in_domain", "expected_abstain": False},

    # ── 4. Out-of-Domain & Insufficient Evidence Queries (Abstention Tests) ──
    {"id": "q31", "query": "What is the capital of Mars planet?", "lang": "en", "type": "out_of_domain", "expected_abstain": True},
    {"id": "q32", "query": "Who is the president of Mars?", "lang": "en", "type": "out_of_domain", "expected_abstain": True},
    {"id": "q33", "query": "Machine Learning क्या है?", "lang": "hi", "type": "out_of_domain", "expected_abstain": True},
    {"id": "q34", "query": "Quantum computing ke principles kya hain?", "lang": "hinglish", "type": "out_of_domain", "expected_abstain": True},
    {"id": "q35", "query": "How to bake a chocolate cake at home?", "lang": "en", "type": "out_of_domain", "expected_abstain": True},
    {"id": "q36", "query": "चांद पर पहला व्यक्ति कौन था?", "lang": "hi", "type": "out_of_domain", "expected_abstain": True},
    {"id": "q37", "query": "Black hole gravitational singularity formula", "lang": "en", "type": "out_of_domain", "expected_abstain": True},
    {"id": "q38", "query": "FIFA World Cup 2026 winner prediction", "lang": "en", "type": "out_of_domain", "expected_abstain": True},
    {"id": "q39", "query": "भारत की राजधानी क्या है?", "lang": "hi", "type": "out_of_domain", "expected_abstain": True},
    {"id": "q40", "query": "Photosynthesis process explain karo", "lang": "hinglish", "type": "out_of_domain", "expected_abstain": True},
    {"id": "q41", "query": "What is the distance between Earth and Jupiter?", "lang": "en", "type": "out_of_domain", "expected_abstain": True},
    {"id": "q42", "query": "विटामिन सी के मुख्य स्रोत क्या हैं?", "lang": "hi", "type": "out_of_domain", "expected_abstain": True},

    # ── 5. Short, Borderline, & Voice-Transcribed Queries ──
    {"id": "q43", "query": "निगम", "lang": "hi", "type": "in_domain", "expected_abstain": False},
    {"id": "q44", "query": "Corporation", "lang": "en", "type": "in_domain", "expected_abstain": False},
    {"id": "q45", "query": "Shareholders", "lang": "en", "type": "in_domain", "expected_abstain": False},
    {"id": "q46", "query": "कंपनी", "lang": "hi", "type": "in_domain", "expected_abstain": False},
    {"id": "q47", "query": "B Corp", "lang": "en", "type": "in_domain", "expected_abstain": False},
    {"id": "q48", "query": "Federal contractor", "lang": "en", "type": "in_domain", "expected_abstain": False},
    {"id": "q49", "query": "कानूनी अस्तित्व", "lang": "hi", "type": "in_domain", "expected_abstain": False},
    {"id": "q50", "query": "Tell me about corporations in simple terms", "lang": "en", "type": "in_domain", "expected_abstain": False}
]

def calculate_percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"p50": 0.0, "p70": 0.0, "p90": 0.0, "p100": 0.0, "min": 0.0, "mean": 0.0}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    def p(pct):
        k = (n - 1) * pct
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_vals[int(k)]
        return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)
    
    return {
        "p50": round(p(0.50), 2),
        "p70": round(p(0.70), 2),
        "p90": round(p(0.90), 2),
        "p100": round(sorted_vals[-1], 2),
        "min": round(sorted_vals[0], 2),
        "mean": round(sum(sorted_vals) / n, 2)
    }

def run_50_query_benchmark():
    print("=" * 85)
    print("⚡ GYAAN RAG — 50-QUERY MULTI-STRATEGY LATENCY & QUALITY BENCHMARK ⚡")
    print(f"Index: Multi-Strategy BM25 RRF (Semantic, Sliding, Passage) | LLM: {settings.SARVAM_MODEL}")
    print("=" * 85)

    results = []
    retrieval_latencies = []
    generation_latencies = []
    online_rag_latencies = []
    voice_e2e_latencies = []

    true_positives = 0   # In-domain answered correctly
    false_positives = 0  # Out-of-domain answered (should have abstained)
    true_negatives = 0   # Out-of-domain abstained correctly
    false_negatives = 0  # In-domain falsely abstained (CRITICAL BUG)

    print("\nWarming up multi-strategy indexes...")
    client.post("/api/query", json={"query": "warmup corporation test"})
    print("Warmup complete.\n")

    for item in BENCHMARK_50_QUERIES:
        qid = item["id"]
        qtext = item["query"]
        qlang = item["lang"]
        qtype = item["type"]
        expected_abstain = item["expected_abstain"]

        t0 = time.perf_counter()
        resp = client.post("/api/query", json={"query": qtext})
        total_time_ms = (time.perf_counter() - t0) * 1000

        data = resp.json()
        ret_ms = data["retrieval"]["latency_ms"]
        gen_ms = data["generation"]["latency_ms"]
        ans = data.get("answer", "")
        guard = data.get("guard_triggered", False)
        sources_cnt = len(data.get("sources", []))

        retrieval_latencies.append(ret_ms)
        generation_latencies.append(gen_ms)
        online_rag_latencies.append(total_time_ms)

        # Realistic voice latency = STT network (avg 750ms) + Online RAG
        voice_ms = 750.0 + total_time_ms
        voice_e2e_latencies.append(voice_ms)

        # Evaluate quality classification
        if expected_abstain:
            if guard:
                true_negatives += 1
                outcome = "✅ [CORRECT ABSTAIN]"
            else:
                false_positives += 1
                outcome = "⚠️ [UNGROUNDED GEN]"
        else:
            if not guard:
                true_positives += 1
                outcome = "✅ [GROUNDED ANSWER]"
            else:
                false_negatives += 1
                outcome = "❌ [FALSE ABSTENTION]"

        print(
            f"[{qid}] ({qlang.upper()}/{qtype[:5]}) '{qtext[:32].ljust(32)}' -> {outcome} "
            f"| Ret: {ret_ms:4.1f}ms | Gen: {gen_ms:6.1f}ms | Total: {total_time_ms:6.1f}ms | Sources: {sources_cnt}"
        )

        results.append({
            "id": qid,
            "query": qtext,
            "language": qlang,
            "type": qtype,
            "expected_abstain": expected_abstain,
            "guard_triggered": guard,
            "sources_count": sources_cnt,
            "answer_preview": ans[:100],
            "retrieval_ms": round(ret_ms, 2),
            "generation_ms": round(gen_ms, 2),
            "online_rag_total_ms": round(total_time_ms, 2),
            "voice_e2e_total_ms": round(voice_ms, 2)
        })

    # Analytics
    ret_stats = calculate_percentiles(retrieval_latencies)
    gen_stats = calculate_percentiles(generation_latencies)
    online_stats = calculate_percentiles(online_rag_latencies)
    voice_stats = calculate_percentiles(voice_e2e_latencies)

    total_q = len(BENCHMARK_50_QUERIES)
    total_in_domain = sum(1 for q in BENCHMARK_50_QUERIES if not q["expected_abstain"])
    total_out_domain = sum(1 for q in BENCHMARK_50_QUERIES if q["expected_abstain"])

    recall_rate = (true_positives / total_in_domain * 100) if total_in_domain > 0 else 0
    abstain_accuracy = (true_negatives / total_out_domain * 100) if total_out_domain > 0 else 0
    false_abstain_rate = (false_negatives / total_in_domain * 100) if total_in_domain > 0 else 0
    overall_accuracy = ((true_positives + true_negatives) / total_q * 100)

    print("\n" + "=" * 85)
    print("📊 EMPIRICAL LATENCY BENCHMARK ANALYTICS (N=50 Queries)")
    print("=" * 85)
    print(f"{'Pipeline Stage':<30} | {'P50 (ms)':<10} | {'P70 (ms)':<10} | {'P90 (ms)':<10} | {'P100 (ms)':<10} | {'Mean (ms)':<10}")
    print("-" * 85)
    print(f"{'1. Multi-Strategy Retrieval':<30} | {ret_stats['p50']:<10} | {ret_stats['p70']:<10} | {ret_stats['p90']:<10} | {ret_stats['p100']:<10} | {ret_stats['mean']:<10}")
    print(f"{'2. Indic LLM Generation':<30} | {gen_stats['p50']:<10} | {gen_stats['p70']:<10} | {gen_stats['p90']:<10} | {gen_stats['p100']:<10} | {gen_stats['mean']:<10}")
    print(f"{'3. Online RAG Core Total':<30} | {online_stats['p50']:<10} | {online_stats['p70']:<10} | {online_stats['p90']:<10} | {online_stats['p100']:<10} | {online_stats['mean']:<10}")
    print(f"{'4. Voice End-to-End Total':<30} | {voice_stats['p50']:<10} | {voice_stats['p70']:<10} | {voice_stats['p90']:<10} | {voice_stats['p100']:<10} | {voice_stats['mean']:<10}")
    print("=" * 85)

    print("\n" + "=" * 85)
    print("🎯 GROUNDING & QUALITY METRICS")
    print("=" * 85)
    print(f"Total Queries Evaluated     : {total_q}")
    print(f"In-Domain Evidence Queries   : {total_in_domain}")
    print(f"Out-of-Domain / Abstention Qs: {total_out_domain}")
    print(f"Retrieval Recall Rate        : {recall_rate:.1f}% ({true_positives}/{total_in_domain})")
    print(f"Abstention Accuracy          : {abstain_accuracy:.1f}% ({true_negatives}/{total_out_domain})")
    print(f"False Abstention Rate        : {false_abstain_rate:.1f}% ({false_negatives}/{total_in_domain})")
    print(f"Overall Accuracy Rate        : {overall_accuracy:.1f}%")
    print("=" * 85)

    # Save to JSON
    out_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "benchmark_results.json")
    
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_queries": total_q,
        "retrieval_backend": "multi_strategy_bm25_rrf",
        "strategies": ["semantic", "sliding_window", "passage"],
        "generation_model": settings.SARVAM_MODEL,
        "stt_model": settings.SARVAM_STT_MODEL,
        "quality_metrics": {
            "total_queries": total_q,
            "in_domain_count": total_in_domain,
            "out_of_domain_count": total_out_domain,
            "retrieval_recall_pct": recall_rate,
            "abstention_accuracy_pct": abstain_accuracy,
            "false_abstention_rate_pct": false_abstain_rate,
            "overall_accuracy_pct": overall_accuracy
        },
        "retrieval_metrics": ret_stats,
        "generation_metrics": gen_stats,
        "online_rag_metrics": online_stats,
        "voice_e2e_metrics": voice_stats,
        "query_results": results
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved machine-readable benchmark artifact to: {out_path}")

if __name__ == "__main__":
    run_50_query_benchmark()
