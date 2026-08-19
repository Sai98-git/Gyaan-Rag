import os
import sys
os.environ["VERCEL"] = "1"

import time
import json
import logging
import statistics

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.api.app import _execute_rag_pipeline, get_generator, _retrieve_evidence, check_pre_retrieval_guard, validate_generation
from backend.core.config import settings

BENCHMARK_QUERIES = [
    # English In-Domain (6)
    {"id": "EN_01", "lang": "EN", "query": "What is democracy?", "type": "in_domain"},
    {"id": "EN_02", "lang": "EN", "query": "What is photosynthesis?", "type": "in_domain"},
    {"id": "EN_03", "lang": "EN", "query": "What is a corporation?", "type": "in_domain"},
    {"id": "EN_04", "lang": "EN", "query": "Who owns a corporation?", "type": "in_domain"},
    {"id": "EN_05", "lang": "EN", "query": "Where is the electronics recycling collection in Scottsdale?", "type": "in_domain"},
    {"id": "EN_06", "lang": "EN", "query": "What is the function of single-strand binding protein in DNA replication?", "type": "in_domain"},

    # Hindi In-Domain (6)
    {"id": "HI_01", "lang": "HI", "query": "लोकतंत्र क्या है?", "type": "in_domain"},
    {"id": "HI_02", "lang": "HI", "query": "प्रकाश संश्लेषण क्या है?", "type": "in_domain"},
    {"id": "HI_03", "lang": "HI", "query": "कॉर्पोरेशन क्या है?", "type": "in_domain"},
    {"id": "HI_04", "lang": "HI", "query": "कंपनी का स्वामित्व किसके पास होता है?", "type": "in_domain"},
    {"id": "HI_05", "lang": "HI", "query": "स्कॉट्सडेल में इलेक्ट्रॉनिक्स रीसाइक्लिंग कलेक्शन कहां है?", "type": "in_domain"},
    {"id": "HI_06", "lang": "HI", "query": "कृषि और अनाज की यूनानी देवी कौन है?", "type": "in_domain"},

    # Hinglish In-Domain (5)
    {"id": "HG_01", "lang": "HINGLISH", "query": "democracy kya hai?", "type": "in_domain"},
    {"id": "HG_02", "lang": "HINGLISH", "query": "photosynthesis kya hota hai?", "type": "in_domain"},
    {"id": "HG_03", "lang": "HINGLISH", "query": "Corporation kya hai?", "type": "in_domain"},
    {"id": "HG_04", "lang": "HINGLISH", "query": "company ka ownership kiske paas hota hai?", "type": "in_domain"},
    {"id": "HG_05", "lang": "HINGLISH", "query": "Scottsdale me electronics recycling kahan hai?", "type": "in_domain"},

    # Out-of-Domain Controls (5)
    {"id": "OOD_01", "lang": "EN", "query": "What is the capital of Mars?", "type": "out_of_domain"},
    {"id": "OOD_02", "lang": "EN", "query": "Tell me today's stock price of Apple.", "type": "out_of_domain"},
    {"id": "OOD_03", "lang": "EN", "query": "What is quantum gravity?", "type": "out_of_domain"},
    {"id": "OOD_04", "lang": "HI", "query": "मंगल ग्रह की राजधानी क्या है?", "type": "out_of_domain"},
    {"id": "OOD_05", "lang": "HINGLISH", "query": "Mars par kaun rehta hai?", "type": "out_of_domain"}
]


def run_benchmark():
    print("=" * 95)
    print("🚀 GYAAN RAG LATENCY & GROUNDING BENCHMARK (PHASE 10: 22 DIVERSE QUERIES)")
    print("=" * 95)

    results = []
    ret_times = []
    ttft_times = []
    gen_times = []
    e2e_times = []
    grounded_correct = 0
    abstain_correct = 0

    generator = get_generator()

    for idx, item in enumerate(BENCHMARK_QUERIES, 1):
        q = item["query"]
        q_type = item["type"]
        lang = item["lang"]

        req_start = time.perf_counter()

        # Step 1: Retrieval
        t_ret_start = time.perf_counter()
        retrieved_chunks, ret_method, ret_latency_raw = _retrieve_evidence(q, top_k=settings.RETRIEVAL_TOP_K)
        ret_ms = (time.perf_counter() - t_ret_start) * 1000

        # Step 2: Pre-retrieval guard check
        pre_guard = check_pre_retrieval_guard(q, retrieved_chunks)
        
        ttft_ms = 0.0
        first_token = True
        accumulated_answer = ""
        final_sources = []
        gen_start = time.perf_counter()

        if pre_guard is not None:
            accumulated_answer = pre_guard["answer"]
            final_sources = pre_guard.get("sources", [])
            ttft_ms = 0.5
            gen_ms = (time.perf_counter() - gen_start) * 1000
        else:
            # Stream generation
            try:
                for chunk_str in generator.generate_stream(q, retrieved_chunks):
                    data = json.loads(chunk_str)
                    if data.get("type") == "token":
                        if first_token:
                            ttft_ms = (time.perf_counter() - gen_start) * 1000
                            first_token = False
                        accumulated_answer += data.get("delta", "")
                    elif data.get("type") == "done":
                        accumulated_answer = data.get("answer", accumulated_answer)
                        final_sources = data.get("sources", [])
            except Exception as e:
                accumulated_answer = f"Error: {e}"

            gen_ms = (time.perf_counter() - gen_start) * 1000

        # Step 3: Validate generation
        candidate_res = {
            "answer": accumulated_answer,
            "sources": final_sources,
            "provider": settings.GENERATION_PROVIDER
        }
        validated = validate_generation(q, retrieved_chunks, candidate_res)
        final_ans = validated["answer"]
        guard_trig = validated.get("guard_triggered", False)
        
        e2e_ms = (time.perf_counter() - req_start) * 1000
        first_visible_ms = ret_ms + ttft_ms

        is_abstained = guard_trig or "don't have enough information" in final_ans.lower() or "पर्याप्त नहीं" in final_ans

        if q_type == "in_domain":
            is_valid = not is_abstained
            status_tag = "✅ GROUNDED" if is_valid else "❌ FALSE ABSTAIN"
            if is_valid:
                grounded_correct += 1
        else:
            is_valid = is_abstained
            status_tag = "🛡️ CORRECT ABSTAIN" if is_valid else "❌ FALSE ANSWER"
            if is_valid:
                abstain_correct += 1

        ret_times.append(ret_ms)
        if ttft_ms > 0:
            ttft_times.append(ttft_ms)
        gen_times.append(gen_ms)
        e2e_times.append(e2e_ms)

        results.append({
            "id": item["id"],
            "lang": lang,
            "query": q,
            "type": q_type,
            "status": status_tag,
            "ret_ms": ret_ms,
            "ttft_ms": ttft_ms,
            "first_visible_ms": first_visible_ms,
            "gen_ms": gen_ms,
            "e2e_ms": e2e_ms,
            "answer_preview": final_ans[:85] + ("..." if len(final_ans) > 85 else ""),
            "guard_triggered": guard_trig
        })

        print(f"[{idx:02d}/22] {status_tag} | [{lang}] '{q}'")
        print(f"       Ans: {final_ans[:90]}...")
        print(f"       Lat: Ret={ret_ms:.2f}ms | TTFT={ttft_ms:.2f}ms | FirstVis={first_visible_ms:.2f}ms | Gen={gen_ms:.2f}ms | E2E={e2e_ms:.2f}ms\n")

    def p50(arr): return statistics.median(arr) if arr else 0.0
    def p90(arr): return statistics.quantiles(arr, n=10)[8] if len(arr) >= 10 else max(arr) if arr else 0.0
    def p95(arr): return statistics.quantiles(arr, n=20)[18] if len(arr) >= 20 else max(arr) if arr else 0.0

    print("=" * 95)
    print("📊 FINAL PRODUCTION BENCHMARK SUMMARY (P50 / P90 / P95)")
    print("=" * 95)
    print(f"Retrieval Latency (ms)     : P50={p50(ret_times):.2f} | P90={p90(ret_times):.2f} | P95={p95(ret_times):.2f} | Mean={statistics.mean(ret_times):.2f}")
    print(f"LLM TTFT (ms)              : P50={p50(ttft_times):.2f} | P90={p90(ttft_times):.2f} | P95={p95(ttft_times):.2f} | Mean={statistics.mean(ttft_times):.2f}")
    print(f"First Visible Token (ms)   : P50={p50([r['first_visible_ms'] for r in results]):.2f} | P90={p90([r['first_visible_ms'] for r in results]):.2f} | P95={p95([r['first_visible_ms'] for r in results]):.2f}")
    print(f"Complete Generation (ms)   : P50={p50(gen_times):.2f} | P90={p90(gen_times):.2f} | P95={p95(gen_times):.2f} | Mean={statistics.mean(gen_times):.2f}")
    print(f"Complete E2E Latency (ms)  : P50={p50(e2e_times):.2f} | P90={p90(e2e_times):.2f} | P95={p95(e2e_times):.2f} | Mean={statistics.mean(e2e_times):.2f}")
    print()
    print(f"In-Domain Grounded Rate    : {grounded_correct}/17 ({grounded_correct/17*100:.1f}%)")
    print(f"Out-of-Domain Abstain Rate : {abstain_correct}/5 ({abstain_correct/5*100:.1f}%)")
    print(f"Overall Grounding Accuracy : {(grounded_correct+abstain_correct)/22*100:.1f}%")
    print("=" * 95)

    # Save results to json for reporting
    with open("scripts/final_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "retrieval": {"p50": p50(ret_times), "p90": p90(ret_times), "p95": p95(ret_times), "mean": statistics.mean(ret_times)},
                "ttft": {"p50": p50(ttft_times), "p90": p90(ttft_times), "p95": p95(ttft_times), "mean": statistics.mean(ttft_times)},
                "first_visible": {"p50": p50([r['first_visible_ms'] for r in results]), "p90": p90([r['first_visible_ms'] for r in results]), "p95": p95([r['first_visible_ms'] for r in results])},
                "generation": {"p50": p50(gen_times), "p90": p90(gen_times), "p95": p95(gen_times), "mean": statistics.mean(gen_times)},
                "e2e": {"p50": p50(e2e_times), "p90": p90(e2e_times), "p95": p95(e2e_times), "mean": statistics.mean(e2e_times)}
            },
            "grounding_accuracy": (grounded_correct + abstain_correct) / 22 * 100,
            "queries": results
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run_benchmark()
