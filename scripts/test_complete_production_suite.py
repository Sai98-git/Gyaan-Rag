import sys
import os
import time
import json
import logging

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from backend.retrieval.multi_strategy import MultiStrategyRetriever
from backend.generation.sarvam import SarvamGenerator
from backend.generation.guard import check_pre_retrieval_guard, validate_generation

TEST_CASES = [
    # Grounded Domain Queries (Present in MSMARCO-XI)
    {"category": "Corporation (EN)", "query": "What is a corporation?", "expect_abstain": False},
    {"category": "Corporation (HI)", "query": "कॉर्पोरेशन क्या है?", "expect_abstain": False},
    {"category": "Corporation (Hinglish)", "query": "Corporation kya hai?", "expect_abstain": False},
    {"category": "Scottsdale Recycling", "query": "Where is the electronics recycling collection in Scottsdale?", "expect_abstain": False},
    {"category": "Democracy (EN)", "query": "What is democracy?", "expect_abstain": False},
    {"category": "Democracy (HI)", "query": "लोकतंत्र क्या है?", "expect_abstain": False},
    {"category": "Photosynthesis (EN)", "query": "What is photosynthesis?", "expect_abstain": False},
    {"category": "Photosynthesis (HI)", "query": "प्रकाश संश्लेषण क्या है?", "expect_abstain": False},
    {"category": "SSB Protein DNA", "query": "What is the function of single-strand binding protein in DNA replication?", "expect_abstain": False},
    {"category": "Goddess of Agriculture", "query": "Who is the Greek goddess of agriculture and grain?", "expect_abstain": False},
    {"category": "Emperor of Versailles", "query": "Who was the last emperor of Versailles?", "expect_abstain": False},

    # Abstention Queries (NOT present in MSMARCO-XI or Out of Domain)
    {"category": "Abstain: Mars Capital", "query": "What is the capital of Mars?", "expect_abstain": True},
    {"category": "Abstain: Live Stock", "query": "Tell me today's stock price of Apple.", "expect_abstain": True},
    {"category": "Abstain: Quantum Gravity", "query": "What is quantum gravity?", "expect_abstain": True},
    {"category": "Abstain: Cake Recipe", "query": "Tell me a recipe for chocolate cake.", "expect_abstain": True},
]

def run_suite():
    print("=" * 85)
    print("🚀 RUNNING COMPLETE GYAAN RAG PRODUCTION TEST SUITE")
    print("=" * 85)

    index_dir = os.path.join(PROJECT_ROOT, "data", "indexes")
    retriever = MultiStrategyRetriever(index_dir)
    assert retriever.load(load_dense=False), "Failed to load retriever!"
    print(f"Loaded retriever with {retriever.total_bm25_chunks:,} BM25 chunks across {len(retriever.bm25_retrievers)} strategies.\n")

    generator = SarvamGenerator()

    passed = 0
    total = len(TEST_CASES)

    results_summary = []

    for tc in TEST_CASES:
        q = tc["query"]
        expect_abstain = tc["expect_abstain"]
        cat = tc["category"]

        print(f"\n--- Testing [{cat}]: '{q}' ---")
        t0 = time.perf_counter()
        
        # 1. Retrieval
        context = retriever.search(q, top_k=5)
        t_retrieval = (time.perf_counter() - t0) * 1000

        # 2. Pre-retrieval guard
        pre_guard = check_pre_retrieval_guard(q, context)
        
        if pre_guard is not None:
            t_total = (time.perf_counter() - t0) * 1000
            ans = pre_guard["answer"]
            is_abstention = True
            t_gen = 0.0
            print(f"Pre-guard triggered: {pre_guard['guard_reason']}")
        else:
            # 3. Generation
            t_gen_start = time.perf_counter()
            gen_res = generator.generate(q, context)
            t_gen = (time.perf_counter() - t_gen_start) * 1000
            
            # 4. Post-generation guard
            validated = validate_generation(q, context, gen_res)
            ans = validated["answer"]
            is_abstention = validated.get("guard_triggered", False)
            t_total = (time.perf_counter() - t0) * 1000

        top_chunk_preview = context[0]["text"][:120] if context else "None"
        top_score = context[0]["score"] if context else 0.0
        top_bm25 = context[0]["bm25_score"] if context else 0.0

        print(f"Answer: {ans}")
        print(f"Latency: Retrieval={t_retrieval:.1f}ms | Generation={t_gen:.1f}ms | Total={t_total:.1f}ms")
        print(f"Top Source Score: BM25={top_bm25:.2f} | Fused Score={top_score:.4f}")
        print(f"Top Source Preview: {top_chunk_preview}...")

        # Evaluate pass/fail
        if expect_abstain:
            test_pass = is_abstention or "स्रोतों" in ans or "information" in ans.lower()
        else:
            test_pass = not is_abstention and len(ans) > 20 and "information" not in ans.lower()

        status_str = "✅ PASS" if test_pass else "❌ FAIL"
        if test_pass:
            passed += 1
        print(f"Result: {status_str}")

        results_summary.append({
            "category": cat,
            "query": q,
            "answer": ans,
            "retrieval_ms": round(t_retrieval, 1),
            "generation_ms": round(t_gen, 1),
            "total_ms": round(t_total, 1),
            "is_abstention": is_abstention,
            "passed": test_pass
        })

    print("\n" + "=" * 85)
    print(f"SUITE RESULTS: {passed}/{total} Passed ({(passed/total)*100:.1f}%)")
    print("=" * 85)

if __name__ == "__main__":
    run_suite()
