import sys
import os
import json

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.retrieval.multi_strategy import MultiStrategyRetriever
from backend.generation.guard import check_pre_retrieval_guard

def inspect_query(query: str):
    print("=" * 80)
    print(f"QUERY: \"{query}\"")
    print("=" * 80)

    msr = MultiStrategyRetriever(os.path.join(PROJECT_ROOT, "data", "indexes"))
    msr.load(load_dense=True)

    results = msr.search(query, top_k=5)
    
    if not results:
        print("\n❌ NO CANDIDATE PASSAGES RETRIEVED (Empty context)")
        print("Guard Decision: ABSTAIN (No candidates)")
        return

    print(f"\nRetrieved {len(results)} Evidence Passages:\n")
    for i, hit in enumerate(results, 1):
        cid = hit.get("chunk_id", "unknown")
        txt = hit.get("text", "").replace("\n", " ")
        b_sc = hit.get("bm25_score", 0.0)
        d_sc = hit.get("dense_score", 0.0)
        r_sc = hit.get("rrf_score", 0.0)
        sources = ", ".join(hit.get("strategy_hits", []))

        print(f"--- Retrieved #{i} ---")
        print(f"ID     : {cid}")
        print(f"Text   : {txt[:140]}...")
        print(f"BM25   : {b_sc:.4f}")
        print(f"Dense  : {d_sc:.4f}")
        print(f"RRF    : {r_sc:.5f}")
        print(f"Sources: {sources}")
        print()

    guard_decision = check_pre_retrieval_guard(query, results)
    if guard_decision:
        print("🛡️ Pre-Gen Guard Decision: ABSTAIN")
        print(f"   Reason: {guard_decision.get('guard_reason')}")
        print(f"   Fallback Output: \"{guard_decision.get('answer')}\"")
    else:
        print("✅ Pre-Gen Guard Decision: ALLOW GENERATION (Sufficient Evidence Present)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    else:
        q = "What is a corporation?"
    inspect_query(q)
