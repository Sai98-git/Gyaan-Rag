import sys
import os
import time

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.retrieval.bm25 import BM25Retriever

def test_full_dataset():
    bm25 = BM25Retriever()
    t0 = time.perf_counter()
    bm25.load(os.path.join(PROJECT_ROOT, "data", "indexes", "full_dataset", "bm25"))
    print(f"Loaded {len(bm25.chunks):,} dataset passages in {time.perf_counter() - t0:.2f}s!\n")

    queries = [
        "What is cooperation?",
        "What is agriculture?",
        "What is DNA replication?",
        "What is photosynthesis?",
        "What is democracy?",
        "What is a corporation?",
        "कॉर्पोरेशन क्या है?",
        "प्रकाश संश्लेषण क्या है?",
        "What is the capital of Mars planet?"
    ]

    for q in queries:
        t_q = time.perf_counter()
        res = bm25.search(q, top_k=2)
        dt = (time.perf_counter() - t_q) * 1000
        print("=" * 85)
        print(f"QUERY: \"{q}\" (Search time: {dt:.2f}ms)")
        if res:
            for i, hit in enumerate(res, 1):
                cid = hit["chunk_id"]
                sc = hit["score"]
                txt = hit["text"].replace("\n", " ")
                meta = hit.get("metadata", {})
                eng_q = meta.get("eng_query", "")
                hin_q = meta.get("hin_query", "")
                print(f"  [{i}] ID: {cid} | Score: {sc:.2f}")
                print(f"      Text: {txt[:130]}...")
                print(f"      Matched Ground-Truth Query: \"{eng_q}\" (Hindi: \"{hin_q}\")")
        else:
            print("  ❌ NO MATCH (Empty)")

if __name__ == "__main__":
    test_full_dataset()
