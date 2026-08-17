import sys
import os

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.retrieval.multi_strategy import MultiStrategyRetriever

msr = MultiStrategyRetriever("data/indexes")
loaded = msr.load()
print(f"MultiStrategyRetriever loaded: {loaded} | Total chunks indexed: {msr.total_chunks}")
print("=" * 85)

queries = [
    ("q1", "कॉर्पोरेशन क्या है?", "in_domain"),
    ("q2", "CORPORATION KYA HAI?", "in_domain"),
    ("q3", "What is a corporation?", "in_domain"),
    ("q4", "Company ke shareholders ke rights kya hain?", "in_domain"),
    ("q5", "What is B Corp certification?", "in_domain"),
    ("q6", "निगम की परिभाषा क्या है?", "in_domain"),
    ("q7", "Legal entity ka matlab kya hai?", "in_domain"),
    ("q8", "What is the capital of Mars planet?", "out_of_domain"),
    ("q9", "Who is the president of Mars?", "out_of_domain"),
    ("q10", "How to bake a chocolate cake at home?", "out_of_domain")
]

for qid, q, qtype in queries:
    results = msr.search(q, top_k=5)
    cnt = len(results)
    top_sc = results[0]["score"] if results else 0.0
    print(f"[{qid}] ({qtype[:6]}) {q.ljust(45)} -> Matches: {cnt:2d} | Top Score: {top_sc:.2f}")
    if results:
        preview = results[0]["text"][:75].replace("\n", " ")
        strategies = results[0]["strategy_hits"]
        print(f"      Top Match : {preview}...")
        print(f"      Strategies: {strategies}")
    else:
        print("      No evidence found (safe abstention candidate).")
    print("-" * 85)
