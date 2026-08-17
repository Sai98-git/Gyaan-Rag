import sys
import os
import json
import time

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
from starlette.testclient import TestClient
from scripts.diagnose_retrieval import DIAGNOSTIC_QUERIES

client = TestClient(app)

print("=" * 85)
print("🔬 FULL END-TO-END RAG PIPELINE DIAGNOSTIC (PRE-GUARD + LLM + POST-GUARD) 🔬")
print("=" * 85)

# Test a representative 30 queries (15 In-Domain + 15 Out-of-Domain)
eval_subset = DIAGNOSTIC_QUERIES[:15] + DIAGNOSTIC_QUERIES[70:85]
tp, fp, tn, fn = 0, 0, 0, 0
results = []

for qid, qtext, qlang, expected_ev in eval_subset:
    t0 = time.perf_counter()
    resp = client.post("/api/query", json={"query": qtext})
    dt = (time.perf_counter() - t0) * 1000
    
    data = resp.json()
    ans = data.get("answer", "")
    guard = data.get("guard_triggered", False)
    reason = data.get("guard_reason")
    sources_cnt = len(data.get("sources", []))
    
    if expected_ev:
        if not guard:
            tp += 1
            status = "✅ [GROUNDED ANSWER]"
        else:
            fn += 1
            status = "❌ [FALSE ABSTENTION]"
    else:
        if guard:
            tn += 1
            status = "✅ [CORRECT ABSTENTION]"
        else:
            fp += 1
            status = "⚠️ [UNGROUNDED HALLUCINATION]"
            
    print(f"[{qid}] ({qlang.upper()}/{'IN' if expected_ev else 'OUT'}) '{qtext[:30].ljust(30)}' -> {status} | Latency: {dt:6.1f}ms")
    print(f"      Ans: '{ans[:85]}...'")

total_in = sum(1 for q in eval_subset if q[3])
total_out = sum(1 for q in eval_subset if not q[3])
total_q = len(eval_subset)

print("\n" + "=" * 85)
print("🎯 END-TO-END EVALUATION METRICS")
print("=" * 85)
print(f"In-Domain Grounded Answers    : {tp} / {total_in} ({tp/total_in*100:.1f}%)")
print(f"Out-of-Domain Abstentions     : {tn} / {total_out} ({tn/total_out*100:.1f}%)")
print(f"False Abstentions             : {fn} / {total_in} ({fn/total_in*100:.1f}%)")
print(f"Ungrounded Hallucinations     : {fp} / {total_out} ({fp/total_out*100:.1f}%)")
print(f"Overall Accuracy              : {(tp + tn)/total_q*100:.1f}%")
print("=" * 85)
