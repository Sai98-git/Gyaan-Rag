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

from scripts.diagnose_retrieval import DIAGNOSTIC_QUERIES
from backend.retrieval.vector_store import NumpyVectorStore
from backend.retrieval.embeddings import get_embedding_generator
from backend.retrieval.multi_strategy import MultiStrategyRetriever

vs = NumpyVectorStore()
vs.load(os.path.join(PROJECT_ROOT, "data", "indexes", "semantic", "dense"))
emb_gen = get_embedding_generator()
msr = MultiStrategyRetriever(os.path.join(PROJECT_ROOT, "data", "indexes"))
msr.load()

print("=" * 90)
print(f"{'QID':<6} | {'TYPE':<5} | {'QUERY':<40} | {'DENSE E5':<10} | {'BM25 MATCHES':<12}")
print("=" * 90)

# Sample 15 In-Domain and 15 Out-of-Domain queries
samples = DIAGNOSTIC_QUERIES[:15] + DIAGNOSTIC_QUERIES[70:85]
in_dense_scores = []
out_dense_scores = []

for qid, qtext, qlang, expected_ev in samples:
    q_emb = emb_gen.embed_query(qtext)
    d_res = vs.search(q_emb, top_k=3)
    b_res = msr.search(qtext, top_k=3)
    
    top_d = d_res[0]["score"] if d_res else 0.0
    b_cnt = len(b_res)
    
    if expected_ev:
        in_dense_scores.append(top_d)
    else:
        out_dense_scores.append(top_d)
        
    tag = "IN" if expected_ev else "OUT"
    print(f"{qid:<6} | {tag:<5} | {qtext[:38].ljust(40)} | {top_d:<10.4f} | {b_cnt:<12}")

print("=" * 90)
print(f"In-Domain Dense Scores : Min={min(in_dense_scores):.4f}, Max={max(in_dense_scores):.4f}, Avg={sum(in_dense_scores)/len(in_dense_scores):.4f}")
print(f"Out-of-Domain Scores   : Min={min(out_dense_scores):.4f}, Max={max(out_dense_scores):.4f}, Avg={sum(out_dense_scores)/len(out_dense_scores):.4f}")
print("=" * 90)
