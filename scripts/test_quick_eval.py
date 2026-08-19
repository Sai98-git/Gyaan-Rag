import sys
import os
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

from backend.retrieval.multi_strategy import MultiStrategyRetriever
from backend.generation.sarvam import SarvamGenerator
from backend.generation.guard import check_pre_retrieval_guard, validate_generation

r = MultiStrategyRetriever(os.path.join(PROJECT_ROOT, "data", "indexes"))
r.load(load_dense=False)
gen = SarvamGenerator()

queries = [
    "What is a corporation?",
    "कॉर्पोरेशन क्या है?",
    "Corporation kya hai?",
    "Where is the electronics recycling collection in Scottsdale?",
    "What is democracy?",
    "लोकतंत्र क्या है?",
    "What is photosynthesis?",
    "प्रकाश संश्लेषण क्या है?",
    "What is DNA replication?",
    "What is agriculture?",
    "Who was the last emperor of Versailles?",
    "What is the capital of Mars?",
    "Tell me today's stock price of Apple.",
    "What is quantum gravity?",
    "Tell me a recipe for chocolate cake."
]

for q in queries:
    t0 = time.perf_counter()
    ctx = r.search(q, top_k=5)
    t_ret = (time.perf_counter() - t0) * 1000
    
    pre = check_pre_retrieval_guard(q, ctx)
    if pre:
        ans = pre["answer"]
        t_gen = 0.0
    else:
        t_g0 = time.perf_counter()
        gres = gen.generate(q, ctx)
        t_gen = (time.perf_counter() - t_g0) * 1000
        val = validate_generation(q, ctx, gres)
        ans = val["answer"]
        
    t_tot = (time.perf_counter() - t0) * 1000
    top_p = ctx[0]["text"][:100] if ctx else "None"
    top_meta = ctx[0].get("metadata", {}) if ctx else {}
    is_sel = top_meta.get("is_selected", 0)
    top_bm25 = ctx[0].get("bm25_score", 0) if ctx else 0
    
    print(f"\n========================================================", flush=True)
    print(f"QUERY: {q}", flush=True)
    print(f"ANSWER: {ans}", flush=True)
    print(f"LATENCY: Ret={t_ret:.1f}ms | Gen={t_gen:.1f}ms | Tot={t_tot:.1f}ms", flush=True)
    print(f"TOP HIT: is_sel={is_sel} | BM25={top_bm25:.1f} | '{top_p}...'", flush=True)
