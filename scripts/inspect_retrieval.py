import sys
import logging
sys.stdout.reconfigure(encoding='utf-8')
logging.disable(logging.CRITICAL)

from backend.core.config import settings
from backend.retrieval.embeddings import get_embedding_generator
from backend.retrieval.vector_store import NumpyVectorStore

vec = NumpyVectorStore()
loaded = vec.load('data/indexes/semantic/dense')
print(f"Index loaded: {loaded}, total chunks: {len(vec.chunks_metadata)}")

eg = get_embedding_generator()

q = 'What was the immediate impact of the success of the Manhattan Project?'
emb = eg.embed_query(q)
results = vec.search(emb, top_k=5)

print('=== TOP 5 RETRIEVED CHUNKS for Manhattan Project query ===')
for i, r in enumerate(results):
    score = r["score"]
    chunk_id = r["chunk_id"]
    text = r.get("text") or r.get("metadata", {}).get("text", "NO TEXT")
    print(f"--- Rank {i+1} | score={score:.4f} | chunk_id={chunk_id}")
    print("TEXT:", text[:250])
    print()

print()
print('=== TOP 3 RETRIEVED CHUNKS for corporation query ===')
q2 = 'corporation'
emb2 = eg.embed_query(q2)
results2 = vec.search(emb2, top_k=3)
for i, r in enumerate(results2):
    score = r["score"]
    chunk_id = r["chunk_id"]
    text = r.get("text") or r.get("metadata", {}).get("text", "NO TEXT")
    print(f"--- Rank {i+1} | score={score:.4f} | chunk_id={chunk_id}")
    print("TEXT:", text[:200])
    print()
