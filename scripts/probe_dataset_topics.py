import sys
import logging
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
logging.disable(logging.CRITICAL)

from backend.retrieval.vector_store import NumpyVectorStore
from backend.retrieval.embeddings import get_embedding_generator

vec = NumpyVectorStore()
vec.load('data/indexes/semantic/dense')
eg = get_embedding_generator()

# Sample 50 chunk texts to understand what topics are in the index
print(f"Total chunks in index: {len(vec.chunks_metadata)}\n")

# Grab all query_ids and a sample of texts
texts = []
query_ids = set()
for chunk in vec.chunks_metadata:
    query_ids.add(chunk['query_id'])
    t = chunk.get('text') or chunk.get('metadata', {}).get('text', '')
    if t:
        texts.append(t.strip()[:200])

print(f"Unique source documents (query_ids): {len(query_ids)}\n")
print("=== SAMPLE PASSAGE PREVIEWS (first 30 chunks) ===")
for i, t in enumerate(texts[:30]):
    print(f"\n[{i+1}] {t}")

# Now run a battery of candidate queries and report scores
print("\n\n=== QUERY SCORE PROBE ===")
probe_queries = [
    # Likely in-domain (Hindi Wikipedia / news topics)
    "निगम क्या है",
    "corporation definition",
    "B Corp certification",
    "McDonald's corporation",
    "नैशविले का संगीत",
    "Matt Lauer Today Show",
    "Nobel Prize winners",
    "सरकारी निगम",
    "Chet Atkins Nashville sound",
    "television host NBC",
    "University research study",
    "what is a government corporation",
    "public company definition",
    "India economy",
    "legal entity definition",
    "shareholder company",
    "B Corporation movement",
    "what is B Corp",
    # Likely out-of-domain
    "World War 2",
    "Manhattan Project",
    "Cricket match India",
    "Bollywood movie",
    "Prime Minister India",
    "Python programming language",
]

THRESHOLD = 0.78
results = []
for q in probe_queries:
    emb = eg.embed_query(q)
    hits = vec.search(emb, top_k=1)
    if hits:
        score = hits[0]['score']
        results.append((score, q, hits[0].get('text', '')[:120]))

results.sort(reverse=True)
print(f"\n{'Score':>6}  {'Status':<10}  Query")
print("-"*80)
for score, q, snippet in results:
    status = "ANSWERS" if score >= THRESHOLD else "ABSTAINS"
    print(f"{score:>6.4f}  [{status}]  {q}")
    if score >= THRESHOLD:
        print(f"         → {snippet[:100]}")
