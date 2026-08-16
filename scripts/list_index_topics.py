import sys
import logging
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
logging.disable(logging.CRITICAL)

from backend.retrieval.vector_store import NumpyVectorStore

vec = NumpyVectorStore()
vec.load('data/indexes/semantic/dense')

# Group chunks by query_id and grab their first passage
topics = defaultdict(list)
for chunk in vec.chunks_metadata:
    qid = chunk['query_id']
    text = (chunk.get('text') or chunk.get('metadata', {}).get('text', '')).strip()[:200]
    if text:
        topics[qid].append(text)

print(f"Distinct source documents: {len(topics)}\n")
print("=== ALL SOURCE DOCUMENT TOPIC PREVIEWS ===\n")
for qid, passages in sorted(topics.items()):
    print(f"--- doc:{qid} ---")
    print(passages[0][:180])
    print()
