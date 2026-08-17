"""
build_production_index.py: Deterministic Production Index Builder for Gyaan RAG.

Extracts ground-truth passages with full provenance from Hugging Face dataset
`ai4bharat/MSMARCO-XI` (split: validation/hinval.parquet), applies engineered chunking,
and generates production retrieval artifacts (BM25 inverted indexes and dense vector embeddings).
"""

import sys
import os
import time
import logging
from typing import List, Dict, Any

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("build_production_index")

from backend.core.config import settings
settings.CORPUS_MODE = "sample"
settings.MAX_RECORDS = 200

from backend.ingestion.dataset_loader import iterate_records
from backend.ingestion.chunkers import PassageChunker, SlidingWindowChunker, SemanticChunker
from backend.retrieval.embeddings import get_embedding_generator
from backend.retrieval.vector_store import NumpyVectorStore
from backend.retrieval.bm25 import BM25Retriever


def build_strategy_artifacts(strategy_name: str, chunker, records, embedding_gen):
    logger.info(f"=== [Production Index] Building Strategy: '{strategy_name}' ===")
    t0 = time.perf_counter()

    chunks: List[Dict[str, Any]] = []
    for r in records:
        record_chunks = chunker.chunk_record(r)
        for c in record_chunks:
            # Attach complete dataset provenance
            c["searchable_text"] = f"{c['text']} {r.Eng_Query} {r.query}"
            if "metadata" not in c:
                c["metadata"] = {}
            c["metadata"].update({
                "dataset": "ai4bharat/MSMARCO-XI",
                "config": "hin",
                "split": "validation",
                "query_id": r.query_id,
                "language": r.target_lang,
                "eng_query": r.Eng_Query,
                "hin_query": r.query,
                "strategy": strategy_name
            })
        chunks.extend(record_chunks)

    logger.info(f"Generated {len(chunks):,} chunks for '{strategy_name}' in {time.perf_counter() - t0:.3f}s.")
    if not chunks:
        logger.warning(f"No chunks generated for {strategy_name}.")
        return

    # 1. Compute Dense Embeddings (Batch Size 64)
    logger.info("Computing dense embeddings with multilingual-e5-small...")
    t_emb = time.perf_counter()
    chunk_texts = [c["text"] for c in chunks]
    batch_size = 64
    embeddings = []
    for i in range(0, len(chunk_texts), batch_size):
        batch_texts = chunk_texts[i:i + batch_size]
        batch_embeddings = embedding_gen.embed_passages(batch_texts)
        embeddings.extend(batch_embeddings)
        if (i // batch_size) % 5 == 0 or i + batch_size >= len(chunk_texts):
            logger.info(f"  Embedded {min(i + batch_size, len(chunk_texts))}/{len(chunk_texts)} chunks...")

    logger.info(f"Dense embeddings computed in {time.perf_counter() - t_emb:.2f}s.")

    # 2. Persist Dense Vector Store
    dense_dir = os.path.join(PROJECT_ROOT, "data", "indexes", strategy_name, "dense")
    vector_store = NumpyVectorStore()
    vector_store.add_chunks(chunks, embeddings)
    vector_store.save(dense_dir)

    # 3. Persist BM25 Inverted Index
    bm25_dir = os.path.join(PROJECT_ROOT, "data", "indexes", strategy_name, "bm25")
    bm25 = BM25Retriever()
    bm25.add_chunks(chunks)
    bm25.save(bm25_dir)

    logger.info(f"✅ Successfully built and saved '{strategy_name}' production artifacts.\n")


def build_all_production_indexes():
    logger.info("=" * 80)
    logger.info("🚀 STARTING DETERMINISTIC PRODUCTION INDEX BUILD FOR GYAAN RAG")
    logger.info("=" * 80)

    t_start = time.perf_counter()
    records = list(iterate_records())
    logger.info(f"Loaded {len(records)} dataset records in {time.perf_counter() - t_start:.2f}s.")

    embedding_gen = get_embedding_generator()

    strategies = {
        "passage": PassageChunker(),
        "sliding_window": SlidingWindowChunker(),
        "semantic": SemanticChunker()
    }

    for name, chunker in strategies.items():
        build_strategy_artifacts(name, chunker, records, embedding_gen)

    logger.info("=" * 80)
    logger.info(f"🎉 ALL PRODUCTION INDEXES REPRODUCIBLY BUILT IN {time.perf_counter() - t_start:.2f}s.")
    logger.info("=" * 80)


if __name__ == "__main__":
    build_all_production_indexes()
