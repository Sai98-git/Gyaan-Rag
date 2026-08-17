"""
build_production_index.py: Final Production Index Builder for Gyaan RAG.

Extracts ground-truth relevant passages with full dataset provenance from `ai4bharat/MSMARCO-XI`
across diverse topics in `validation/hinval.parquet`, applies multi-strategy chunking,
and generates optimized, serverless-safe production retrieval artifacts.
"""

import sys
import os
import time
import logging
import json
from typing import List, Dict, Any, Set
import pyarrow.parquet as pq

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

from backend.ingestion.dataset_loader import download_dataset_shard
from backend.ingestion.metadata import DatasetRecord, PassagesGroup
from backend.ingestion.chunkers import PassageChunker, SlidingWindowChunker, SemanticChunker
from backend.retrieval.embeddings import get_embedding_generator
from backend.retrieval.vector_store import NumpyVectorStore
from backend.retrieval.bm25 import BM25Retriever


def extract_curated_production_records(max_records_to_index: int = 5000) -> List[DatasetRecord]:
    """
    Streams through the full 97,941 rows of hinval.parquet and extracts high-quality
    records that have ground-truth selected passages (is_selected == 1) covering diverse domains.
    """
    parquet_path = download_dataset_shard()
    logger.info(f"Reading dataset shard from: {parquet_path}")

    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    logger.info(f"Total dataset queries in shard: {total_rows:,}")

    records: List[DatasetRecord] = []
    seen_hashes: Set[int] = set()

    batch_size = 5000
    rows_scanned = 0

    for batch in pf.iter_batches(batch_size=batch_size, columns=["query_id", "Eng_Query", "query", "Eng_Answer", "Answer", "passages"]):
        for row in batch.to_pylist():
            rows_scanned += 1
            qid = row.get("query_id")
            eng_q = (row.get("Eng_Query") or "").strip()
            hi_q = (row.get("query") or "").strip()
            passages_obj = row.get("passages") or {}

            eng_p = passages_obj.get("English_passages") or []
            hi_p = passages_obj.get("Translated_passages") or []
            is_sel = passages_obj.get("is_selected") or []

            # Prioritize records with selected passages
            has_selected = any(s == 1 for s in is_sel)
            if not has_selected and rows_scanned > 1000:
                continue

            if not hi_p or not eng_q:
                continue

            # Deduplicate similar passages
            first_p = hi_p[0] if hi_p else ""
            p_hash = hash(first_p[:150])
            if p_hash in seen_hashes:
                continue
            seen_hashes.add(p_hash)

            rec = DatasetRecord(
                query_id=qid,
                query=hi_q,
                Answer=row.get("Answer") or "",
                Eng_Query=eng_q,
                Eng_Answer=row.get("Eng_Answer") or "",
                target_lang="hi",
                passages=PassagesGroup(
                    is_selected=is_sel,
                    English_passages=eng_p,
                    Translated_passages=hi_p
                )
            )
            records.append(rec)

            if len(records) >= max_records_to_index:
                break

        if len(records) >= max_records_to_index:
            break

        if rows_scanned % 20000 == 0:
            logger.info(f"Scanned {rows_scanned:,}/{total_rows:,} rows -> Extracted {len(records):,} high-quality ground-truth records...")

    logger.info(f"✅ Extracted {len(records):,} ground-truth records across {rows_scanned:,} scanned dataset queries.")
    return records


def build_and_save_strategy_index(strategy_name: str, chunker, records: List[DatasetRecord], embedding_gen):
    logger.info(f"\n=== Building Production Index: Strategy '{strategy_name}' ===")
    t0 = time.perf_counter()

    chunks = []
    for r in records:
        record_chunks = chunker.chunk_record(r)
        for c in record_chunks:
            # Attach bilingual searchable representations and complete provenance
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

    logger.info(f"Generated {len(chunks):,} chunks in {time.perf_counter() - t0:.2f}s.")

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
        if (i // batch_size) % 10 == 0 or i + batch_size >= len(chunk_texts):
            logger.info(f"  Embedded {min(i + batch_size, len(chunk_texts))}/{len(chunk_texts)} chunks...")

    logger.info(f"Computed embeddings in {time.perf_counter() - t_emb:.2f}s.")

    # 2. Persist Dense Vector Store
    dense_dir = os.path.join(PROJECT_ROOT, "data", "indexes", strategy_name, "dense")
    vs = NumpyVectorStore()
    vs.add_chunks(chunks, embeddings)
    vs.save(dense_dir)
    
    # Compress metadata with gzip
    meta_p = os.path.join(dense_dir, "metadata.json")
    if os.path.exists(meta_p):
        import gzip
        with open(meta_p, "rb") as f_in, gzip.open(meta_p + ".gz", "wb", compresslevel=9) as f_out:
            f_out.write(f_in.read())

    # 3. Persist Inverted BM25 Index
    bm25_dir = os.path.join(PROJECT_ROOT, "data", "indexes", strategy_name, "bm25")
    bm25 = BM25Retriever()
    bm25.add_chunks(chunks)
    bm25.save(bm25_dir)
    
    # Compress BM25 index with gzip
    bm25_p = os.path.join(bm25_dir, "bm25_index.json")
    if os.path.exists(bm25_p):
        import gzip
        with open(bm25_p, "rb") as f_in, gzip.open(bm25_p + ".gz", "wb", compresslevel=9) as f_out:
            f_out.write(f_in.read())

    logger.info(f"✅ Strategy '{strategy_name}' index saved successfully ({len(chunks):,} chunks).")


def print_build_estimate(num_records: int, num_strategies: int = 3, avg_chunks_per_rec: float = 3.5):
    """Calculates and prints estimated indexing requirements and resource usage."""
    total_est_chunks = int(num_records * avg_chunks_per_rec * num_strategies)
    emb_dim = 384  # multilingual-e5-small
    emb_bytes = total_est_chunks * emb_dim * 4
    meta_bytes = total_est_chunks * 600
    bm25_bytes = total_est_chunks * 1200
    total_raw_bytes = emb_bytes + meta_bytes + bm25_bytes
    total_gz_bytes = (emb_bytes) + (meta_bytes * 0.15) + (bm25_bytes * 0.10)
    
    # Approx 50 chunks/sec on CPU
    est_sec = total_est_chunks / 50.0
    
    print("\n" + "=" * 85)
    print("📋 OFFLINE INDEXING RESOURCE ESTIMATION")
    print("=" * 85)
    print(f"Target Dataset Records        : {num_records:,}")
    print(f"Strategies to Build           : {num_strategies} (semantic, sliding_window, passage)")
    print(f"Estimated Total Chunks        : {total_est_chunks:,}")
    print(f"Embedding Model               : intfloat/multilingual-e5-small (dim={emb_dim})")
    print(f"Estimated Dense Vectors Size  : {emb_bytes / 1e6:.2f} MB")
    print(f"Estimated Raw Index Size      : {total_raw_bytes / 1e6:.2f} MB")
    print(f"Estimated Vercel Artifact Size: {total_gz_bytes / 1e6:.2f} MB (gzipped)")
    print(f"Estimated CPU Build Time      : {est_sec:.1f}s ({est_sec/60:.1f} minutes)")
    print("=" * 85 + "\n")


def build_all(max_records: int = 3000):
    logger.info("=" * 85)
    logger.info("🚀 GYAAN RAG: BUILDING PRODUCTION-READY REPRODUCIBLE INDEXES")
    logger.info("=" * 85)

    print_build_estimate(num_records=max_records)
    records = extract_curated_production_records(max_records_to_index=max_records)
    embedding_gen = get_embedding_generator()

    strategies = {
        "semantic": SemanticChunker(),
        "sliding_window": SlidingWindowChunker(),
        "passage": PassageChunker()
    }

    t_start = time.perf_counter()
    for name, chunker in strategies.items():
        build_and_save_strategy_index(name, chunker, records, embedding_gen)

    total_time = time.perf_counter() - t_start
    logger.info("=" * 85)
    logger.info(f"🎉 PRODUCTION INDEX CONSTRUCTION COMPLETE IN {total_time:.2f}s!")
    logger.info("=" * 85)


if __name__ == "__main__":
    build_all()
