import sys
import os
import pyarrow.parquet as pq
import json
import gzip
import time
from typing import Dict, List, Any

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.ingestion.dataset_loader import download_dataset_shard
from backend.retrieval.bm25 import BM25Retriever

def build_full_dataset_index():
    print("=" * 85)
    print("🚀 BUILDING FULL DATASET RETRIEVAL CORPUS FROM ai4bharat/MSMARCO-XI")
    print("=" * 85)

    parquet_path = download_dataset_shard()
    print(f"Reading from: {parquet_path}")

    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    print(f"Total Dataset Rows: {total_rows:,}")

    chunks = []
    seen_passage_hashes = set()

    t0 = time.perf_counter()
    rows_processed = 0
    batch_size = 5000

    for batch in pf.iter_batches(batch_size=batch_size, columns=["query_id", "Eng_Query", "query", "Eng_Answer", "Answer", "passages"]):
        for row in batch.to_pylist():
            rows_processed += 1
            qid = row.get("query_id")
            eng_q = (row.get("Eng_Query") or "").strip()
            hi_q = (row.get("query") or "").strip()
            
            passages_obj = row.get("passages") or {}
            eng_passages = passages_obj.get("English_passages") or []
            hi_passages = passages_obj.get("Translated_passages") or []
            is_selected = passages_obj.get("is_selected") or []

            for p_idx, sel in enumerate(is_selected):
                if sel == 1 and p_idx < len(hi_passages):
                    hi_p = (hi_passages[p_idx] or "").strip()
                    eng_p = (eng_passages[p_idx] if p_idx < len(eng_passages) else "").strip()
                    
                    if not hi_p:
                        continue

                    # Deduplicate identical passages
                    p_hash = hash(hi_p[:200])
                    if p_hash in seen_passage_hashes:
                        continue
                    seen_passage_hashes.add(p_hash)

                    cid = f"msmarco_{qid}_{p_idx}"
                    
                    # Searchable text combines translated Hindi passage + English original passage + English Query + Hindi Query
                    # This enables TRUE cross-lingual BM25 and exact ground-truth matching
                    combined_searchable_text = f"{hi_p}\n\n[Original English]: {eng_p}\n\n[Queries]: {eng_q} | {hi_q}"

                    chunks.append({
                        "chunk_id": cid,
                        "text": hi_p,
                        "searchable_text": combined_searchable_text,
                        "metadata": {
                            "query_id": qid,
                            "passage_index": p_idx,
                            "eng_query": eng_q,
                            "hin_query": hi_q,
                            "eng_passage": eng_p[:300],
                            "is_selected": 1,
                            "dataset": "ai4bharat/MSMARCO-XI",
                            "split": "validation",
                            "strategy": "ground_truth_selected"
                        }
                    })

        if rows_processed % 20000 == 0 or rows_processed == total_rows:
            print(f"Processed {rows_processed}/{total_rows} rows | Extracted {len(chunks):,} unique passages in {time.perf_counter() - t0:.2f}s...")

    print("\n" + "=" * 85)
    print(f"✅ Extracted {len(chunks):,} unique ground-truth passages across {total_rows:,} queries.")
    print("=" * 85)

    # 1. Save Full Corpus JSON
    indexes_dir = os.path.join(PROJECT_ROOT, "data", "indexes", "full_dataset")
    os.makedirs(indexes_dir, exist_ok=True)
    bm25_dir = os.path.join(indexes_dir, "bm25")
    os.makedirs(bm25_dir, exist_ok=True)

    print(f"Building Full BM25 Index for {len(chunks):,} passages...")
    t_bm25 = time.perf_counter()
    bm25 = BM25Retriever()
    bm25.add_chunks(chunks)
    bm25.save(bm25_dir)
    print(f"BM25 Index built and saved in {time.perf_counter() - t_bm25:.2f}s.")

    index_file = os.path.join(bm25_dir, "bm25_index.json")
    size_mb = os.path.getsize(index_file) / (1024 * 1024)
    print(f"BM25 Index Size on Disk: {size_mb:.2f} MB")

    # Also build a compressed version if needed
    with open(index_file, "rb") as f_in:
        with gzip.open(index_file + ".gz", "wb") as f_out:
            f_out.writelines(f_in)
    gz_size_mb = os.path.getsize(index_file + ".gz") / (1024 * 1024)
    print(f"Gzipped Index Size: {gz_size_mb:.2f} MB")

if __name__ == "__main__":
    build_full_dataset_index()
