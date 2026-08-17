import sys
import os
import pyarrow.parquet as pq
import json
import time

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.ingestion.dataset_loader import download_dataset_shard

def inspect_full_dataset():
    print("=" * 85)
    print("🔍 INSPECTING FULL HUGGING FACE DATASET SHARD: ai4bharat/MSMARCO-XI (hinval.parquet)")
    print("=" * 85)

    local_path = download_dataset_shard()
    print(f"Local Parquet Path: {local_path}")
    file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"File Size on Disk: {file_size_mb:.2f} MB")

    pf = pq.ParquetFile(local_path)
    total_rows = pf.metadata.num_rows
    print(f"Total Rows (Queries): {total_rows}")

    total_passages = 0
    total_selected_passages = 0
    total_bytes_translated = 0
    total_bytes_english = 0

    topics_found = {}
    target_topics = ["cooperation", "agriculture", "climate", "economy", "computer", "corporation", "gravity", "dna", "photosynthesis", "democracy"]
    
    t0 = time.perf_counter()
    batch_size = 5000
    rows_processed = 0

    sample_records_with_selected = []

    for batch in pf.iter_batches(batch_size=batch_size, columns=["query_id", "Eng_Query", "query", "Eng_Answer", "Answer", "passages"]):
        pylist = batch.to_pylist()
        for row in pylist:
            rows_processed += 1
            eng_q = row.get("Eng_Query", "") or ""
            hi_q = row.get("query", "") or ""
            passages_obj = row.get("passages", {}) or {}
            
            eng_passages = passages_obj.get("English_passages", []) or []
            hi_passages = passages_obj.get("Translated_passages", []) or []
            is_selected = passages_obj.get("is_selected", []) or []

            num_p = len(hi_passages)
            total_passages += num_p

            for p_idx, selected in enumerate(is_selected):
                if selected == 1 and p_idx < len(hi_passages):
                    total_selected_passages += 1
                    p_text = hi_passages[p_idx] or ""
                    eng_p_text = eng_passages[p_idx] if p_idx < len(eng_passages) else ""
                    total_bytes_translated += len(p_text.encode('utf-8'))
                    total_bytes_english += len(eng_p_text.encode('utf-8'))

                    # Check topics
                    eng_q_lower = eng_q.lower()
                    for top in target_topics:
                        if top in eng_q_lower:
                            topics_found[top] = topics_found.get(top, 0) + 1

                    if len(sample_records_with_selected) < 10:
                        sample_records_with_selected.append({
                            "query_id": row.get("query_id"),
                            "Eng_Query": eng_q,
                            "query": hi_q,
                            "selected_passage_preview": p_text[:120].replace("\n", " "),
                            "eng_passage_preview": eng_p_text[:120].replace("\n", " ") if eng_p_text else ""
                        })

        if rows_processed % 20000 == 0 or rows_processed == total_rows:
            print(f"Processed {rows_processed}/{total_rows} rows in {time.perf_counter() - t0:.2f}s...")

    print("\n" + "=" * 85)
    print("📊 FULL DATASET AUDIT SUMMARY")
    print("=" * 85)
    print(f"Total Query Records in Split       : {total_rows:,}")
    print(f"Total Raw Passages                 : {total_passages:,}")
    print(f"Total Ground-Truth Selected (Rel=1): {total_selected_passages:,}")
    print(f"Total UTF-8 Size (Rel=1 Passages)  : {total_bytes_translated / (1024 * 1024):.2f} MB (Hindi) | {total_bytes_english / (1024 * 1024):.2f} MB (English)")
    print(f"Avg Passage Length (chars)         : {total_bytes_translated / total_selected_passages if total_selected_passages else 0:.1f}")
    
    print("\n" + "=" * 85)
    print("🎯 SAMPLE TOPIC OCCURRENCES IN FULL DATASET QUERIES")
    print("=" * 85)
    for top in target_topics:
        cnt = topics_found.get(top, 0)
        print(f"  • '{top:<15}': {cnt:5d} query instances with ground-truth passages")

    print("\n" + "=" * 85)
    print("📝 SAMPLE GROUND-TRUTH RECORD PAIRS")
    print("=" * 85)
    for s in sample_records_with_selected[:5]:
        print(f"Query ID: {s['query_id']}")
        print(f"  Eng Query: {s['Eng_Query']}")
        print(f"  Hin Query: {s['query']}")
        print(f"  Hin Rel=1 Passage: {s['selected_passage_preview']}...")
        print(f"  Eng Rel=1 Passage: {s['eng_passage_preview']}...")
        print("-" * 85)

if __name__ == "__main__":
    inspect_full_dataset()
