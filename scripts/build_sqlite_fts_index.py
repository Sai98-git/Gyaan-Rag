import sys
import os
import sqlite3
import pyarrow.parquet as pq
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

def build_sqlite_fts_index():
    print("=" * 85)
    print("🚀 BUILDING HIGH-PERFORMANCE SQLITE FTS5 INDEX OVER 57,302 DATASET PASSAGES")
    print("=" * 85)

    parquet_path = download_dataset_shard()
    print(f"Reading from: {parquet_path}")

    db_dir = os.path.join(PROJECT_ROOT, "data", "indexes", "full_dataset")
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "corpus_fts.db")

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create FTS5 virtual table
    cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(
            chunk_id UNINDEXED,
            query_id UNINDEXED,
            text,
            searchable_text,
            eng_query,
            hin_query,
            eng_passage UNINDEXED,
            tokenize='unicode61 remove_diacritics 0'
        );
    """)

    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows

    t0 = time.perf_counter()
    rows_processed = 0
    batch_size = 5000
    total_inserted = 0
    seen_hashes = set()

    for batch in pf.iter_batches(batch_size=batch_size, columns=["query_id", "Eng_Query", "query", "passages"]):
        insert_batch = []
        for row in batch.to_pylist():
            rows_processed += 1
            qid = str(row.get("query_id") or "")
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

                    p_hash = hash(hi_p[:200])
                    if p_hash in seen_hashes:
                        continue
                    seen_hashes.add(p_hash)

                    cid = f"msmarco_{qid}_{p_idx}"
                    combined = f"{hi_p} {eng_p} {eng_q} {hi_q}"

                    insert_batch.append((cid, qid, hi_p, combined, eng_q, hi_q, eng_p[:300]))

        if insert_batch:
            cur.executemany("""
                INSERT INTO passages_fts (chunk_id, query_id, text, searchable_text, eng_query, hin_query, eng_passage)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, insert_batch)
            conn.commit()
            total_inserted += len(insert_batch)

        if rows_processed % 20000 == 0 or rows_processed == total_rows:
            print(f"Processed {rows_processed}/{total_rows} rows | Indexed {total_inserted:,} passages in {time.perf_counter() - t0:.2f}s...")

    # Optimize FTS index
    cur.execute("INSERT INTO passages_fts(passages_fts) VALUES('optimize');")
    conn.commit()
    conn.close()

    db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"\n✅ SQLite FTS5 database created successfully: {db_path}")
    print(f"Total Passages Indexed: {total_inserted:,}")
    print(f"Total SQLite Database Size on Disk: {db_size_mb:.2f} MB")

if __name__ == "__main__":
    build_sqlite_fts_index()
