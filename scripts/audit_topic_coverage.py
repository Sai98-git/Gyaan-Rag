import sys
import os
import pyarrow.parquet as pq

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.ingestion.dataset_loader import download_dataset_shard

def audit_topics():
    path = download_dataset_shard()
    pf = pq.ParquetFile(path)
    print("=" * 85)
    print(f"📊 FULL DATASET SHARD AUDIT: {path}")
    print(f"Total Rows (Queries): {pf.metadata.num_rows:,}")
    print("=" * 85)

    topics = ['cooperat', 'photosynthesis', 'dna', 'democracy', 'agriculture', 'corporation', 'recycling', 'emperor']
    found = {t: [] for t in topics}

    for batch in pf.iter_batches(batch_size=10000, columns=['query_id', 'Eng_Query', 'query', 'passages']):
        for row in batch.to_pylist():
            eng_q = (row.get('Eng_Query') or '').lower()
            hi_q = (row.get('query') or '').lower()
            passages_obj = row.get('passages') or {}
            hi_p = passages_obj.get('Translated_passages') or []
            sel = passages_obj.get('is_selected') or []

            for t in topics:
                if t in eng_q or t in hi_q:
                    has_sel = any(s == 1 for s in sel)
                    # Extract first selected or first passage text
                    p_text = ""
                    for idx, s in enumerate(sel):
                        if s == 1 and idx < len(hi_p):
                            p_text = hi_p[idx]
                            break
                    if not p_text and hi_p:
                        p_text = hi_p[0]

                    found[t].append({
                        "query_id": row.get('query_id'),
                        "eng_query": row.get('Eng_Query'),
                        "hin_query": row.get('query'),
                        "has_selected": has_sel,
                        "passage_preview": p_text[:120].replace('\n', ' ') if p_text else ""
                    })

    for t in topics:
        print(f"\n🎯 Topic: '{t}' -> Found {len(found[t])} query records in dataset")
        for item in found[t][:3]:
            print(f"  • QID: {item['query_id']} | HasRel1: {item['has_selected']}")
            print(f"    Eng: \"{item['eng_query']}\"")
            print(f"    Hin: \"{item['hin_query']}\"")
            print(f"    Passage: \"{item['passage_preview']}...\"")

if __name__ == "__main__":
    audit_topics()
