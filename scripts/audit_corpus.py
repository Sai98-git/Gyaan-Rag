import sys
import os
import json
import re
from collections import Counter
from typing import Dict, List, Any

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def audit_corpus():
    print("=" * 80)
    print("🔍 GYAAN RAG — COMPREHENSIVE CORPUS AUDIT (MSMARCO-XI)")
    print("=" * 80)

    indexes_dir = os.path.join(PROJECT_ROOT, "data", "indexes")
    strategies = ["semantic", "sliding_window", "passage"]
    
    audit_data = {
        "summary": {},
        "strategies": {},
        "discovered_topics": {},
        "sample_documents": []
    }

    # Load semantic index as the primary reference corpus
    semantic_bm25 = os.path.join(indexes_dir, "semantic", "bm25", "bm25_index.json")
    if not os.path.exists(semantic_bm25):
        print(f"Error: {semantic_bm25} not found!")
        return

    with open(semantic_bm25, "r", encoding="utf-8") as f:
        semantic_data = json.load(f)

    chunks = semantic_data.get("chunks", [])
    print(f"Loaded {len(chunks)} chunks from semantic index.")

    # Audit each strategy
    for strat in strategies:
        bm25_path = os.path.join(indexes_dir, strat, "bm25", "bm25_index.json")
        dense_meta = os.path.join(indexes_dir, strat, "dense", "metadata.json")
        dense_emb = os.path.join(indexes_dir, strat, "dense", "embeddings.npy")

        b_count = 0
        m_count = 0
        emb_exists = os.path.exists(dense_emb)
        emb_size_kb = os.path.getsize(dense_emb) / 1024 if emb_exists else 0

        if os.path.exists(bm25_path):
            with open(bm25_path, "r", encoding="utf-8") as f:
                b_count = len(json.load(f).get("chunks", []))
        if os.path.exists(dense_meta):
            with open(dense_meta, "r", encoding="utf-8") as f:
                m_count = len(json.load(f))

        audit_data["strategies"][strat] = {
            "bm25_chunks": b_count,
            "metadata_chunks": m_count,
            "embeddings_exist": emb_exists,
            "embeddings_size_kb": round(emb_size_kb, 1),
            "aligned": b_count == m_count
        }
        print(f"Strategy '{strat}': BM25={b_count}, DenseMeta={m_count}, EmbSize={emb_size_kb:.1f}KB, Aligned={b_count == m_count}")

    # Analyze vocabulary, entities, and topics across the corpus
    word_counts = Counter()
    doc_topics = []
    
    # Search for specific topic keywords
    keyword_search = {
        "india_capital": ["भारत", "राजधानी", "दिल्ली", "नई दिल्ली"],
        "corporation_business": ["निगम", "कंपनी", "शेयरधारक", "कॉर्पोरेशन", "व्यापार", "स्वामित्व", "बी कॉर्प"],
        "medical_health": ["दवा", "इलाज", "रोग", "दांत", "डॉक्टर", "स्वास्थ्य", "उपचार", "लक्षण"],
        "technology_science": ["तकनीक", "कंप्यूटर", "इंटरनेट", "सॉफ्टवेयर", "विज्ञान", "उपकरण"],
        "law_government": ["कानून", "अदालत", "सरकार", "ठेकेदार", "संघीय", "नियम", "अधिकार"],
        "geography_places": ["देश", "शहर", "राज्य", "नदी", "स्थान", "स्कॉट्सडेल", "फीनिक्स"],
        "finance_economics": ["पैसा", "डॉलर", "ऋण", "बैंक", "निवेश", "लागत", "मूल्य", "वित्तीय"]
    }
    
    topic_matches = {k: 0 for k in keyword_search}
    
    for i, c in enumerate(chunks):
        txt = c.get("text", "")
        cid = c.get("chunk_id", f"chunk_{i}")
        doc_id = c.get("metadata", {}).get("doc_id", "")
        
        # Token count
        tokens = [w.strip() for w in re.sub(r'[।॥\|!\?\.,;:\(\)\"\'\-\n\r\t]', ' ', txt).split() if w.strip()]
        word_counts.update(tokens)
        
        # Check topic presence
        matched_topics = []
        for top_name, kws in keyword_search.items():
            if any(kw in txt for kw in kws):
                topic_matches[top_name] += 1
                matched_topics.append(top_name)
                
        if i < 20 or (i % 50 == 0):
            audit_data["sample_documents"].append({
                "chunk_id": cid,
                "doc_id": doc_id,
                "token_count": len(tokens),
                "preview": txt[:120].replace("\n", " "),
                "matched_topics": matched_topics
            })

    print("\n" + "=" * 80)
    print("📊 TOPIC DISTRIBUTION ACROSS CORPUS (N=1,023 CHUNKS)")
    print("=" * 80)
    for top_name, cnt in topic_matches.items():
        pct = (cnt / len(chunks)) * 100
        print(f"  • {top_name:<25}: {cnt:4d} chunks ({pct:.1f}%)")

    # Top 30 most frequent content words
    stopwords = {"है", "के", "की", "में", "और", "से", "का", "को", "एक", "यह", "पर", "हैं", "भी", "कि", "या", "होता", "लिए", "नहीं", "किया", "करने", "तो", "गया", "था", "हो", "जो", "कर", "होती", "ने", "इस", "रूप"}
    content_words = [(w, c) for w, c in word_counts.most_common(100) if w not in stopwords and len(w) > 1][:30]

    print("\n" + "=" * 80)
    print("🔤 TOP 30 FREQUENT CONTENT WORDS IN CORPUS")
    print("=" * 80)
    for w, c in content_words:
        print(f"  {w:<20}: {c}")

    audit_data["summary"] = {
        "total_chunks": len(chunks),
        "topic_counts": topic_matches,
        "top_content_words": dict(content_words)
    }

    # Save to data/corpus_audit.json
    out_path = os.path.join(PROJECT_ROOT, "data", "corpus_audit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)

    print(f"\nCorpus audit written to: {out_path}")

if __name__ == "__main__":
    audit_corpus()
