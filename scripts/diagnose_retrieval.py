import sys
import os
import json
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

from backend.retrieval.multi_strategy import MultiStrategyRetriever
from backend.retrieval.vector_store import NumpyVectorStore
from backend.retrieval.embeddings import get_embedding_generator
from backend.generation.guard import check_pre_retrieval_guard

# 100 Representative Test Queries based on actual MSMARCO-XI topics + Out-of-Domain controls
DIAGNOSTIC_QUERIES = [
    # ── 1. Business / Corporate (In-Domain) ──
    ("q01", "What is a corporation?", "en", True),
    ("q02", "Who owns a corporation?", "en", True),
    ("q03", "What are the characteristics of a corporation?", "en", True),
    ("q04", "Define legal entity in business", "en", True),
    ("q05", "What is B Corp certification?", "en", True),
    ("q06", "What is McDonald's Corporation?", "en", True),
    ("q07", "How do shareholders participate in company ownership?", "en", True),
    ("q08", "Explain federal contractor requirements", "en", True),
    ("q09", "कॉर्पोरेशन क्या है?", "hi", True),
    ("q10", "निगम की परिभाषा क्या है?", "hi", True),
    ("q11", "शेयरधारक क्या होते हैं?", "hi", True),
    ("q12", "मैकडॉनल्ड्स कॉर्पोरेशन क्या है?", "hi", True),
    ("q13", "कानूनी अस्तित्व क्या होता है?", "hi", True),
    ("q14", "बी कॉर्प समुदाय क्या है?", "hi", True),
    ("q15", "कंपनी का स्वामित्व किसके पास होता है?", "hi", True),
    ("q16", "संघीय ठेकेदार कौन होते हैं?", "hi", True),
    ("q17", "Corporation kya hota hai?", "hinglish", True),
    ("q18", "Company ke shareholders ke rights kya hain?", "hinglish", True),
    ("q19", "Legal entity ka matlab kya hai?", "hinglish", True),
    ("q20", "B Corp community kya hai?", "hinglish", True),

    # ── 2. Health, Medicine & Dental (In-Domain) ──
    ("q21", "How much does a dental crown recement cost?", "en", True),
    ("q22", "What is the cost of recementing a tooth cap in Charlotte?", "en", True),
    ("q23", "What are ingrown toenail symptoms and treatments?", "en", True),
    ("q24", "What did Ontario research discover about cannabis?", "en", True),
    ("q25", "How do nasal screens filter pollutants?", "en", True),
    ("q26", "दांत का क्राउन फिर से लगाने की कीमत कितनी है?", "hi", True),
    ("q27", "दांत की टोपी लगाने का खर्च कितना होता है?", "hi", True),
    ("q28", "इनग्रोन टोनेल के लक्षण क्या हैं?", "hi", True),
    ("q29", "कनाडा में भांग पर नया शोध क्या है?", "hi", True),
    ("q30", "नाक की स्क्रीन प्रदूषकों को कैसे छानती है?", "hi", True),
    ("q31", "Tooth crown lagane ka cost kitna hai?", "hinglish", True),
    ("q32", "Ingrown toenail ka ilaj kya hai?", "hinglish", True),
    ("q33", "Bhaang par research kya kehta hai?", "hinglish", True),
    ("q34", "Nose screen pollution filter kaise karta hai?", "hinglish", True),

    # ── 3. History, Geography & Architecture (In-Domain) ──
    ("q35", "Who spent 100 million dollars building the Palace of Versailles?", "en", True),
    ("q36", "Who was the last emperor of Versailles between 1754 and 1793?", "en", True),
    ("q37", "What is the phenomenon of Chhir Batti in Kutch, Gujarat?", "en", True),
    ("q38", "Where is the electronics recycling collection in Scottsdale?", "en", True),
    ("q39", "What is the Household Hazardous Waste program in Phoenix?", "en", True),
    ("q40", "वर्साय का महल किसने बनवाया था?", "hi", True),
    ("q41", "वर्साय के अंतिम सम्राट कौन थे?", "hi", True),
    ("q42", "कच्छ गुजरात में चिर बट्टी क्या है?", "hi", True),
    ("q43", "स्कॉट्सडेल में इलेक्ट्रॉनिक्स रीसाइक्लिंग कब होती है?", "hi", True),
    ("q44", "फीनिक्स शहर का एच.एच.डब्ल्यू कार्यक्रम क्या है?", "hi", True),
    ("q45", "Versailles ka mahal kisne banwaya tha?", "hinglish", True),
    ("q46", "Kutch Gujarat mein Chhir Batti kya hota hai?", "hinglish", True),
    ("q47", "Scottsdale mein electronics recycling kahan hoti hai?", "hinglish", True),
    ("q48", "Phoenix city ka hazardous waste program kya hai?", "hinglish", True),

    # ── 4. Technology, Television & Home (In-Domain) ──
    ("q49", "What factors should you consider before buying a TV on Snapdeal?", "en", True),
    ("q50", "What is the most expensive LG 105-inch curved 5K Ultra HD TV?", "en", True),
    ("q51", "What types of curtain hooks are used for pinch pleated curtains?", "en", True),
    ("q52", "स्नैपडील पर टीवी खरीदने से पहले क्या देखना चाहिए?", "hi", True),
    ("q53", "एल.जी. 105 इंच कर्व्ड 5K टीवी की विशेषता क्या है?", "hi", True),
    ("q54", "पर्दे के हुक के विभिन्न प्रकार क्या हैं?", "hi", True),
    ("q55", "LG TV ka sabse expensive model kaun sa hai?", "hinglish", True),
    ("q56", "Curtain hooks ke types kya hain?", "hinglish", True),

    # ── 5. Paraphrases of In-Domain Topics ──
    ("q57", "Tell me about corporations in simple terms", "en", True),
    ("q58", "Explain how companies get incorporated legally", "en", True),
    ("q59", "How much money is required to fix a loose crown?", "en", True),
    ("q60", "King Louis XVI and the French Revolution at Versailles", "en", True),
    ("q61", "Mysterious glowing lights in Banni grasslands of Gujarat", "en", True),
    ("q62", "Recycle clean electronics drive in Scottsdale city yard", "en", True),
    ("q63", "निगमों के बारे में सरल शब्दों में समझाइए", "hi", True),
    ("q64", "ढीले दांत के क्राउन को ठीक करने में कितना खर्च आता है?", "hi", True),
    ("q65", "फ्रांसीसी क्रांति के समय वर्साय का महल", "hi", True),
    ("q66", "गुजरात के कच्छ में रहस्यमयी रोशनी का क्या नाम है?", "hi", True),
    ("q67", "स्कॉट्सडेल शहर में पुराने इलेक्ट्रॉनिक्स कहां जमा करें?", "hi", True),
    ("q68", "Loose tooth crown theek karne ka kharcha kitna hai?", "hinglish", True),
    ("q69", "Gujarat ke Kutch mein mysterious light kya hai?", "hinglish", True),
    ("q70", "Corporation legal person kaise hota hai?", "hinglish", True),

    # ── 6. Out-of-Domain & Unsupported Control Queries (Must Cleanly Abstain) ──
    ("q71", "What is the capital of Mars planet?", "en", False),
    ("q72", "Who is the president of Mars?", "en", False),
    ("q73", "What is the capital of India?", "en", False),
    ("q74", "What is the formula for quantum gravity?", "en", False),
    ("q75", "How to bake a chocolate cake at home?", "en", False),
    ("q76", "Who won the FIFA World Cup 2022?", "en", False),
    ("q77", "What is my bank account balance?", "en", False),
    ("q78", "What is Machine Learning?", "en", False),
    ("q79", "How to fix an airplane jet engine?", "en", False),
    ("q80", "What is the distance between Earth and Moon?", "en", False),
    ("q81", "Who was the first person to walk on Mars?", "en", False),
    ("q82", "What is the chemical formula of sulfuric acid?", "en", False),
    ("q83", "What is Python programming language?", "en", False),
    ("q84", "Who is the current Prime Minister of Japan?", "en", False),
    ("q85", "How to train a deep neural network?", "en", False),
    ("q86", "भारत की राजधानी क्या है?", "hi", False),
    ("q87", "मशीन लर्निंग क्या है?", "hi", False),
    ("q88", "चांद पर पहला व्यक्ति कौन गया था?", "hi", False),
    ("q89", "जापान के प्रधानमंत्री कौन हैं?", "hi", False),
    ("q90", "चॉकलेट केक कैसे बनाएं?", "hi", False),
    ("q91", "क्वांटम कंप्यूटर कैसे काम करता है?", "hi", False),
    ("q92", "पृथ्वी से सूर्य की दूरी कितनी है?", "hi", False),
    ("q93", "India ki capital kya hai?", "hinglish", False),
    ("q94", "Machine learning kya hota hai?", "hinglish", False),
    ("q95", "Chocolate cake kaise banaye ghar par?", "hinglish", False),
    ("q96", "Quantum computing ke rules kya hain?", "hinglish", False),
    ("q97", "Mera bank balance check karo", "hinglish", False),
    ("q98", "Earth se Moon ka distance kitna hai?", "hinglish", False),
    ("q99", "Python programming language kya hai?", "hinglish", False),
    ("q100", "Mars planet ka president kaun hai?", "hinglish", False)
]

def run_diagnostic():
    print("=" * 85)
    print("🔬 GYAAN RAG — 100-QUERY FULL RETRIEVAL & GROUNDING DIAGNOSTIC 🔬")
    print("=" * 85)

    msr = MultiStrategyRetriever(os.path.join(PROJECT_ROOT, "data", "indexes"))
    msr.load()

    vector_store = NumpyVectorStore()
    vector_store.load(os.path.join(PROJECT_ROOT, "data", "indexes", "semantic", "dense"))

    emb_gen = get_embedding_generator()

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    results = []

    for qid, qtext, qlang, expected_ev in DIAGNOSTIC_QUERIES:
        # 1. Multi-Strategy BM25 RRF Retrieval
        bm25_res = msr.search(qtext, top_k=5)

        # 2. Dense Multilingual-E5 Retrieval
        q_emb = emb_gen.embed_query(qtext)
        dense_res = vector_store.search(q_emb, top_k=5)

        # 3. Hybrid RRF Fusion between Multi-Strategy BM25 and Dense E5
        combined_chunks = {}
        dense_w = 1.0
        bm25_w = 1.0

        for rk, c in enumerate(dense_res):
            cid = c["chunk_id"]
            combined_chunks[cid] = {**c, "score": dense_w / (60 + rk + 1)}

        for rk, c in enumerate(bm25_res):
            cid = c["chunk_id"]
            if cid in combined_chunks:
                combined_chunks[cid]["score"] += bm25_w / (60 + rk + 1)
            else:
                combined_chunks[cid] = {**c, "score": bm25_w / (60 + rk + 1)}

        fused_sorted = sorted(combined_chunks.values(), key=lambda x: x["score"], reverse=True)[:10]

        # Normalize scores
        max_sc = fused_sorted[0]["score"] if fused_sorted else 1.0
        for item in fused_sorted:
            item["score"] = round(item["score"] / max_sc, 4) if max_sc > 0 else 0.0

        # Tier 1 Pre-generation guard
        pre_guard = check_pre_retrieval_guard(qtext, fused_sorted)
        has_evidence = len(fused_sorted) > 0 and (pre_guard is None)

        if expected_ev:
            if has_evidence:
                true_positives += 1
                status = "✅ [CORRECT EVIDENCE]"
            else:
                false_negatives += 1
                status = "❌ [FALSE ABSTENTION / RETRIEVAL FAIL]"
        else:
            if not has_evidence:
                true_negatives += 1
                status = "✅ [CORRECT ABSTAIN]"
            else:
                false_positives += 1
                status = "⚠️ [UNGROUNDED EVIDENCE HIT]"

        top_preview = fused_sorted[0]["text"][:65].replace("\n", " ") if fused_sorted else "EMPTY"
        print(f"[{qid}] ({qlang.upper()}/{'IN' if expected_ev else 'OUT'}) '{qtext[:30].ljust(30)}' -> {status} | Top: '{top_preview}...'")

        results.append({
            "id": qid,
            "query": qtext,
            "language": qlang,
            "expected_evidence": expected_ev,
            "retrieved_evidence": has_evidence,
            "guard_decision": "ABSTAIN" if pre_guard else "ANSWER",
            "guard_reason": pre_guard.get("guard_reason") if pre_guard else None,
            "fused_count": len(fused_sorted),
            "top_match": top_preview
        })

    total_q = len(DIAGNOSTIC_QUERIES)
    total_in = sum(1 for q in DIAGNOSTIC_QUERIES if q[3])
    total_out = sum(1 for q in DIAGNOSTIC_QUERIES if not q[3])

    print("\n" + "=" * 85)
    print("📊 100-QUERY DIAGNOSTIC EVALUATION SUMMARY")
    print("=" * 85)
    print(f"Total Test Queries Evaluated : {total_q}")
    print(f"In-Domain Evidence Queries    : {total_in}")
    print(f"Out-of-Domain Control Queries : {total_out}")
    print(f"True Positives (Correct Ev)   : {true_positives} / {total_in} ({true_positives/total_in*100:.1f}%)")
    print(f"True Negatives (Correct Abs)  : {true_negatives} / {total_out} ({true_negatives/total_out*100:.1f}%)")
    print(f"False Negatives (False Abs)   : {false_negatives} / {total_in} ({false_negatives/total_in*100:.1f}%)")
    print(f"False Positives (Halluc Risk) : {false_positives} / {total_out} ({false_positives/total_out*100:.1f}%)")
    print(f"Overall Retrieval Accuracy    : {(true_positives + true_negatives)/total_q*100:.1f}%")
    print("=" * 85)

    # Save to data/rag_evaluation.json
    out_path = os.path.join(PROJECT_ROOT, "data", "rag_evaluation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_queries": total_q,
            "metrics": {
                "in_domain_total": total_in,
                "out_of_domain_total": total_out,
                "recall_rate": round(true_positives / total_in * 100, 2),
                "abstention_accuracy": round(true_negatives / total_out * 100, 2),
                "false_abstention_rate": round(false_negatives / total_in * 100, 2),
                "overall_accuracy": round((true_positives + true_negatives) / total_q * 100, 2)
            },
            "queries": results
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDiagnostic evaluation report written to: {out_path}")

if __name__ == "__main__":
    run_diagnostic()
