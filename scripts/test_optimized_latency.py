import os
import sys
import time
import json
import numpy as np

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from backend.api.app import _execute_rag_pipeline, handle_stream, QueryRequest

TEST_QUERIES = [
    # 1. English In-Domain (10)
    ("What is democracy?", "en", "in_domain"),
    ("What is photosynthesis?", "en", "in_domain"),
    ("What is a corporation?", "en", "in_domain"),
    ("Who owns a corporation?", "en", "in_domain"),
    ("Where is the electronics recycling collection in Scottsdale?", "en", "in_domain"),
    ("What is the function of single-strand binding protein in DNA replication?", "en", "in_domain"),
    ("Who is the Greek goddess of agriculture and grain?", "en", "in_domain"),
    ("Who was the last emperor of Versailles?", "en", "in_domain"),
    ("What is Chhir Batti in Kutch Gujarat?", "en", "in_domain"),
    ("What happens during cellular respiration?", "en", "in_domain"),
    
    # 2. Hindi In-Domain (10)
    ("लोकतंत्र क्या है?", "hi", "in_domain"),
    ("प्रकाश संश्लेषण क्या है?", "hi", "in_domain"),
    ("कॉर्पोरेशन क्या है?", "hi", "in_domain"),
    ("कंपनी का स्वामित्व किसके पास होता है?", "hi", "in_domain"),
    ("स्कॉट्सडेल में इलेक्ट्रॉनिक्स रीसाइक्लिंग कलेक्शन कहां है?", "hi", "in_domain"),
    ("चिर बट्टी क्या है?", "hi", "in_domain"),
    ("कृषि और अनाज की यूनानी देवी कौन है?", "hi", "in_domain"),
    ("डीएनए प्रतिकृति में सिंगल स्ट्रैंड बाइंडिंग प्रोटीन का कार्य क्या है?", "hi", "in_domain"),
    ("वर्सैल्स का अंतिम सम्राट कौन था?", "hi", "in_domain"),
    ("पौधे सूर्य के प्रकाश का उपयोग कैसे करते हैं?", "hi", "in_domain"),

    # 3. Hinglish In-Domain (10)
    ("democracy kya hai?", "hinglish", "in_domain"),
    ("photosynthesis kya hota hai?", "hinglish", "in_domain"),
    ("Corporation kya hai?", "hinglish", "in_domain"),
    ("company ka ownership kiske paas hota hai?", "hinglish", "in_domain"),
    ("Scottsdale me electronics recycling kahan hai?", "hinglish", "in_domain"),
    ("Democracy me elections kaise hote hain?", "hinglish", "in_domain"),
    ("DNA replication me protein ka role kya hai?", "hinglish", "in_domain"),
    ("Plants me photosynthesis kahan hota hai?", "hinglish", "in_domain"),
    ("Agriculture ki Greek goddess kaun thi?", "hinglish", "in_domain"),
    ("Chhir Batti Gujarat me kya hai?", "hinglish", "in_domain"),

    # 4. Out-of-Domain / Abstention Controls (10)
    ("What is the capital of Mars?", "en", "out_of_domain"),
    ("Tell me today's stock price of Apple.", "en", "out_of_domain"),
    ("What is quantum gravity?", "en", "out_of_domain"),
    ("Tell me a recipe for chocolate cake.", "en", "out_of_domain"),
    ("Who is the president of Mars?", "en", "out_of_domain"),
    ("मंगल ग्रह की राजधानी क्या है?", "hi", "out_of_domain"),
    ("आज एप्पल का शेयर भाव क्या है?", "hi", "out_of_domain"),
    ("चॉकलेट केक कैसे बनाएं?", "hi", "out_of_domain"),
    ("Mars par kaun rehta hai?", "hinglish", "out_of_domain"),
    ("How to build a supersonic rocket in the backyard?", "en", "out_of_domain")
]

def run_benchmark():
    print("=" * 90)
    print("🚀 GYAAN RAG LATENCY & GROUNDING BENCHMARK (40 COMPREHENSIVE QUERIES)")
    print("=" * 90)
    
    retrieval_latencies = []
    generation_latencies = []
    total_latencies = []
    
    in_domain_correct = 0
    in_domain_total = 0
    
    out_domain_abstained = 0
    out_domain_total = 0
    
    for idx, (query, lang, q_type) in enumerate(TEST_QUERIES):
        t0 = time.perf_counter()
        res = _execute_rag_pipeline(query)
        tot_ms = (time.perf_counter() - t0) * 1000
        
        ret_ms = res.get("retrieval_latency", 0.0)
        gen_ms = res.get("generation_latency", 0.0)
        ans = res.get("answer", "")
        guard_triggered = res.get("guard_triggered", False)
        
        retrieval_latencies.append(ret_ms)
        generation_latencies.append(gen_ms)
        total_latencies.append(tot_ms)
        
        is_abstention = guard_triggered or "don't have enough information" in ans.lower() or "पर्याप्त जानकारी नहीं" in ans or "पर्याप्त नहीं" in ans
        
        if q_type == "in_domain":
            in_domain_total += 1
            if not is_abstention:
                in_domain_correct += 1
                status = "✅ GROUNDED"
            else:
                status = "❌ FALSE ABSTENTION"
        else:
            out_domain_total += 1
            if is_abstention:
                out_domain_abstained += 1
                status = "🛡️ CORRECT ABSTENTION"
            else:
                status = "❌ FALSE ANSWER (HALLUCINATION)"
                
        print(f"[{idx+1:02d}/40] {status} | [{lang.upper()}] '{query}'")
        print(f"       Ans: {ans[:90]}...")
        print(f"       Lat: Ret={ret_ms:.2f}ms | Gen={gen_ms:.2f}ms | Tot={tot_ms:.2f}ms")

    print("\n" + "=" * 90)
    print("📊 BENCHMARK SUMMARY STATISTICS")
    print("=" * 90)
    
    ret_arr = np.array(retrieval_latencies)
    gen_arr = np.array(generation_latencies)
    tot_arr = np.array(total_latencies)
    
    print(f"Retrieval Latency (ms) : P50={np.percentile(ret_arr, 50):.2f} | P90={np.percentile(ret_arr, 90):.2f} | P95={np.percentile(ret_arr, 95):.2f} | Mean={np.mean(ret_arr):.2f}")
    print(f"Generation Latency (ms): P50={np.percentile(gen_arr, 50):.2f} | P90={np.percentile(gen_arr, 90):.2f} | P95={np.percentile(gen_arr, 95):.2f} | Mean={np.mean(gen_arr):.2f}")
    print(f"Total E2E Latency (ms) : P50={np.percentile(tot_arr, 50):.2f} | P90={np.percentile(tot_arr, 90):.2f} | P95={np.percentile(tot_arr, 95):.2f} | Mean={np.mean(tot_arr):.2f}")
    
    in_acc = (in_domain_correct / in_domain_total) * 100 if in_domain_total else 0
    out_acc = (out_domain_abstained / out_domain_total) * 100 if out_domain_total else 0
    
    print(f"\nGrounded Answer Rate (In-Domain)  : {in_domain_correct}/{in_domain_total} ({in_acc:.1f}%)")
    print(f"Correct Abstention Rate (Out-Domain): {out_domain_abstained}/{out_domain_total} ({out_acc:.1f}%)")
    print(f"Overall Grounding Accuracy          : {(in_domain_correct + out_domain_abstained) / len(TEST_QUERIES) * 100:.1f}%")
    print("=" * 90)

if __name__ == "__main__":
    run_benchmark()
