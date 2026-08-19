"""
test_live_production_final.py: Live verification of https://gyaan-rag.vercel.app
Tests:
- GET /health
- GET /api/health
- POST /api/query (10 real queries: grounded & abstention)
- POST /api/voice (synthetic audio)
"""

import sys
import io
import time
import json
import wave
import struct
import math
import requests

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

PROD_URL = "https://gyaan-rag.vercel.app"

QUERIES = [
    # Grounded Hindi & English
    ("What is a corporation?", True),
    ("कॉर्पोरेशन क्या है?", True),
    ("Who owns a corporation?", True),
    ("कंपनी का स्वामित्व किसके पास होता है?", True),
    ("Where is the electronics recycling collection in Scottsdale?", True),
    ("स्कॉट्सडेल में इलेक्ट्रॉनिक्स रीसाइक्लिंग कलेक्शन कहां है?", True),
    ("चिर बट्टी क्या है?", True),
    ("What is Chhir Batti in Kutch Gujarat?", True),
    # Abstention cases
    ("What is the capital of Mars?", False),
    ("What is the capital of India?", False),
    ("Who is the president of Mars?", False),
]

def create_synthetic_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        for i in range(int(16000 * 0.5)):
            wf.writeframes(struct.pack('<h', int(32767.0 * 0.5 * math.sin(2.0 * math.pi * 440.0 * (i / 16000)))))
    return buf.getvalue()

def run_prod_verification():
    print("=" * 90)
    print(f"🚀 VERIFYING LIVE VERCEL PRODUCTION DEPLOYMENT AT: {PROD_URL}")
    print("=" * 90)

    # 1. Test GET /health
    print("\n1. Testing GET /health...")
    try:
        r = requests.get(f"{PROD_URL}/health", timeout=15)
        print(f"   Status: {r.status_code} | Body: {r.json()}")
    except Exception as e:
        print(f"   GET /health failed: {e}")

    # 2. Test GET /api/health
    print("\n2. Testing GET /api/health...")
    try:
        r = requests.get(f"{PROD_URL}/api/health", timeout=15)
        print(f"   Status: {r.status_code} | Body: {r.json()}")
    except Exception as e:
        print(f"   GET /api/health failed: {e}")

    # 3. Test POST /api/query across 11 queries
    print("\n3. Testing POST /api/query across 11 Real Queries...")
    passed_queries = 0
    total_queries = len(QUERIES)

    for q, expected_grounded in QUERIES:
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{PROD_URL}/api/query", json={"query": q}, timeout=30)
            lat = (time.perf_counter() - t0) * 1000
            data = r.json()
            ans = data.get("answer", "")
            guard = data.get("guard_triggered", False)
            sources = len(data.get("sources", []))
            ret_ms = data.get("retrieval", {}).get("latency_ms", 0.0)

            is_abstention = (
                guard or
                "enough information" in ans.lower() or
                "पर्याप्त जानकारी नहीं मिली" in ans or
                "उपलब्ध स्रोतों" in ans
            )

            if expected_grounded:
                success = (not is_abstention) and (sources > 0)
                status_icon = "✅ GROUNDED" if success else "❌ FAILED"
            else:
                success = is_abstention
                status_icon = "🛡️ ABSTAINED" if success else "❌ FALSE ANSWER"

            if success:
                passed_queries += 1

            print(f"\n   [{status_icon}] Query: '{q}'")
            print(f"      Ans     : {ans[:100]}...")
            print(f"      Sources : {sources} | Ret Latency: {ret_ms:.1f}ms | Total Latency: {lat:.1f}ms")
        except Exception as e:
            print(f"   Query '{q}' failed: {e}")

    # 4. Test POST /api/voice
    print("\n4. Testing POST /api/voice...")
    try:
        wav_bytes = create_synthetic_wav()
        files = {"file": ("test.wav", wav_bytes, "audio/wav")}
        data = {"language_code": "hi-IN"}
        r = requests.post(f"{PROD_URL}/api/voice", files=files, data=data, timeout=30)
        print(f"   Status: {r.status_code} | Body: {r.json()}")
    except Exception as e:
        print(f"   POST /api/voice failed: {e}")

    print("\n" + "=" * 90)
    print(f"🎯 PRODUCTION QUERY VERIFICATION: {passed_queries}/{total_queries} PASSED ({(passed_queries/total_queries)*100:.1f}%)")
    print("=" * 90)

if __name__ == "__main__":
    run_prod_verification()
