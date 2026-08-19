import sys
import time
import requests
import json

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass


BASE_URL = "https://gyaan-rag.vercel.app"

def test_live_production():
    print("=" * 80)
    print(f"🚀 VERIFYING LIVE OPTIMIZED PRODUCTION DEPLOYMENT AT: {BASE_URL}")
    print("=" * 80)
    
    # 1. Health check
    print("\n1. Testing GET /health...")
    r = requests.get(f"{BASE_URL}/health", timeout=15)
    print(f"   Status: {r.status_code} | Body: {r.json()}")
    assert r.status_code == 200, f"Health check failed with {r.status_code}"
    
    # 2. Text Queries
    test_queries = [
        ("What is democracy?", True),
        ("What is photosynthesis?", True),
        ("What is a corporation?", True),
        ("प्रकाश संश्लेषण क्या है?", True),
        ("कॉर्पोरेशन क्या है?", True),
        ("What is the capital of Mars?", False)
    ]
    
    print("\n2. Testing POST /api/query across standard queries...")
    for q, expect_grounded in test_queries:
        t0 = time.perf_counter()
        res = requests.post(f"{BASE_URL}/api/query", json={"query": q}, timeout=20)
        tot_ms = (time.perf_counter() - t0) * 1000
        
        if res.status_code == 200:
            data = res.json()
            ans = data.get("answer", "")
            ret_ms = data.get("retrieval", {}).get("latency_ms", 0.0)
            gen_ms = data.get("generation", {}).get("latency_ms", 0.0)
            guard = data.get("guard_triggered", False)
            
            is_abstained = guard or "don't have enough information" in ans.lower() or "पर्याप्त नहीं" in ans
            
            if expect_grounded:
                assert not is_abstained, f"Query '{q}' was unexpectedly abstained: {ans}"
                tag = "✅ GROUNDED"
            else:
                assert is_abstained, f"Query '{q}' should have abstained but answered: {ans}"
                tag = "🛡️ ABSTAINED"
                
            print(f"   [{tag}] '{q}'")
            print(f"      Ans: {ans[:80]}...")
            print(f"      Lat: Ret={ret_ms:.1f}ms | Gen={gen_ms:.1f}ms | E2E={tot_ms:.1f}ms")
        else:
            print(f"   ❌ FAILED: status={res.status_code} body={res.text[:100]}")
            assert False, f"Query failed with HTTP {res.status_code}"

    # 3. Test Streaming Endpoint
    print("\n3. Testing POST /api/stream (SSE Streaming)...")
    t0 = time.perf_counter()
    r = requests.post(f"{BASE_URL}/api/stream", json={"query": "What is photosynthesis?"}, stream=True, timeout=15)
    print(f"   Status: {r.status_code}")
    assert r.status_code == 200, f"Streaming failed with status {r.status_code}"
    
    first_token_time = None
    stream_answer = ""
    for line in r.iter_lines():
        if line:
            dec = line.decode("utf-8")
            if dec.startswith("data: "):
                data_json = json.loads(dec[6:])
                if data_json.get("type") == "token":
                    if first_token_time is None:
                        first_token_time = (time.perf_counter() - t0) * 1000
                    stream_answer += data_json.get("delta", "")
                elif data_json.get("type") == "done":
                    total_stream_ms = (time.perf_counter() - t0) * 1000
                    print(f"   [SSE Stream Complete] TTFT: {first_token_time:.1f}ms | Total Stream: {total_stream_ms:.1f}ms")
                    print(f"   Streamed Answer: {data_json.get('answer', stream_answer)[:80]}...")
                    break
                    
    # 4. Voice Endpoint Test with Valid Audio
    print("\n4. Testing POST /api/voice...")
    import io, wave
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b'\x00' * 3200)
    valid_wav = buf.getvalue()
    files = {"file": ("recording.wav", valid_wav, "audio/wav")}
    r_voice = requests.post(f"{BASE_URL}/api/voice", files=files, timeout=15)
    print(f"   Status: {r_voice.status_code} | Body: {r_voice.json()}")
    assert r_voice.status_code == 200, f"Voice test failed with {r_voice.status_code}"
    
    print("\n" + "=" * 80)
    print("🎉 ALL LIVE PRODUCTION OPTIMIZATION TESTS PASSED (100.0%)!")
    print("=" * 80)

if __name__ == "__main__":
    test_live_production()
