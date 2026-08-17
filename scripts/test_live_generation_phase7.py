import sys
import os
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

from backend.api.app import app
from starlette.testclient import TestClient

client = TestClient(app)

PHASE7_QUERIES = [
    ("What is a corporation?", "en", True),
    ("कॉर्पोरेशन क्या है?", "hi", True),
    ("Corporation kya hai?", "hinglish", True),
    ("What are shareholder rights?", "en", True),
    ("What is B Corp certification?", "en", True),
    ("What is a dental crown?", "en", True),
    ("Where is Scottsdale?", "en", True),
    ("What is the capital of India?", "en", False),
    ("What is machine learning?", "en", False),
    ("What is the capital of Mars?", "en", False)
]

print("=" * 85)
print("🎯 TESTING 10 PHASE 7 QUERIES THROUGH END-TO-END RAG API")
print("=" * 85)

for i, (q, lang, expected_grounded) in enumerate(PHASE7_QUERIES, 1):
    t0 = time.perf_counter()
    resp = client.post("/api/query", json={"query": q})
    dt = (time.perf_counter() - t0) * 1000
    
    data = resp.json()
    ans = data.get("answer", "")
    guard = data.get("guard_triggered", False)
    reason = data.get("guard_reason")
    sources = len(data.get("sources", []))
    
    if expected_grounded:
        status = "✅ [GROUNDED ANSWER]" if not guard else "❌ [FALSE ABSTAIN]"
    else:
        status = "✅ [CORRECT ABSTAIN]" if guard else "⚠️ [UNSUPPORTED ANSWER]"
        
    print(f"[{i:02d}/10] ({lang.upper()}) '{q.ljust(35)}' -> {status} | Guard: {str(guard).ljust(5)} | Lat: {dt:6.1f}ms")
    print(f"       Ans: {ans[:90]}...")
    print("-" * 85)
