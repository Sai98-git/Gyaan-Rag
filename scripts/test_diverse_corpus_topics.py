import sys
import os

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

queries = [
    "Where is the electronics recycling collection in Scottsdale?",
    "How much does a dental crown recement cost in Charlotte?",
    "Who was the last emperor of Versailles?",
    "स्कॉट्सडेल में इलेक्ट्रॉनिक्स रीसाइक्लिंग कलेक्शन कहां है?",
    "चिर बट्टी क्या है?",
    "What is Chhir Batti in Kutch Gujarat?",
    "What is a corporation?",
    "कॉर्पोरेशन क्या है?",
    "What is the capital of Mars?",
    "What is the capital of India?"
]

print("=" * 85)
print("🎯 TESTING DIVERSE TOPICS DIRECTLY AGAINST HUGGING FACE DATASET CORPUS")
print("=" * 85)

for q in queries:
    resp = client.post("/api/query", json={"query": q})
    data = resp.json()
    guard = data.get("guard_triggered", False)
    ans = data.get("answer", "")
    lat = data.get("latency", {}).get("total_ms", 0.0)
    sources = len(data.get("sources", []))
    
    print(f"QUERY: {q}")
    print(f"  Status: {'🛡️ ABSTAINED' if guard else '✅ GROUNDED ANSWER'} | Sources: {sources} | Latency: {lat:.1f}ms")
    print(f"  Answer: {ans[:120]}...")
    print("-" * 85)
