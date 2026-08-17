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

from scripts.inspect_retrieval import inspect_query

PHASE7_QUERIES = [
    "What is a corporation?",
    "कॉर्पोरेशन क्या है?",
    "Corporation kya hai?",
    "What are shareholder rights?",
    "What is B Corp certification?",
    "What is a dental crown?",
    "Where is Scottsdale?",
    "What is the capital of India?",
    "What is machine learning?",
    "What is the capital of Mars?"
]

print("=" * 85)
print("🎯 TESTING 10 PHASE 7 QUERIES (TRUE DATASET-GROUNDED RETRIEVAL)")
print("=" * 85)

for i, q in enumerate(PHASE7_QUERIES, 1):
    print(f"\n[{i}/10] -------------------------------------------------------------")
    inspect_query(q)
