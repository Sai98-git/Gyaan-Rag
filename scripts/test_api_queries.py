import sys
import json
import requests

sys.stdout.reconfigure(encoding='utf-8')

BASE = "http://127.0.0.1:8000"

TESTS = [
    ("A - Hindi in-domain", "कॉर्पोरेशन क्या है?"),
    ("B - English in-domain", "what is corporation"),
    ("C - Hinglish in-domain", "corporation kya hai?"),
    ("D - Out-of-domain (Manhattan Project)", "What was the immediate impact of the success of the Manhattan Project?"),
    ("E - Clearly unrelated", "what is the capital of Mars?"),
]

for label, query in TESTS:
    print(f"\n{'='*60}")
    print(f"TEST {label}")
    print(f"QUERY: {query}")
    try:
        r = requests.post(f"{BASE}/api/query", json={"query": query}, timeout=30)
        print(f"HTTP STATUS: {r.status_code}")
        data = r.json()
        print(f"GUARD TRIGGERED: {data.get('guard_triggered')}")
        print(f"GUARD REASON: {data.get('guard_reason')}")
        print(f"PROVIDER: {data.get('provider')}")
        print(f"SOURCES COUNT: {len(data.get('sources', []))}")
        answer = data.get("answer", "")
        print(f"ANSWER (first 300 chars): {answer[:300]}")

        # Verify no raw metadata dumped in answer
        bad_strings = ["chunk_id", "Relevance Score", "Source 1:", "[Source", "semantic_"]
        found_bad = [b for b in bad_strings if b.lower() in answer.lower()]
        if found_bad:
            print(f"  !! BAD STRINGS IN ANSWER: {found_bad}")
        else:
            print("  ANSWER CLEAN: No raw metadata in answer text.")

        # Check first source has preview and correct score label
        if data.get('sources'):
            s0 = data['sources'][0]
            print(f"  Source[0] score: {s0.get('score')}")
            print(f"  Source[0] has preview: {'preview' in s0}")
            print(f"  Source[0] preview (first 80 chars): {str(s0.get('preview', ''))[:80]}")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "="*60)
print("All tests complete.")
