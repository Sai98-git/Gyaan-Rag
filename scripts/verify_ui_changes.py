import sys
import requests

sys.stdout.reconfigure(encoding='utf-8')
BASE = 'http://127.0.0.1:8000'

# --- 1. Static file content checks ---
print("=== STATIC FILE CHECKS ===")

checks = {
    '/components/WhyGyaanRag.js': [
        ('WHY GYAAN RAG section heading', lambda t: 'WHY GYAAN RAG' in t),
        ('Pipeline flow steps', lambda t: 'why-flow' in t),
        ('Feature: EVIDENCE-GROUNDED', lambda t: 'EVIDENCE-GROUNDED' in t),
        ('Feature: ABSTENTION', lambda t: 'ABSTENTION' in t),
        ('Feature: INDIC-FIRST', lambda t: 'INDIC-FIRST' in t),
    ],
    '/components/ResearchMetrics.js': [
        ('R@1 = 41.3%', lambda t: '41.3%' in t),
        ('R@5 = 92.3%', lambda t: '92.3%' in t),
        ('R@10 = 97.1%', lambda t: '97.1%' in t),
        ('MRR@10 = 62.5%', lambda t: '62.5%' in t),
        ('Latency = 88.81 ms', lambda t: '88.81' in t),
        ('Index size = 9.20 MB', lambda t: '9.20 MB' in t),
        ('Hindi eval label', lambda t: '104 QUERIES' in t or '104' in t),
    ],
    '/components/AnswerCard.js': [
        ('stripMockNote function', lambda t: 'stripMockNote' in t),
        ('Mock provider regex', lambda t: 'mock offline provider' in t.lower()),
        ('ANSWER heading', lambda t: "'ANSWER'" in t or '"ANSWER"' in t),
        ('Grounding badge text', lambda t: 'Grounded in retrieved context' in t),
        ('De-emphasised latency row', lambda t: 'rgba(0,20,13,0.55)' in t),
    ],
    '/style.css': [
        ('why-features-grid class', lambda t: 'why-features-grid' in t),
        ('why-flow class', lambda t: 'why-flow' in t),
        ('metrics-grid class', lambda t: 'metrics-grid' in t),
        ('metric-value class', lambda t: 'metric-value' in t),
        ('metrics-secondary-row', lambda t: 'metrics-secondary-row' in t),
    ],
    '/app.js': [
        ('WhyGyaanRag import', lambda t: 'WhyGyaanRag' in t),
        ('ResearchMetrics import', lambda t: 'ResearchMetrics' in t),
        ('why-container mount', lambda t: 'why-container' in t),
        ('metrics-container mount', lambda t: 'metrics-container' in t),
    ],
    '/index.html': [
        ('why-container section', lambda t: 'why-container' in t),
        ('metrics-container section', lambda t: 'metrics-container' in t),
    ],
}

all_pass = True
for path, tests in checks.items():
    r = requests.get(BASE + path)
    status = r.status_code
    text = r.text
    print(f"\n  {status} {path}")
    for name, fn in tests:
        ok = fn(text)
        mark = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"    [{mark}] {name}")

# --- 2. API query tests ---
print("\n\n=== API QUERY TESTS ===")

# Hindi in-domain
r = requests.post(BASE + '/api/query', json={'query': 'निगम क्या है?'}, timeout=15)
d = r.json()
answer = d.get('answer', '')
guard = d.get('guard_triggered', True)
sources = d.get('sources', [])
print("\n  Query: 'निगम क्या है?'")
print(f"    guard_triggered = {guard}")
print(f"    answer length   = {len(answer)}")
print(f"    mock note in answer = {'mock offline provider' in answer.lower()}")
print(f"    Source[0] has 'preview' = {'preview' in sources[0] if sources else 'NO SOURCES'}")
if 'mock offline provider' in answer.lower():
    all_pass = False
    print("    [FAIL] Mock provider note leaked into answer!")
else:
    print("    [PASS] Answer clean — no mock provider note")
if guard:
    all_pass = False
    print("    [FAIL] Guard triggered on in-domain query!")
else:
    print("    [PASS] Guard not triggered on in-domain query")

# Manhattan Project abstention
r2 = requests.post(BASE + '/api/query', json={'query': 'What was the immediate impact of the success of the Manhattan Project?'}, timeout=15)
d2 = r2.json()
guard2 = d2.get('guard_triggered', False)
print("\n  Query: 'Manhattan Project'")
print(f"    guard_triggered = {guard2}")
if guard2:
    print("    [PASS] Guard correctly abstained")
else:
    all_pass = False
    print("    [FAIL] Guard did NOT abstain!")

print("\n" + "="*50)
print("OVERALL:", "ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED")
