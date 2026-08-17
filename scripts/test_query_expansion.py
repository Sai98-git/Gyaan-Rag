import sys
import os
import re

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.retrieval.bm25 import BM25Retriever

TRANSLITERATION_MAP = {
    'corporation': ['कॉर्पोरेशन', 'निगम', 'कंपनी'],
    'corporations': ['कॉर्पोरेशन', 'निगम', 'कंपनियों'],
    'company': ['कंपनी', 'निगम', 'संस्था'],
    'companies': ['कंपनियां', 'कंपनियों', 'निगमों'],
    'shareholder': ['शेयरधारक', 'शेयरधारकों', 'स्वामित्व'],
    'shareholders': ['शेयरधारक', 'शेयरधारकों', 'हिस्सेदार'],
    'definition': ['परिभाषा', 'अर्थ'],
    'define': ['परिभाषित', 'परिभाषा'],
    'legal': ['कानूनी', 'वैधानिक'],
    'entity': ['इकाई', 'अस्तित्व', 'संस्था'],
    'b corp': ['बी कॉर्प', 'प्रमाणित बी कोर'],
    'mcdonalds': ['मैकडॉनल्ड्स', 'मैकडॉनल्ड'],
    'mcdonald': ['मैकडॉनल्ड', 'मैकडॉनल्ड्स'],
    'ownership': ['स्वामित्व', 'मालिकाना'],
    'profit': ['लाभ', 'मुनाफा'],
    'kya': ['क्या'],
    'hai': ['है'],
    'hota': ['होता'],
    'hain': ['हैं'],
    'ka': ['का'],
    'ke': ['के'],
    'ki': ['की'],
    'matlab': ['मतलब', 'अर्थ']
}

def expand_query(query: str) -> str:
    tokens = re.sub(r'[।॥\|!\?\.,;:\(\)\"\']', ' ', query.lower()).split()
    expanded = list(tokens)
    for tok in tokens:
        if tok in TRANSLITERATION_MAP:
            expanded.extend(TRANSLITERATION_MAP[tok])
    return ' '.join(expanded)

retriever = BM25Retriever()
retriever.load('data/indexes/semantic/bm25')

test_queries = [
    'कॉर्पोरेशन क्या है?',
    'CORPORATION KYA HAI?',
    'What is a corporation?',
    'Company ke shareholders ke rights kya hain?',
    'What is B Corp certification?',
    'What is the capital of Mars planet?'
]

print("=" * 80)
print("TESTING MULTILINGUAL QUERY EXPANSION ON BM25 RETRIEVAL")
print("=" * 80)

for q in test_queries:
    exp_q = expand_query(q)
    results = retriever.search(exp_q, top_k=5)
    cnt = len(results)
    top_sc = results[0]['score'] if results else 0.0
    print(f"Original : {q}")
    print(f"Expanded : {exp_q}")
    print(f"Matches  : {cnt} | Top score: {top_sc:.2f}")
    if results:
        print(f"Top Text : {results[0]['text'][:90]}...")
    print("-" * 80)
