import sys
import os
import json

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(PROJECT_ROOT, "data", "indexes", "semantic", "bm25", "bm25_index.json"), "r", encoding="utf-8") as f:
    chunks = json.load(f)["chunks"]

print("=" * 80)
print("INDIAN GEOGRAPHY / CAPITAL PASSAGES IN CORPUS")
print("=" * 80)
for c in chunks:
    txt = c["text"]
    if any(k in txt for k in ["भारत", "दिल्ली", "राजधानी", "इंडिया"]):
        cid = c["chunk_id"]
        print(f"[{cid}] {txt[:220].replace(chr(10), ' ')}...")
        print("-" * 80)

print("\n" + "=" * 80)
print("SAMPLE TOPICS DISCOVERED IN CORPUS")
print("=" * 80)
sample_keywords = ["दांत", "दवा", "बैंक", "स्कॉट्सडेल", "फीनिक्स", "ऋण", "विटामिन", "इलेक्ट्रॉनिक्स", "ठेकेदार", "कंपनी"]
for kw in sample_keywords:
    matching = [c for c in chunks if kw in c["text"]]
    print(f"Keyword '{kw}': {len(matching)} matching passages. Example: {matching[0]['text'][:100].replace(chr(10), ' ')}...")
