import sys
import io
import wave
import struct
import math
import json
import requests

# Ensure UTF-8 output on Windows terminal
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

base = "https://gyaan-rag.vercel.app"

print("=" * 75)
print("🔥 LIVE PRODUCTION SMOKE TEST REPORT FOR:", base)
print("=" * 75)

# 1. GET /health
r_h = requests.get(f"{base}/health", timeout=15)
print("\n[1] GET /health:")
print("HTTP Status:", r_h.status_code)
print("Response JSON:", json.dumps(r_h.json(), indent=2, ensure_ascii=False))
assert r_h.status_code == 200

# 2. GET /api/health
r_ah = requests.get(f"{base}/api/health", timeout=15)
print("\n[2] GET /api/health:")
print("HTTP Status:", r_ah.status_code)
print("Response JSON:", json.dumps(r_ah.json(), indent=2, ensure_ascii=False))
assert r_ah.status_code == 200

# 3. POST /api/query - English
r_q1 = requests.post(f"{base}/api/query", json={"query": "What is a corporation?"}, timeout=45)
print("\n[3] POST /api/query - English: 'What is a corporation?':")
print("HTTP Status:", r_q1.status_code)
q1_data = r_q1.json()
print("Answer:", q1_data.get("answer"))
print("Sources Count:", len(q1_data.get("sources", [])))
print("Guard Triggered:", q1_data.get("guard_triggered"))
assert r_q1.status_code == 200

# 4. POST /api/query - Hindi
r_q2 = requests.post(f"{base}/api/query", json={"query": "कॉर्पोरेशन क्या है?"}, timeout=45)
print("\n[4] POST /api/query - Hindi: 'कॉर्पोरेशन क्या है?':")
print("HTTP Status:", r_q2.status_code)
q2_data = r_q2.json()
print("Answer:", q2_data.get("answer"))
print("Sources Count:", len(q2_data.get("sources", [])))
print("Guard Triggered:", q2_data.get("guard_triggered"))
assert r_q2.status_code == 200

# 5. POST /api/query - Mars (Out of domain)
r_q3 = requests.post(f"{base}/api/query", json={"query": "What is the capital of Mars?"}, timeout=45)
print("\n[5] POST /api/query - Out-of-Domain: 'What is the capital of Mars?':")
print("HTTP Status:", r_q3.status_code)
q3_data = r_q3.json()
print("Answer:", q3_data.get("answer"))
print("Guard Triggered:", q3_data.get("guard_triggered"), "Reason:", q3_data.get("guard_reason"))
assert r_q3.status_code == 200

# 6. POST /api/voice - Live Multipart Audio Upload
buf = io.BytesIO()
with wave.open(buf, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    for i in range(16000):
        val = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * 440.0 * (i / 16000)))
        wf.writeframes(struct.pack('<h', val))

files = {'file': ('audio.wav', buf.getvalue(), 'audio/wav')}
data = {'language_code': 'hi-IN'}
r_v = requests.post(f"{base}/api/voice", files=files, data=data, timeout=45)
print("\n[6] POST /api/voice (Live Multipart Audio Upload):")
print("HTTP Status:", r_v.status_code)
v_data = r_v.json()
print("Response JSON:", json.dumps(v_data, indent=2, ensure_ascii=False))
assert r_v.status_code == 200

print("\n" + "=" * 75)
print("🎉 ALL PRODUCTION SMOKE TESTS PASSED ON LIVE DEPLOYMENT: " + base)
print("=" * 75)
