import os, sys, wave, struct, math
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import io
from fastapi.testclient import TestClient
from backend.api.app import app

client = TestClient(app)

# Generate a valid 1-second 16kHz PCM audio WAV file
sample_rate = 16000
duration = 1.0
n_samples = int(sample_rate * duration)

buf = io.BytesIO()
with wave.open(buf, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    for i in range(n_samples):
        # 440 Hz tone
        sample = int(32767.0 * 0.1 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
        wf.writeframesraw(struct.pack("<h", sample))

audio_bytes = buf.getvalue()
print(f"Generated valid WAV audio: {len(audio_bytes)} bytes")

res = client.post(
    "/api/voice",
    files={"file": ("recording.wav", audio_bytes, "audio/wav")},
    data={"language": "hi-IN", "prompt": "कॉर्पोरेशन क्या है?"}
)
print("Voice Response status:", res.status_code)
print("Voice Response JSON:", res.json())
