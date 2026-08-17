import io
import sys
import os
import wave
import struct
import math
import logging
from typing import Dict, Any
from fastapi.testclient import TestClient

# Ensure UTF-8 output on Windows terminal
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.api.app import app
from backend.core.config import settings
from backend.voice.cleaner import normalize_voice_query
from backend.voice.base import BaseSTTProvider, STTResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("test_pipeline")

client = TestClient(app)

def create_synthetic_wav_bytes(duration_sec: float = 1.0, freq: float = 440.0, sample_rate: int = 16000) -> bytes:
    """Generates a valid PCM 16-bit mono WAV in memory for testing audio endpoints."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav_file:
        wav_file.setnchannels(1)        # mono
        wav_file.setsampwidth(2)        # 16-bit
        wav_file.setframerate(sample_rate)
        num_frames = int(duration_sec * sample_rate)
        for i in range(num_frames):
            val = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * freq * (i / sample_rate)))
            wav_file.writeframes(struct.pack('<h', val))
    return buf.getvalue()

class MockSTTProvider(BaseSTTProvider):
    """Mock STT provider for deterministic offline integration testing."""
    def __init__(self, transcript_override="कॉर्पोरेशन क्या है?"):
        self.transcript_override = transcript_override

    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav", mime_type: str = "audio/wav", language_code=None) -> STTResult:
        if not audio_bytes:
            raise ValueError("Audio content is empty.")
        return STTResult(
            transcript=self.transcript_override,
            language_code=language_code or "hi-IN",
            duration_ms=45.2,
            provider="mock-stt"
        )

def run_tests():
    print("=" * 70)
    print("🧪 GYAAN RAG — VOICE & RETRIEVAL PIPELINE TEST SUITE 🧪")
    print("=" * 70)
    
    passed = 0
    total = 0

    # -------------------------------------------------------------
    # TEST 1: Health Endpoint Check
    # -------------------------------------------------------------
    total += 1
    print("\n[TEST 1] GET /health check")
    res = client.get("/health")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    health_data = res.json()
    assert health_data["status"] == "ok"
    assert "pipeline_type" in health_data
    assert "stt_provider" in health_data
    print(f"  ✓ Health endpoint returned OK: retrieval={health_data['retrieval']}, backend={health_data['retrieval_backend']}")
    passed += 1

    # -------------------------------------------------------------
    # TEST 2: Text In-Domain Query (Hindi)
    # -------------------------------------------------------------
    total += 1
    print("\n[TEST 2] POST /api/query - Hindi In-Domain Query")
    res = client.post("/api/query", json={"query": "कॉर्पोरेशन क्या है?"})
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    q_data = res.json()
    assert len(q_data["sources"]) > 0, "Expected retrieved sources"
    assert q_data["answer"] != "", "Expected non-empty answer"
    print(f"  ✓ Hindi Query succeeded. Answer preview: {q_data['answer'][:80]}...")
    print(f"  ✓ Sources count: {len(q_data['sources'])}, Guard triggered: {q_data['guard_triggered']}")
    passed += 1

    # -------------------------------------------------------------
    # TEST 3: Text Out-of-Domain Query (Abstention Guard Check)
    # -------------------------------------------------------------
    total += 1
    print("\n[TEST 3] POST /api/query - Out-of-Domain Grounding Guard Check")
    res = client.post("/api/query", json={"query": "what is the capital of Mars planet?"})
    assert res.status_code == 200
    ood_data = res.json()
    assert ood_data["guard_triggered"] is True or "don't have enough information" in ood_data["answer"].lower(), "Expected guard trigger or abstention"
    print(f"  ✓ Out-of-domain properly abstained: guard_triggered={ood_data['guard_triggered']}, reason='{ood_data.get('guard_reason')}'")
    passed += 1

    # -------------------------------------------------------------
    # TEST 4: Query Normalization Utility
    # -------------------------------------------------------------
    total += 1
    print("\n[TEST 4] Query Normalizer Unit Test")
    raw = '  "  haan   कॉर्पोरेशन क्या है???  "  '
    normalized = normalize_voice_query(raw)
    assert normalized == "haan कॉर्पोरेशन क्या है?", f"Unexpected normalized output: '{normalized}'"
    print(f"  ✓ Raw: '{raw}' -> Normalized: '{normalized}'")
    passed += 1

    # -------------------------------------------------------------
    # TEST 5: Voice Endpoint - Empty Audio Rejection
    # -------------------------------------------------------------
    total += 1
    print("\n[TEST 5] POST /api/voice - Empty Audio (0 Bytes Rejection)")
    files = {"file": ("empty.wav", b"", "audio/wav")}
    res = client.post("/api/voice", files=files)
    assert res.status_code == 400
    err_json = res.json()
    assert err_json.get("error") == "EMPTY_AUDIO"
    print(f"  ✓ Empty audio correctly rejected with HTTP 400: error={err_json.get('error')}")
    passed += 1

    # -------------------------------------------------------------
    # TEST 6: Voice Endpoint - Full Pipeline Execution (with Mock STT hook)
    # -------------------------------------------------------------
    total += 1
    print("\n[TEST 6] POST /api/voice - Synthetic Audio End-to-End Pipeline Execution")
    wav_bytes = create_synthetic_wav_bytes(duration_sec=0.5, freq=440.0)
    
    # Temporarily patch STT provider to MockSTTProvider to test full pipeline offline
    import backend.api.app as app_module
    orig_get_stt = app_module.get_stt_provider
    app_module.get_stt_provider = lambda: MockSTTProvider(transcript_override="कॉर्पोरेशन क्या है?")

    try:
        files = {"file": ("test_recording.wav", wav_bytes, "audio/wav")}
        data_form = {"language_code": "hi-IN"}
        res = client.post("/api/voice", files=files, data=data_form)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
        voice_res = res.json()
        assert voice_res["success"] is True
        assert voice_res["transcript"] == "कॉर्पोरेशन क्या है?"
        assert len(voice_res["sources"]) > 0
        assert "latency" in voice_res
        assert voice_res["latency"]["stt_ms"] > 0
        assert voice_res["latency"]["retrieval_ms"] >= 0
        assert voice_res["latency"]["total_ms"] > 0
        print(f"  ✓ Voice End-to-End Success:")
        print(f"    • Transcript     : {voice_res['transcript']}")
        print(f"    • STT Latency    : {voice_res['latency']['stt_ms']:.1f}ms")
        print(f"    • Retrieval Lat  : {voice_res['latency']['retrieval_ms']:.1f}ms")
        print(f"    • Gen Latency    : {voice_res['latency']['generation_ms']:.1f}ms")
        print(f"    • Total Latency  : {voice_res['latency']['total_ms']:.1f}ms")
        print(f"    • Answer Preview : {voice_res['answer'][:70]}...")
        passed += 1
    finally:
        app_module.get_stt_provider = orig_get_stt

    # -------------------------------------------------------------
    # TEST 7: Source Attribution Verification
    # -------------------------------------------------------------
    total += 1
    print("\n[TEST 7] Source Attribution Verification")
    source0 = voice_res["sources"][0]
    assert "chunk_id" in source0
    assert "score" in source0
    assert "preview" in source0
    assert "metadata" in source0
    print(f"  ✓ Source chunk structure verified: chunk_id={source0['chunk_id']}, score={source0['score']:.4f}, preview_len={len(source0['preview'])}")
    passed += 1

    print("\n" + "=" * 70)
    print(f"🎉 ALL TESTS PASSED: {passed}/{total} tests successful.")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
