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
            duration_ms=42.5,
            provider="sarvam"
        )

def run_comprehensive_tests():
    print("=" * 75)
    print("🧪 GYAAN RAG — PRODUCTION READINESS & VOICE PIPELINE TEST SUITE 🧪")
    print("=" * 75)
    
    passed = 0
    total = 0

    # -------------------------------------------------------------
    # 1. GET /health
    # -------------------------------------------------------------
    total += 1
    print("\n[TEST 1] GET /health check")
    res = client.get("/health")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    h1 = res.json()
    assert h1["status"] == "ok"
    assert h1["service"] == "gyaan-rag"
    assert "retrieval" in h1
    print(f"  ✓ /health: status={h1['status']}, retrieval={h1['retrieval']}, backend={h1['retrieval_backend']}")
    passed += 1

    # -------------------------------------------------------------
    # 2. GET /api/health
    # -------------------------------------------------------------
    total += 1
    print("\n[TEST 2] GET /api/health check")
    res = client.get("/api/health")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    h2 = res.json()
    assert h2["status"] == "ok"
    assert h2["stt_provider"] == "sarvam"
    print(f"  ✓ /api/health: status={h2['status']}, stt={h2['stt_provider']}, model={h2.get('stt_model')}")
    passed += 1

    # -------------------------------------------------------------
    # 3. POST /api/query English
    # -------------------------------------------------------------
    total += 1
    print("\n[TEST 3] POST /api/query - English In-Domain Query")
    res = client.post("/api/query", json={"query": "what is a corporation"})
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    q_en = res.json()
    assert len(q_en["sources"]) > 0, "Expected sources"
    assert len(q_en["answer"]) > 0, "Expected answer"
    print(f"  ✓ English Query: Answer='{q_en['answer'][:75]}...'")
    print(f"    • Sources: {len(q_en['sources'])}, RetLat={q_en['retrieval']['latency_ms']:.1f}ms, GenLat={q_en['generation']['latency_ms']:.1f}ms")
    passed += 1

    # -------------------------------------------------------------
    # 4. POST /api/query Hindi
    # -------------------------------------------------------------
    total += 1
    print("\n[TEST 4] POST /api/query - Hindi In-Domain Query")
    res = client.post("/api/query", json={"query": "कॉर्पोरेशन क्या है?"})
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    q_hi = res.json()
    assert len(q_hi["sources"]) > 0
    assert len(q_hi["answer"]) > 0
    print(f"  ✓ Hindi Query: Answer='{q_hi['answer'][:75]}...'")
    print(f"    • Sources: {len(q_hi['sources'])}, GuardTriggered={q_hi['guard_triggered']}")
    passed += 1

    # -------------------------------------------------------------
    # 5. POST /api/query Hinglish
    # -------------------------------------------------------------
    total += 1
    print("\n[TEST 5] POST /api/query - Hinglish In-Domain Query")
    res = client.post("/api/query", json={"query": "corporation kya hota hai?"})
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    q_hing = res.json()
    assert len(q_hing["sources"]) > 0
    print(f"  ✓ Hinglish Query: Answer='{q_hing['answer'][:75]}...'")
    passed += 1

    # -------------------------------------------------------------
    # 6. POST /api/voice with Valid Audio (Full pipeline test)
    # -------------------------------------------------------------
    total += 1
    print("\n[TEST 6] POST /api/voice - Valid Audio Pipeline Execution")
    wav_bytes = create_synthetic_wav_bytes(duration_sec=0.5, freq=440.0)
    
    import backend.api.app as app_module
    orig_get_stt = app_module.get_stt_provider
    app_module.get_stt_provider = lambda: MockSTTProvider(transcript_override="कॉर्पोरेशन क्या है?")

    try:
        files = {"file": ("recording.wav", wav_bytes, "audio/wav")}
        res = client.post("/api/voice", files=files, data={"language_code": "hi-IN"})
        assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
        voice_res = res.json()
        assert voice_res["success"] is True
        assert voice_res["transcript"] == "कॉर्पोरेशन क्या है?"
        assert voice_res["language"] == "hi-IN"
        assert len(voice_res["sources"]) > 0
        assert "latency" in voice_res
        assert "stt_ms" in voice_res["latency"]
        assert "retrieval_ms" in voice_res["latency"]
        assert "generation_ms" in voice_res["latency"]
        assert "total_ms" in voice_res["latency"]
        print(f"  ✓ Voice Pipeline Response Validated:")
        print(f"    • Transcript  : {voice_res['transcript']}")
        print(f"    • Language    : {voice_res['language']}")
        print(f"    • STT Latency : {voice_res['latency']['stt_ms']:.1f} ms")
        print(f"    • Ret Latency : {voice_res['latency']['retrieval_ms']:.1f} ms")
        print(f"    • Gen Latency : {voice_res['latency']['generation_ms']:.1f} ms")
        print(f"    • Tot Latency : {voice_res['latency']['total_ms']:.1f} ms")
        print(f"    • Sources     : {len(voice_res['sources'])} chunks")
        passed += 1
    finally:
        app_module.get_stt_provider = orig_get_stt

    # -------------------------------------------------------------
    # 7. POST /api/voice with Empty Audio (0 bytes)
    # -------------------------------------------------------------
    total += 1
    print("\n[TEST 7] POST /api/voice - Empty Audio (0 Bytes) Rejection")
    files = {"file": ("empty.wav", b"", "audio/wav")}
    res = client.post("/api/voice", files=files)
    assert res.status_code == 400
    err = res.json()
    assert err.get("error") == "EMPTY_AUDIO"
    print(f"  ✓ Correctly rejected with HTTP 400 and error code 'EMPTY_AUDIO'")
    passed += 1

    # -------------------------------------------------------------
    # 8. Out-of-Domain Query (Abstention Verification)
    # -------------------------------------------------------------
    total += 1
    print("\n[TEST 8] Out-of-Domain Query Grounding Guard Verification")
    res = client.post("/api/query", json={"query": "what is the capital of Mars planet?"})
    assert res.status_code == 200
    ood = res.json()
    assert ood["guard_triggered"] is True or "don't have enough information" in ood["answer"].lower()
    print(f"  ✓ Abstention Guard Triggered: guard_triggered={ood['guard_triggered']}, reason='{ood.get('guard_reason')}'")
    print(f"  ✓ Guard Safe Answer: '{ood['answer']}'")
    passed += 1

    # -------------------------------------------------------------
    # 9. Source Attribution & Preview Check
    # -------------------------------------------------------------
    total += 1
    print("\n[TEST 9] Source Attribution & Evidence Verification")
    source0 = q_en["sources"][0]
    assert "chunk_id" in source0
    assert "score" in source0
    assert "preview" in source0
    assert "metadata" in source0
    assert len(source0["preview"]) > 0
    print(f"  ✓ Source chunk 0: chunk_id='{source0['chunk_id']}', score={source0['score']:.4f}")
    print(f"    Preview: '{source0['preview'][:80]}...'")
    passed += 1

    # -------------------------------------------------------------
    # 10. Grounding Guard Lexical Overlap Check
    # -------------------------------------------------------------
    total += 1
    print("\n[TEST 10] Grounding Guard Lexical Overlap Validation")
    from backend.generation.guard import validate_generation
    dummy_context = [{"chunk_id": "c1", "text": "Solar energy comes from the sun.", "score": 0.85, "metadata": {}, "retrieval_method": "dense"}]
    hallucinated_dict = {"answer": "Bananas are rich in potassium and grow on trees.", "sources": dummy_context, "provider": "test"}
    guard_res = validate_generation("how solar works", dummy_context, hallucinated_dict)
    assert guard_res["guard_triggered"] is True
    assert "Zero lexical overlap" in guard_res["guard_reason"]
    print(f"  ✓ Hallucination caught: guard_triggered={guard_res['guard_triggered']}, reason='{guard_res['guard_reason']}'")
    passed += 1

    # -------------------------------------------------------------
    # 11. Vercel Production Serverless Simulation (BM25 fallback mode)
    # -------------------------------------------------------------
    total += 1
    print("\n[TEST 11] Vercel Serverless Simulation (Lightweight BM25 without PyTorch)")
    # Temporarily set embedding_gen to None to simulate Vercel serverless environment
    orig_emb = app_module.embedding_gen
    app_module.embedding_gen = None
    try:
        res = client.post("/api/query", json={"query": "कॉर्पोरेशन क्या है?"})
        assert res.status_code == 200, f"Expected 200 in serverless mode, got {res.status_code}"
        bm25_q = res.json()
        assert len(bm25_q["sources"]) > 0
        print(f"  ✓ Serverless BM25 Query Success: {len(bm25_q['sources'])} sources returned in {bm25_q['retrieval']['latency_ms']:.2f}ms")
        passed += 1
    finally:
        app_module.embedding_gen = orig_emb

    print("\n" + "=" * 75)
    print(f"🎉 ALL {passed}/{total} PRODUCTION-READINESS TESTS PASSED!")
    print("=" * 75)

if __name__ == "__main__":
    run_comprehensive_tests()
