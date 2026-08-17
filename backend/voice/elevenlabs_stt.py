import time
import logging
import requests
from typing import Optional, Dict, Any
from backend.core.config import settings
from backend.voice.base import BaseSTTProvider, STTResult
from backend.voice.retry import execute_with_retry

logger = logging.getLogger(__name__)

class ElevenLabsSTTProvider(BaseSTTProvider):
    """
    Optional Speech-to-Text provider leveraging ElevenLabs Speech-to-Text API.
    """

    def __init__(self):
        self.api_key = settings.ELEVENLABS_API_KEY
        self.base_url = "https://api.elevenlabs.io/v1/speech-to-text"
        self.timeout = settings.STT_TIMEOUT_SECONDS

        if not self.api_key or self.api_key == "your_elevenlabs_api_key_here":
            logger.warning("ELEVENLABS_API_KEY is not configured.")

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        mime_type: str = "audio/wav",
        language_code: Optional[str] = None
    ) -> STTResult:
        if not audio_bytes or len(audio_bytes) == 0:
            raise ValueError("Audio content is empty (0 bytes received).")

        if not self.api_key or self.api_key == "your_elevenlabs_api_key_here":
            raise ValueError("ElevenLabs API key is missing. Set ELEVENLABS_API_KEY in your environment.")

        headers = {
            "xi-api-key": self.api_key
        }

        files = {
            "file": (filename, audio_bytes, mime_type)
        }
        
        data: Dict[str, str] = {
            "model_id": "scribe_v1"
        }
        
        if language_code:
            data["language_code"] = language_code

        logger.info(f"Sending {len(audio_bytes)} bytes audio to ElevenLabs STT...")

        def _do_request() -> requests.Response:
            res = requests.post(
                self.base_url,
                headers=headers,
                files=files,
                data=data,
                timeout=self.timeout
            )
            res.raise_for_status()
            return res

        t0 = time.perf_counter()
        try:
            response = execute_with_retry(
                _do_request,
                max_retries=2,
                initial_delay=0.5,
                backoff_factor=2.0,
                operation_name="ElevenLabs STT"
            )
            duration_ms = (time.perf_counter() - t0) * 1000
            res_json = response.json()
            
            transcript = res_json.get("text", "").strip()
            detected_lang = res_json.get("language_code", language_code or "en")

            return STTResult(
                transcript=transcript,
                language_code=detected_lang,
                duration_ms=duration_ms,
                provider="elevenlabs",
                raw_response=res_json
            )

        except requests.exceptions.Timeout as e:
            logger.error(f"ElevenLabs STT request timed out: {e}")
            raise TimeoutError("ElevenLabs STT service timed out.")
        except Exception as e:
            logger.error(f"ElevenLabs STT failure: {e}")
            raise RuntimeError(f"ElevenLabs STT failed: {e}")
