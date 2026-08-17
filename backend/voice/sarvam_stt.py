import time
import logging
import requests
from typing import Optional, Dict, Any
from backend.core.config import settings
from backend.voice.base import BaseSTTProvider, STTResult
from backend.voice.retry import execute_with_retry

logger = logging.getLogger(__name__)

class SarvamSTTProvider(BaseSTTProvider):
    """
    Production Speech-to-Text provider leveraging Sarvam AI's Saaras Indic STT API.
    
    Supports Indic languages (Hindi, Marathi, Bengali, Tamil, Telugu, etc.)
    as well as Indian English with automatic language detection.
    """

    def __init__(self):
        self.api_key = settings.SARVAM_API_KEY
        self.model = settings.SARVAM_STT_MODEL
        self.base_url = "https://api.sarvam.ai/speech-to-text"
        self.timeout = settings.STT_TIMEOUT_SECONDS

        if not self.api_key or self.api_key == "your_sarvam_api_key_here":
            logger.warning("SARVAM_API_KEY is not configured for SarvamSTTProvider.")

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        mime_type: str = "audio/wav",
        language_code: Optional[str] = None
    ) -> STTResult:
        if not audio_bytes or len(audio_bytes) == 0:
            raise ValueError("Audio content is empty (0 bytes received).")

        if not self.api_key or self.api_key == "your_sarvam_api_key_here":
            raise ValueError("Sarvam API key is missing. Set SARVAM_API_KEY in your environment.")

        headers = {
            "api-subscription-key": self.api_key
        }

        # Format multipart data
        files = {
            "file": (filename, audio_bytes, mime_type)
        }
        
        data: Dict[str, str] = {
            "model": self.model
        }
        
        if language_code:
            data["language_code"] = language_code

        logger.info(
            f"Sending {len(audio_bytes)} bytes audio ({filename}, {mime_type}) "
            f"to Sarvam STT model '{self.model}'..."
        )

        def _do_request() -> requests.Response:
            res = requests.post(
                self.base_url,
                headers=headers,
                files=files,
                data=data,
                timeout=self.timeout
            )
            # Raise for status so retry handler catches 429, 500, 502, 503
            res.raise_for_status()
            return res

        t0 = time.perf_counter()
        try:
            response = execute_with_retry(
                _do_request,
                max_retries=2,
                initial_delay=0.5,
                backoff_factor=2.0,
                operation_name="Sarvam STT"
            )
            duration_ms = (time.perf_counter() - t0) * 1000
            
            res_json = response.json()
            transcript = res_json.get("transcript", "").strip()
            detected_lang = res_json.get("language_code", language_code or "hi-IN")

            logger.info(
                f"Sarvam STT success in {duration_ms:.2f}ms: "
                f"transcript='{transcript[:100]}', lang='{detected_lang}'"
            )

            return STTResult(
                transcript=transcript,
                language_code=detected_lang,
                duration_ms=duration_ms,
                provider="sarvam",
                raw_response=res_json
            )

        except requests.exceptions.Timeout as e:
            logger.error(f"Sarvam STT request timed out after {self.timeout}s: {e}")
            raise TimeoutError("Speech-to-text service timed out. Please try speaking again.")
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else "unknown"
            error_body = ""
            try:
                error_body = e.response.text if e.response is not None else ""
            except Exception:
                pass
            logger.error(f"Sarvam STT HTTP error {status_code}: {error_body}")
            raise RuntimeError(f"Sarvam STT API returned HTTP {status_code}: {error_body}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Sarvam STT network error: {e}")
            raise RuntimeError(f"Network error connecting to Sarvam STT service: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in Sarvam STT: {e}", exc_info=True)
            raise RuntimeError(f"Speech transcription failed: {e}")
