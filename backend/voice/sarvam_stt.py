import time
import logging
import requests
from typing import Optional, Dict, Any, Tuple
from backend.core.config import settings
from backend.voice.base import BaseSTTProvider, STTResult
from backend.voice.retry import execute_with_retry

logger = logging.getLogger(__name__)


def normalize_audio_mime(mime_type: str, filename: str, audio_bytes: bytes) -> Tuple[str, str]:
    """
    Normalizes audio MIME type and filename to match Sarvam API strict whitelist:
    1. Inspects magic bytes (RIFF -> audio/wav, OggS -> audio/ogg, EBML -> audio/webm).
    2. Strips parameters (e.g. 'audio/webm;codecs=opus' -> 'audio/webm').
    3. Normalizes common aliases (audio/x-wav -> audio/wav, audio/mpeg -> audio/mp3).
    4. Ensures filename extension matches clean MIME.
    """
    # 1. Inspect magic bytes if available
    if len(audio_bytes) >= 4:
        if audio_bytes[:4] == b"RIFF":
            return "audio/wav", "recording.wav"
        elif audio_bytes[:4] == b"OggS":
            return "audio/ogg", "recording.ogg"
        elif audio_bytes[:4] == b"\x1a\x45\xdf\xa3":
            return "audio/webm", "recording.webm"
        elif audio_bytes[:3] == b"ID3" or (audio_bytes[:2] == b"\xff\xfb"):
            return "audio/mp3", "recording.mp3"

    # 2. Strip parameters from MIME type
    clean_mime = mime_type.split(";")[0].strip().lower() if mime_type else "audio/wav"
    
    # 3. Canonical whitelist mapping for Sarvam API
    mime_map = {
        "audio/webm": ("audio/webm", "recording.webm"),
        "video/webm": ("video/webm", "recording.webm"),
        "audio/wav": ("audio/wav", "recording.wav"),
        "audio/x-wav": ("audio/wav", "recording.wav"),
        "audio/wave": ("audio/wav", "recording.wav"),
        "audio/ogg": ("audio/ogg", "recording.ogg"),
        "audio/opus": ("audio/opus", "recording.opus"),
        "audio/mp3": ("audio/mp3", "recording.mp3"),
        "audio/mpeg": ("audio/mpeg", "recording.mp3"),
        "audio/mp4": ("audio/mp4", "recording.mp4"),
        "audio/m4a": ("audio/x-m4a", "recording.m4a"),
        "audio/x-m4a": ("audio/x-m4a", "recording.m4a"),
        "audio/flac": ("audio/flac", "recording.flac"),
        "audio/aac": ("audio/aac", "recording.aac"),
    }
    
    if clean_mime in mime_map:
        return mime_map[clean_mime]
        
    return "audio/wav", "recording.wav"


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

        # Normalize MIME type and filename for Sarvam strict whitelist compliance
        normalized_mime, normalized_filename = normalize_audio_mime(mime_type, filename, audio_bytes)

        headers = {
            "api-subscription-key": self.api_key
        }

        # Format multipart data with sanitized MIME type
        files = {
            "file": (normalized_filename, audio_bytes, normalized_mime)
        }
        
        data: Dict[str, str] = {
            "model": self.model
        }
        
        if language_code:
            data["language_code"] = language_code

        logger.info(
            f"Sending {len(audio_bytes)} bytes audio (orig='{filename}', mime='{mime_type}' -> "
            f"norm_file='{normalized_filename}', norm_mime='{normalized_mime}') to Sarvam STT '{self.model}'..."
        )

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
