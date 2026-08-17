from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

@dataclass
class STTResult:
    """Structured result returned from a speech-to-text provider."""
    transcript: str
    language_code: Optional[str] = None
    confidence: Optional[float] = None
    duration_ms: float = 0.0
    provider: str = "unknown"
    raw_response: Optional[Dict[str, Any]] = None

class BaseSTTProvider(ABC):
    """Abstract base class for all Speech-to-Text providers."""

    @abstractmethod
    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        mime_type: str = "audio/wav",
        language_code: Optional[str] = None
    ) -> STTResult:
        """
        Transcribes the provided audio bytes into text.
        
        Args:
            audio_bytes: Raw binary content of the audio recording.
            filename: Original or synthesized filename (e.g., 'recording.webm').
            mime_type: MIME type of the audio (e.g., 'audio/webm', 'audio/wav').
            language_code: Optional language hint (e.g., 'hi-IN', 'en-IN').
            
        Returns:
            STTResult: Structured transcription result with timing metrics.
            
        Raises:
            ValueError: If input audio is empty or invalid.
            TimeoutError: If transcription provider times out.
            RuntimeError: If provider returns an error or fails.
        """
        pass
