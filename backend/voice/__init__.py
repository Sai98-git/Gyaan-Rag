import logging
from typing import Optional
from backend.core.config import settings
from backend.voice.base import BaseSTTProvider, STTResult
from backend.voice.sarvam_stt import SarvamSTTProvider
from backend.voice.elevenlabs_stt import ElevenLabsSTTProvider
from backend.voice.cleaner import normalize_voice_query

logger = logging.getLogger(__name__)

def get_stt_provider() -> BaseSTTProvider:
    """
    Factory to retrieve the configured Speech-to-Text provider.
    
    Defaults to Sarvam STT ('sarvam') for Indic-first audio processing.
    """
    provider_name = (settings.STT_PROVIDER or "sarvam").lower().strip()
    
    if provider_name == "elevenlabs":
        logger.info("Initializing ElevenLabs STT provider...")
        return ElevenLabsSTTProvider()
    
    logger.info("Initializing Sarvam STT provider (Indic-first)...")
    return SarvamSTTProvider()

__all__ = [
    "BaseSTTProvider",
    "STTResult",
    "SarvamSTTProvider",
    "ElevenLabsSTTProvider",
    "get_stt_provider",
    "normalize_voice_query"
]
