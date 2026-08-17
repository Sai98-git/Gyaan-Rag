import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Common speech filler words and noise tokens to clean up
SPEECH_FILLERS = [
    r"\b(um|uh|er|ah|like|you know|hmm|haan|achha|matlab)\b",
]

def normalize_voice_query(raw_transcript: str) -> str:
    """
    Cleans and normalizes a voice transcript for downstream RAG retrieval.
    
    1. Trims leading/trailing whitespace.
    2. Strips repeated punctuation or trailing audio artifacts (e.g., '...', '???').
    3. Normalizes whitespace (collapsing multiple spaces and tabs).
    4. Preserves Devanagari script and punctuation intact.
    
    Args:
        raw_transcript: Raw string returned by the STT provider.
        
    Returns:
        Cleaned, normalized query string ready for vector embedding and retrieval.
    """
    if not raw_transcript:
        return ""
        
    text = raw_transcript.strip()
    
    # Remove surrounding quotation marks if STT provider wrapped the entire text
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
        
    # Collapse multiple consecutive whitespace characters
    text = re.sub(r'\s+', ' ', text)
    
    # Clean excessive trailing punctuation while preserving single ? or Hindi purna viram (।)
    text = re.sub(r'[\.\!\?]{2,}$', '?', text)
    
    # Strip non-printable control characters
    text = "".join(ch for ch in text if ch.isprintable() or ch in ('\n', ' '))
    
    clean_text = text.strip()
    logger.debug(f"Normalized transcript: '{raw_transcript}' -> '{clean_text}'")
    return clean_text
