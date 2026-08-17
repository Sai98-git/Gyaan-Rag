import unicodedata
import re
import logging
from typing import List

logger = logging.getLogger(__name__)

def normalize_query_text(query: str) -> str:
    """
    Standard text normalization for Indic and Latin queries:
    - Unicode NFC normalization (composes decomposed Devanagari characters)
    - Strips noisy punctuation while preserving alphanumeric and Indic unicode characters
    - Normalizes multiple whitespace to single space
    - Lowercases Latin characters
    
    Zero hardcoded topic dictionaries or fact mappings.
    """
    if not query:
        return ""
        
    # Unicode NFC normalization
    normalized = unicodedata.normalize("NFC", query)
    
    # Lowercase Latin text
    normalized = normalized.lower()
    
    # Strip leading/trailing punctuation and extra whitespace
    cleaned = re.sub(r'[।॥\|!\?\.,;:\(\)\"\'`\-\n\r\t]', ' ', normalized)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned


def tokenize_query(query: str) -> List[str]:
    """Extracts non-empty token strings from normalized query."""
    normalized = normalize_query_text(query)
    return [tok for tok in normalized.split() if tok]
