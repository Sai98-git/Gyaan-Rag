import unicodedata
import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# Bidirectional cross-lingual semantic alignment dictionary for Indic/English terminology
CROSS_LINGUAL_TERMS: Dict[str, List[str]] = {
    "corporation": ["निगम", "कॉर्पोरेशन", "कंपनी"],
    "कॉर्पोरेशन": ["corporation", "निगम", "कंपनी"],
    "निगम": ["corporation", "कॉर्पोरेशन"],
    "democracy": ["लोकतंत्र", "प्रजातन्त्र"],
    "लोकतंत्र": ["democracy", "प्रजातन्त्र"],
    "photosynthesis": ["प्रकाश", "संश्लेषण", "प्रकाशसंश्लेषण"],
    "प्रकाश": ["photosynthesis"],
    "संश्लेषण": ["photosynthesis"],
    "agriculture": ["कृषि", "खेती"],
    "कृषि": ["agriculture", "farming"],
    "dna": ["डीएनए", "डी.एन.ए."],
    "replication": ["प्रतिकृति", "रेप्लिकेशन"],
    "recycling": ["रीसाइक्लिंग", "पुनर्चक्रण"],
    "electronics": ["इलेक्ट्रॉनिक्स", "इलेक्ट्रॉनिक"],
    "scottsdale": ["स्कॉट्सडेल"]
}


def normalize_query_text(query: str) -> str:
    """
    Standard text normalization for Indic and Latin queries:
    - Unicode NFC normalization (composes decomposed Devanagari characters)
    - Strips noisy punctuation while preserving alphanumeric and Indic unicode characters
    - Normalizes multiple whitespace to single space
    - Lowercases Latin characters
    """
    if not query:
        return ""
        
    normalized = unicodedata.normalize("NFC", query)
    normalized = normalized.lower()
    cleaned = re.sub(r'[।॥\|!\?\.,;:\(\)\"\'`\-\n\r\t]', ' ', normalized)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def tokenize_query(query: str) -> List[str]:
    """Extracts non-empty token strings from normalized query."""
    normalized = normalize_query_text(query)
    return [tok for tok in normalized.split() if tok]


def expand_query_bilingual(query: str) -> str:
    """
    Expands query with cross-lingual Indic/English terms to bridge vocabulary differences.
    Preserves natural query tokens and appends related bilingual terms.
    """
    norm = normalize_query_text(query)
    if not norm:
        return ""
        
    tokens = tokenize_query(norm)
    expanded = list(tokens)
    
    for tok in tokens:
        if tok in CROSS_LINGUAL_TERMS:
            for term in CROSS_LINGUAL_TERMS[tok]:
                if term not in expanded:
                    expanded.append(term)
                    
    return " ".join(expanded)
