import re
import logging
from typing import List, Dict, Any, Optional
from backend.core.config import settings
from backend.retrieval.normalizer import expand_indic_query

logger = logging.getLogger(__name__)

HINDI_ABSTENTION = "मुझे उपलब्ध स्रोतों में इस प्रश्न का विश्वसनीय उत्तर देने के लिए पर्याप्त जानकारी नहीं मिली।"
ENGLISH_ABSTENTION = "I don't have enough information in the retrieved sources to answer that reliably."

GENERIC_STOP_WORDS = {
    "what", "is", "a", "an", "the", "of", "in", "on", "at", "to", "for", "with", 
    "about", "how", "who", "where", "when", "why", "which", "can", "you", "tell", "me",
    "kya", "hai", "hain", "hota", "hoti", "hote", "ka", "ke", "ki", "ko", "se", "mein", 
    "me", "par", "batao", "kise", "kaise", "kyun", "kaun", "bhi", "aur", "ya",
    "क्या", "है", "हैं", "होता", "होती", "होते", "का", "के", "की", "को", "से", "में", 
    "पर", "बताओ", "किसे", "कैसे", "क्यों", "कौन", "भी", "और", "या", "एक", "यह", "वह"
}


def is_devanagari(text: str) -> bool:
    """Checks if text contains Devanagari script characters."""
    return bool(re.search(r'[\u0900-\u097f]', text))


def get_localized_abstention(query: str) -> str:
    """Returns a natural abstention message in the user's query language."""
    if is_devanagari(query):
        return HINDI_ABSTENTION
    return ENGLISH_ABSTENTION


def extract_key_terms(text: str) -> set:
    """Extracts non-stopword tokens from a text string."""
    clean = re.sub(r'[।॥\|!\?\.,;:\(\)\"\'\-\n\r\t]', ' ', text.lower())
    words = [w.strip() for w in clean.split() if len(w.strip()) > 1]
    return {w for w in words if w not in GENERIC_STOP_WORDS}


def check_pre_retrieval_guard(query: str, context: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Tier 1 Pre-Generation Evidence Guard:
    Checks if any evidence passages were retrieved for the query.
    If context is completely empty or top score is 0.0, safely abstains in ~1ms.
    """
    safe_fallback = get_localized_abstention(query)
    
    if not context:
        logger.info(f"[Pre-Gen Guard] Empty context for query '{query}' -> Abstaining.")
        return {
            "answer": safe_fallback,
            "sources": [],
            "guard_triggered": True,
            "guard_reason": "Empty context"
        }

    top_score = context[0].get("score", 0.0)
    if top_score <= 0.0:
        logger.info(f"[Pre-Gen Guard] Zero score for query '{query}' -> Abstaining.")
        return {
            "answer": safe_fallback,
            "sources": [],
            "guard_triggered": True,
            "guard_reason": "Zero retrieval relevance"
        }

    return None


def validate_generation(query: str, context: List[Dict[str, Any]], answer_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tier 2 Post-Generation Grounding Guard:
    Validates that the generated answer is grounded in retrieved context and not a hallucination.
    """
    safe_fallback = get_localized_abstention(query)
    
    if not context:
        return {
            "answer": safe_fallback,
            "sources": [],
            "provider": answer_dict.get("provider", "unknown"),
            "guard_triggered": True,
            "guard_reason": "Empty context"
        }
        
    answer_text = answer_dict.get("answer", "").strip()
    
    # Refusal keywords detection
    refusal_keywords = {
        "sorry", "insufficient", "पर्याप्त", "जानकारी", "सॉरी", "reliably", 
        "not have enough information", "डोंट हैव इनफ", "उपलब्ध स्रोतों", "विश्वसनीय उत्तर",
        "i don't have enough", "does not contain"
    }
    is_refusal = any(kw in answer_text.lower() for kw in refusal_keywords)
    
    if is_refusal:
        return {
            "answer": safe_fallback,
            "sources": answer_dict.get("sources", context),
            "provider": answer_dict.get("provider", "unknown"),
            "guard_triggered": True,
            "guard_reason": "Insufficient evidence in context"
        }
    
    # Cross-lingual lexical grounding check
    context_text = " ".join(chunk.get("text", "") for chunk in context)
    expanded_context = expand_indic_query(context_text)
    context_words = set(
        w.strip() 
        for w in re.sub(r'[।॥\|!\?\.,;:\(\)\"\'\-\n\r\t]', ' ', expanded_context.lower()).split() 
        if len(w) > 2 and w not in GENERIC_STOP_WORDS
    )
    
    expanded_answer = expand_indic_query(answer_text)
    answer_words = set(
        w.strip() 
        for w in re.sub(r'[।॥\|!\?\.,;:\(\)\"\'\-\n\r\t]', ' ', expanded_answer.lower()).split() 
        if len(w) > 2 and w not in GENERIC_STOP_WORDS
    )
    
    overlap = answer_words.intersection(context_words)
    
    if len(overlap) < 1:
        logger.warning("Grounding guard: Zero lexical overlap with context (potential hallucination).")
        return {
            "answer": safe_fallback,
            "sources": answer_dict.get("sources", context),
            "provider": answer_dict.get("provider", "unknown"),
            "guard_triggered": True,
            "guard_reason": "Zero lexical overlap with context (potential hallucination)"
        }
        
    return answer_dict
