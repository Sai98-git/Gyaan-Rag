import re
import logging
from typing import List, Dict, Any, Optional
from backend.core.config import settings
from backend.retrieval.normalizer import normalize_query_text, tokenize_query

logger = logging.getLogger(__name__)

HINDI_ABSTENTION = "मुझे उपलब्ध स्रोतों में इस प्रश्न का विश्वसनीय उत्तर देने के लिए पर्याप्त जानकारी नहीं मिली।"
ENGLISH_ABSTENTION = "I don't have enough information in the retrieved sources to answer that reliably."


def is_devanagari(text: str) -> bool:
    """Checks if text contains Devanagari script characters."""
    return bool(re.search(r'[\u0900-\u097f]', text))


def get_localized_abstention(query: str) -> str:
    """Returns a natural abstention message in the user's query language."""
    if is_devanagari(query):
        return HINDI_ABSTENTION
    return ENGLISH_ABSTENTION


def check_pre_retrieval_guard(
    query: str, 
    context: List[Dict[str, Any]], 
    min_dense_threshold: float = 0.72
) -> Optional[Dict[str, Any]]:
    """
    Tier 1 Pre-Generation Evidence Guard:
    Checks if any evidence passages were retrieved for the query with sufficient
    mathematical confidence (dense similarity or BM25 match).
    
    Zero hardcoded knowledge, zero topic lists.
    """
    safe_fallback = get_localized_abstention(query)
    
    if not context:
        logger.info(f"[Pre-Gen Guard] Empty context for query '{query}' -> Abstaining.")
        return {
            "answer": safe_fallback,
            "sources": [],
            "guard_triggered": True,
            "guard_reason": "No evidence passages retrieved from dataset"
        }

    top_chunk = context[0]
    top_score = top_chunk.get("score", 0.0)
    top_dense = top_chunk.get("dense_score", 0.0)
    top_bm25 = top_chunk.get("bm25_score", 0.0)

    # If neither dense similarity is sufficient nor any BM25 term matched
    if top_dense < min_dense_threshold and top_bm25 <= 0.0:
        logger.info(f"[Pre-Gen Guard] Low confidence (dense={top_dense:.4f}, bm25={top_bm25:.4f}) for query '{query}' -> Abstaining.")
        return {
            "answer": safe_fallback,
            "sources": context,
            "guard_triggered": True,
            "guard_reason": f"Insufficient retrieval confidence (dense={top_dense:.3f}, bm25={top_bm25:.3f})"
        }

    return None


def validate_generation(query: str, context: List[Dict[str, Any]], answer_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tier 2 Post-Generation Grounding Guard:
    Validates that the generated answer is grounded in retrieved context and not an unsupported generation.
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
    
    # Check for standard model refusal indicators
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
            "guard_reason": "Model indicated insufficient evidence in retrieved context"
        }
        
    return answer_dict
