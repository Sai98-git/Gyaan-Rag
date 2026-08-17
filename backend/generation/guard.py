import re
import logging
from typing import List, Dict, Any, Optional
from backend.core.config import settings

logger = logging.getLogger(__name__)

HINDI_ABSTENTION = "मुझे उपलब्ध स्रोतों में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं मिली।"
ENGLISH_ABSTENTION = "I don't have enough information in the retrieved sources to answer that reliably."


def is_devanagari(text: str) -> bool:
    """Checks if text contains Devanagari script characters."""
    return bool(re.search(r'[\u0900-\u097f]', text))


def get_localized_abstention(query: str) -> str:
    """Returns a natural abstention message in the user's query language."""
    if is_devanagari(query):
        return HINDI_ABSTENTION
    return ENGLISH_ABSTENTION


def check_pre_retrieval_guard(query: str, context: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Fast pre-generation check.
    If context is empty or maximum retrieval score is below threshold,
    immediately returns an abstention dictionary in ~1ms without invoking the LLM.
    """
    safe_fallback = get_localized_abstention(query)
    
    # 1. Empty context check
    if not context:
        logger.info("[Pre-Gen Guard] Empty context -> Fast abstention.")
        return {
            "answer": safe_fallback,
            "sources": [],
            "guard_triggered": True,
            "guard_reason": "Empty context"
        }

    # 2. Maximum retrieval score threshold check
    max_score = max(chunk.get("score", 0.0) for chunk in context)
    retrieval_method = context[0].get("retrieval_method", "dense") if context else "dense"
    effective_threshold = settings.MIN_RETRIEVAL_SCORE if retrieval_method == "dense" else 1.0

    if max_score < effective_threshold:
        logger.info(
            f"[Pre-Gen Guard] Low confidence (score={max_score:.4f} < {effective_threshold:.4f}) -> Fast abstention."
        )
        return {
            "answer": safe_fallback,
            "sources": [
                {
                    "chunk_id": c["chunk_id"],
                    "score": c.get("score", 0.0),
                    "preview": c.get("text", "")[:200].strip(),
                    "metadata": c.get("metadata", {})
                }
                for c in context
            ],
            "guard_triggered": True,
            "guard_reason": f"Low retrieval confidence: {max_score:.4f} < {effective_threshold:.4f}"
        }

    return None


def validate_generation(query: str, context: List[Dict[str, Any]], answer_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-generation validation:
    1. Ensures lexical word overlap between generated answer and context passages.
    2. Enforces localized refusal fallback if hallucination or zero overlap is detected.
    """
    safe_fallback = get_localized_abstention(query)
    
    # 1. Empty context fallback
    if not context:
        return {
            "answer": safe_fallback,
            "sources": [],
            "provider": answer_dict.get("provider", "unknown"),
            "guard_triggered": True,
            "guard_reason": "Empty context"
        }
        
    answer_text = answer_dict.get("answer", "").strip()
    
    # 2. Refusal keywords bypass
    refusal_keywords = {
        "sorry", "insufficient", "पर्याप्त", "जानकारी", "सॉरी", "reliably", 
        "not have enough information", "डोंट हैव इनफ", "उपलब्ध स्रोतों"
    }
    is_refusal = any(kw in answer_text.lower() for kw in refusal_keywords)
    
    if not is_refusal:
        # Extract meaningful terms from context and answer
        context_words = set()
        for chunk in context:
            words = [
                w.strip() 
                for w in re.sub(r'[।॥\|!\?\.,;:\(\)"\'\-\n\r\t]', ' ', chunk.get("text", "").lower()).split() 
                if len(w) > 2
            ]
            context_words.update(words)
            
        answer_words = [
            w.strip() 
            for w in re.sub(r'[।॥\|!\?\.,;:\(\)"\'\-\n\r\t]', ' ', answer_text.lower()).split() 
            if len(w) > 2
        ]
        
        overlap = [w for w in answer_words if w in context_words]
        
        # If zero overlap detected for non-refusal answer, trigger fallback
        if len(overlap) < 1:
            logger.warning("Grounding guard: Zero lexical overlap with context (potential hallucination).")
            return {
                "answer": safe_fallback,
                "sources": answer_dict.get("sources", []),
                "provider": answer_dict.get("provider", "unknown"),
                "guard_triggered": True,
                "guard_reason": "Zero lexical overlap with context (potential hallucination)"
            }
            
    return answer_dict
