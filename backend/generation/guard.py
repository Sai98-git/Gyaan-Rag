import re
import logging
from typing import List, Dict, Any
from backend.core.config import settings

logger = logging.getLogger(__name__)

def validate_generation(query: str, context: List[Dict[str, Any]], answer_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Implements a deterministic hallucination and grounding guard.
    
    Checks:
    1. Empty context.
    2. Retrieval confidence thresholding (max retrieval score < MIN_RETRIEVAL_SCORE).
    3. Lexical word overlap between generated answer and context.
    
    If any check fails, immediately overrides the answer with a safe, 
    predefined refusal fallback.
    """
    safe_fallback = "I don't have enough information in the retrieved sources to answer that reliably."
    
    # Check 1: Empty context
    if not context:
        logger.warning("Grounding guard triggered: Empty context.")
        return {
            "answer": safe_fallback,
            "sources": [],
            "provider": answer_dict.get("provider", "unknown"),
            "guard_triggered": True,
            "guard_reason": "Empty context"
        }
        
    # Check 2: Max retrieval score check (method-aware)
    max_score = max(chunk.get("score", 0.0) for chunk in context)
    retrieval_method = context[0].get("retrieval_method", "dense") if context else "dense"
    effective_threshold = settings.MIN_RETRIEVAL_SCORE if retrieval_method == "dense" else 1.0

    logger.info(
        f"Grounding guard: max_score={max_score:.4f}, method='{retrieval_method}', "
        f"threshold={effective_threshold:.4f}"
    )

    if max_score < effective_threshold:
        logger.warning(
            f"Grounding guard triggered: Max retrieval score ({max_score:.4f}) "
            f"is below threshold ({effective_threshold:.4f}) for method '{retrieval_method}'."
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
            "provider": answer_dict.get("provider", "unknown"),
            "guard_triggered": True,
            "guard_reason": f"Low retrieval confidence: {max_score:.4f} < {effective_threshold:.4f}"
        }
        
    # Check 3: Lexical overlap check (to ensure answer references retrieved text content)
    answer_text = answer_dict["answer"]
    
    # Common refusal indicators to bypass overlap checks for valid refusals
    refusal_keywords = {
        "sorry", "insufficient", "पर्याप्त", "जानकारी", "सॉरी", "reliably", 
        "not have enough information", "डोंट हैव इनफ"
    }
    is_refusal = any(kw in answer_text.lower() for kw in refusal_keywords)
    
    if not is_refusal:
        # Extract word sets from context and answer
        context_words = set()
        for chunk in context:
            words = [
                w.strip() 
                for w in re.sub(r'[।॥\|!\?\.,;:\(\)"\'\-\n\r\t]', ' ', chunk["text"].lower()).split() 
                if len(w) > 2
            ]
            context_words.update(words)
            
        answer_words = [
            w.strip() 
            for w in re.sub(r'[।॥\|!\?\.,;:\(\)"\'\-\n\r\t]', ' ', answer_text.lower()).split() 
            if len(w) > 2
        ]
        
        # Calculate overlap of terms
        overlap = [w for w in answer_words if w in context_words]
        
        # If less than 1 keyword overlaps between context and non-refusal answer, trigger fallback
        if len(overlap) < 1:
            logger.warning("Grounding guard triggered: Generated answer has zero lexical overlap with retrieved context.")
            return {
                "answer": safe_fallback,
                "sources": answer_dict["sources"],
                "provider": answer_dict.get("provider", "unknown"),
                "guard_triggered": True,
                "guard_reason": "Zero lexical overlap with context (potential hallucination)"
            }
            
    # If all checks pass, return original generated dict
    return answer_dict
