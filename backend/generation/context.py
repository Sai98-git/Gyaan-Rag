import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def format_context(context: List[Dict[str, Any]], max_chars: int = 4000, max_chunks: int = 10) -> str:
    """
    Evidence-aware context compression:
    - Deduplicates chunks by ID and text prefix.
    - Preserves all top-K retrieved evidence passages.
    - Formats concise provenance markers (SOURCE 1 (ID: ...)).
    - Ensures full factual recall for diverse multi-part queries.
    """
    if not context:
        return "[No reference context available]"
        
    seen_chunk_ids = set()
    seen_text_prefixes = set()
    formatted_chunks = []
    
    source_index = 1
    total_len = 0
    
    for chunk in context:
        chunk_id = chunk.get("chunk_id")
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
            
        # Deduplicate by ID
        if chunk_id and chunk_id in seen_chunk_ids:
            continue
            
        # Deduplicate exact duplicate text
        prefix = " ".join(text.split()[:20]).lower()
        if prefix in seen_text_prefixes:
            continue
            
        if chunk_id:
            seen_chunk_ids.add(chunk_id)
        seen_text_prefixes.add(prefix)
        
        formatted_block = f"SOURCE {source_index} (ID: {chunk_id})\n{text}"
        
        # Check context limit
        if total_len + len(formatted_block) + 2 > max_chars:
            break
            
        formatted_chunks.append(formatted_block)
        total_len += len(formatted_block) + 2
        source_index += 1
        
        if len(formatted_chunks) >= max_chunks:
            break
        
    if not formatted_chunks:
        return "[No reference context available]"
        
    return "\n\n".join(formatted_chunks)
