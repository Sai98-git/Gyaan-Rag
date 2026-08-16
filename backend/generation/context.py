import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def format_context(context: List[Dict[str, Any]], max_chars: int = 4000) -> str:
    """
    Deduplicates retrieved chunks, structures them cleanly, 
    and limits the context size to fit within the configured context limit.
    
    Format:
    SOURCE 1 (ID: <chunk_id>)
    [chunk text]
    
    SOURCE 2 (ID: <chunk_id>)
    ...
    """
    if not context:
        return "[No reference context available]"
        
    seen_chunk_ids = set()
    formatted_chunks = []
    
    source_index = 1
    total_len = 0
    
    for chunk in context:
        chunk_id = chunk["chunk_id"]
        
        # Deduplicate
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        
        text = chunk["text"].strip()
        formatted_block = f"SOURCE {source_index} (ID: {chunk_id})\n{text}"
        
        # Check context limit
        if total_len + len(formatted_block) + 2 > max_chars:
            logger.warning(f"Context length limit ({max_chars} chars) reached. Truncating remainder.")
            break
            
        formatted_chunks.append(formatted_block)
        total_len += len(formatted_block) + 2  # including delimiter
        source_index += 1
        
    return "\n\n".join(formatted_chunks)
