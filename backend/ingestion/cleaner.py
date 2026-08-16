import logging
from typing import Dict, Any, Optional
from backend.ingestion.metadata import DatasetRecord, TranslationMeta, PassagesGroup

logger = logging.getLogger(__name__)

def clean_string(val: Any) -> str:
    """Helper to clean string fields: strips whitespace and handles None values."""
    if val is None:
        return ""
    return str(val).strip()

def normalize_row(raw_row: Dict[str, Any]) -> Optional[DatasetRecord]:
    """
    Accepts a raw record dictionary from a Parquet file row and cleans,
    normalizes, and validates it against the DatasetRecord schema.
    
    Returns:
        Optional[DatasetRecord]: The validated record, or None if validation fails.
    """
    try:
        # 1. Validate mandatory fields
        query_id = raw_row.get("query_id")
        if query_id is None:
            logger.warning("Skipping record: missing 'query_id'")
            return None
        
        query = clean_string(raw_row.get("query"))
        if not query:
            logger.warning(f"Skipping record query_id={query_id}: empty 'query'")
            return None
            
        target_lang = clean_string(raw_row.get("target_lang"))
        if not target_lang:
            # Fallback target language from source or default if missing
            target_lang = "hi"
            
        # 2. Extract nested 'meta' struct
        raw_meta = raw_row.get("meta") or {}
        if isinstance(raw_meta, dict):
            meta = TranslationMeta(
                model_name=clean_string(raw_meta.get("model_name")),
                temperature=float(raw_meta.get("temperature") or 0.0),
                max_tokens=int(raw_meta.get("max_tokens") or 0),
                top_p=float(raw_meta.get("top_p") or 1.0),
                frequency_penalty=float(raw_meta.get("frequency_penalty") or 0.0),
                presence_penalty=float(raw_meta.get("presence_penalty") or 0.0)
            )
        else:
            meta = TranslationMeta()

        # 3. Extract nested 'passages' struct
        raw_passages = raw_row.get("passages") or {}
        is_selected = []
        english_passages = []
        translated_passages = []
        
        if isinstance(raw_passages, dict):
            # PyArrow lists can be numpy arrays or lists of elements
            is_selected_raw = raw_passages.get("is_selected")
            if is_selected_raw is not None:
                is_selected = [int(x) for x in is_selected_raw]
                
            eng_pass_raw = raw_passages.get("English_passages")
            if eng_pass_raw is not None:
                english_passages = [clean_string(x) for x in eng_pass_raw]
                
            trans_pass_raw = raw_passages.get("Translated_passages")
            if trans_pass_raw is not None:
                translated_passages = [clean_string(x) for x in trans_pass_raw]
                
        passages = PassagesGroup(
            is_selected=is_selected,
            English_passages=english_passages,
            Translated_passages=translated_passages
        )

        # 4. Construct and validate DatasetRecord
        record = DatasetRecord(
            query_id=int(query_id),
            query_type=clean_string(raw_row.get("query_type")),
            query=query,
            Answer=clean_string(raw_row.get("Answer")),
            Eng_Query=clean_string(raw_row.get("Eng_Query")),
            Eng_Answer=clean_string(raw_row.get("Eng_Answer")),
            source_lang=clean_string(raw_row.get("source_lang") or "en"),
            target_lang=target_lang,
            meta=meta,
            passages=passages
        )
        return record
        
    except Exception as e:
        logger.error(f"Error normalizing row {raw_row.get('query_id')}: {e}")
        return None
