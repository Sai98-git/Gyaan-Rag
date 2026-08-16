from typing import List, Dict, Any
from backend.ingestion.chunkers.base import BaseChunker
from backend.ingestion.metadata import DatasetRecord
from backend.core.config import settings

class PassageChunker(BaseChunker):
    """
    Passage-aware chunking strategy (Strategy A).
    
    Preserves naturally occurring passage boundaries in MSMARCO-XI, 
    avoiding slicing text in arbitrary character lengths.
    """
    
    def chunk_record(self, record: DatasetRecord) -> List[Dict[str, Any]]:
        chunks = []
        translated_passages = record.passages.Translated_passages
        
        for idx, text in enumerate(translated_passages):
            clean_text = text.strip()
            if not clean_text:
                continue  # Skip empty passages
                
            # Safely check parallel lists to avoid IndexErrors
            is_selected = (
                record.passages.is_selected[idx] 
                if idx < len(record.passages.is_selected) 
                else 0
            )
            english_passage = (
                record.passages.English_passages[idx] 
                if idx < len(record.passages.English_passages) 
                else ""
            )
            
            # Construct metadata
            metadata = {
                "source_passage_id": idx,
                "original_passage_position": idx,
                "language": record.target_lang,
                "dataset_split": settings.DATA_SPLIT,
                "dataset_language": settings.DATA_LANGUAGE,
                "parent_query_id": record.query_id,
                "is_selected": is_selected,
                "english_passage": english_passage
            }
            
            # Create chunk identifier
            chunk_id = f"{record.query_id}_passage_{idx}"
            doc_id = str(record.query_id)
            
            chunk = self.create_chunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                query_id=record.query_id,
                text=clean_text,
                strategy="passage",
                position=idx,
                metadata=metadata
            )
            chunks.append(chunk)
            
        return chunks
