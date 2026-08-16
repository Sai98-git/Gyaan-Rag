from typing import List, Dict, Any
from backend.ingestion.chunkers.base import BaseChunker
from backend.ingestion.metadata import DatasetRecord
from backend.core.config import settings

class SlidingWindowChunker(BaseChunker):
    """
    Sliding-window chunking strategy (Strategy B).
    
    Splits the translated passages text into chunks, 
    preserving sentence boundaries, tracking selection flags, 
    and enforcing configurable size and overlap.
    """
    
    def chunk_record(self, record: DatasetRecord) -> List[Dict[str, Any]]:
        translated_passages = record.passages.Translated_passages
        
        # Split each passage into sentences separately to track parent selection metadata
        sentences = []
        sentence_metadata = []
        
        for idx, text in enumerate(translated_passages):
            clean_pass = text.strip()
            if not clean_pass:
                continue
                
            is_sel = (
                record.passages.is_selected[idx] 
                if idx < len(record.passages.is_selected) 
                else 0
            )
            
            pass_sentences = self.split_into_sentences(clean_pass)
            for s in pass_sentences:
                sentences.append(s)
                sentence_metadata.append({
                    "is_selected": is_sel,
                    "passage_idx": idx
                })
                
        if not sentences:
            return []
            
        chunks = []
        start_idx = 0
        position = 0
        n_sentences = len(sentences)
        
        while start_idx < n_sentences:
            curr_len = 0
            end_idx = start_idx
            
            # Pack sentences until CHUNK_SIZE limit is reached
            while end_idx < n_sentences:
                sent_len = len(sentences[end_idx])
                if end_idx == start_idx:
                    curr_len += sent_len
                    end_idx += 1
                elif curr_len + 1 + sent_len <= settings.CHUNK_SIZE:
                    curr_len += 1 + sent_len  # Adding delimiter length
                    end_idx += 1
                else:
                    break
            
            # Join sentences into chunk text
            chunk_text = " ".join(sentences[start_idx:end_idx]).strip()
            
            # Enforce MIN_CHUNK_SIZE constraints unless it's the only text block
            if len(chunk_text) >= settings.MIN_CHUNK_SIZE or (start_idx == 0 and end_idx == n_sentences):
                # Calculate aggregated is_selected and passage positions
                chunk_is_selected = max(sentence_metadata[k]["is_selected"] for k in range(start_idx, end_idx))
                passage_positions = list(set(sentence_metadata[k]["passage_idx"] for k in range(start_idx, end_idx)))
                
                # Construct metadata
                metadata = {
                    "strategy": "sliding_window",
                    "position": position,
                    "language": record.target_lang,
                    "dataset_split": settings.DATA_SPLIT,
                    "dataset_language": settings.DATA_LANGUAGE,
                    "parent_query_id": record.query_id,
                    "start_sentence_idx": start_idx,
                    "end_sentence_idx": end_idx - 1,
                    "is_selected": chunk_is_selected,
                    "original_passage_positions": passage_positions
                }
                
                chunk_id = f"{record.query_id}_sliding_{position}"
                doc_id = str(record.query_id)
                
                chunk = self.create_chunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    query_id=record.query_id,
                    text=chunk_text,
                    strategy="sliding_window",
                    position=position,
                    metadata=metadata
                )
                chunks.append(chunk)
                position += 1
                
            # Determine the next start index based on the overlap configuration
            next_start_idx = end_idx
            for k in range(end_idx - 1, start_idx, -1):
                # Calculate the character length of the overlapping sentences
                overlap_len = sum(len(s) + 1 for s in sentences[k:end_idx]) - 1
                if overlap_len <= settings.CHUNK_OVERLAP:
                    next_start_idx = k
                else:
                    break
            
            # Safety check: guarantee forward progress
            if next_start_idx <= start_idx:
                next_start_idx = start_idx + 1
                
            start_idx = next_start_idx
            
        return chunks
