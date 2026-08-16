import re
from typing import List, Dict, Any, Set
from backend.ingestion.chunkers.base import BaseChunker
from backend.ingestion.metadata import DatasetRecord
from backend.core.config import settings

class SemanticChunker(BaseChunker):
    """
    Semantic/Structure-aware chunking strategy (Strategy C).
    
    A lightweight, offline-safe semantic chunker.
    1. Keeps passage boundaries where possible.
    2. Sub-splits any passage exceeding MAX_CHUNK_SIZE using sentence boundaries.
    3. Merges adjacent passages if they share semantic context (measured via Jaccard 
       word similarity above a threshold) and fit within MAX_CHUNK_SIZE.
    """

    def tokenize_words(self, text: str) -> Set[str]:
        """Normalizes and extracts a set of words from a text string."""
        if not text:
            return set()
        # Strip common punctuation and split by spaces
        clean_text = re.sub(r'[।॥\|!\?\.,;:\(\)"\'\-]', ' ', text.lower())
        words = {w.strip() for w in clean_text.split() if len(w.strip()) > 1}
        return words

    def compute_jaccard_similarity(self, text_a: str, text_b: str) -> float:
        """Computes Jaccard word similarity between two text strings."""
        words_a = self.tokenize_words(text_a)
        words_b = self.tokenize_words(text_b)
        
        if not words_a or not words_b:
            return 0.0
            
        intersection = len(words_a.intersection(words_b))
        union = len(words_a.union(words_b))
        return intersection / union

    def chunk_record(self, record: DatasetRecord) -> List[Dict[str, Any]]:
        translated_passages = record.passages.Translated_passages
        
        # Step 1: Pre-process passages: split any that exceed MAX_CHUNK_SIZE
        pre_processed: List[Dict[str, Any]] = []
        for idx, text in enumerate(translated_passages):
            clean_text = text.strip()
            if not clean_text:
                continue
                
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

            # If passage is small enough, keep as a single block
            if len(clean_text) <= settings.MAX_CHUNK_SIZE:
                pre_processed.append({
                    "text": clean_text,
                    "original_pos": idx,
                    "is_selected": is_selected,
                    "english_passage": english_passage
                })
            else:
                # Split large passage into sentences
                sentences = self.split_into_sentences(clean_text)
                temp_text = ""
                for s in sentences:
                    # Accumulate sentences to form small sub-chunks
                    if not temp_text:
                        temp_text = s
                    elif len(temp_text) + 1 + len(s) <= settings.MAX_CHUNK_SIZE:
                        temp_text += " " + s
                    else:
                        pre_processed.append({
                            "text": temp_text,
                            "original_pos": idx,
                            "is_selected": is_selected,
                            "english_passage": english_passage
                        })
                        temp_text = s
                if temp_text:
                    pre_processed.append({
                        "text": temp_text,
                        "original_pos": idx,
                        "is_selected": is_selected,
                        "english_passage": english_passage
                    })

        if not pre_processed:
            return []

        # Step 2: Merge adjacent sub-passages based on Jaccard similarity and size limits
        merged_chunks: List[Dict[str, Any]] = []
        current = pre_processed[0]
        current_text = current["text"]
        current_original_positions = [current["original_pos"]]
        current_is_selected = [current["is_selected"]]
        current_english_passages = [current["english_passage"]]

        similarity_threshold = 0.08  # Default Jaccard similarity threshold for merging

        for i in range(1, len(pre_processed)):
            candidate = pre_processed[i]
            candidate_text = candidate["text"]

            # Check if merged size fits within MAX_CHUNK_SIZE
            fits_size = len(current_text) + 2 + len(candidate_text) <= settings.MAX_CHUNK_SIZE
            
            # Compute similarity
            similarity = self.compute_jaccard_similarity(current_text, candidate_text)
            is_similar = similarity >= similarity_threshold

            if fits_size and is_similar:
                # Merge candidate into current chunk
                current_text += "\n\n" + candidate_text
                current_original_positions.append(candidate["original_pos"])
                current_is_selected.append(candidate["is_selected"])
                current_english_passages.append(candidate["english_passage"])
            else:
                # Yield current accumulated chunk
                merged_chunks.append({
                    "text": current_text,
                    "original_positions": current_original_positions,
                    "is_selected": current_is_selected,
                    "english_passages": current_english_passages
                })
                # Start new chunk
                current = candidate
                current_text = current["text"]
                current_original_positions = [current["original_pos"]]
                current_is_selected = [current["is_selected"]]
                current_english_passages = [current["english_passage"]]

        # Yield last chunk
        merged_chunks.append({
            "text": current_text,
            "original_positions": current_original_positions,
            "is_selected": current_is_selected,
            "english_passages": current_english_passages
        })

        # Step 3: Format chunks into strict schema
        chunks = []
        for idx, item in enumerate(merged_chunks):
            metadata = {
                "original_positions": item["original_positions"],
                "language": record.target_lang,
                "dataset_split": settings.DATA_SPLIT,
                "dataset_language": settings.DATA_LANGUAGE,
                "parent_query_id": record.query_id,
                "is_selected": max(item["is_selected"]),  # Selected if any merged passage was selected
                "english_passages": item["english_passages"]
            }
            
            chunk_id = f"{record.query_id}_semantic_{idx}"
            
            chunk = self.create_chunk(
                chunk_id=chunk_id,
                document_id=str(record.query_id),
                query_id=record.query_id,
                text=item["text"],
                strategy="semantic",
                position=idx,
                metadata=metadata
            )
            chunks.append(chunk)

        return chunks
