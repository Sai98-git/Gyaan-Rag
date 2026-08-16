import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from backend.ingestion.metadata import DatasetRecord

class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""
    
    @abstractmethod
    def chunk_record(self, record: DatasetRecord) -> List[Dict[str, Any]]:
        """
        Processes a single DatasetRecord and returns a list of chunk dictionaries.
        
        Each chunk dictionary conforms to the schema:
        {
            "chunk_id": str,
            "document_id": str,
            "query_id": int,
            "text": str,
            "strategy": str,
            "position": int,
            "metadata": Dict[str, Any]
        }
        """
        pass

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Splits a text string into sentences, respecting both English and
        Indic sentence boundary markers (e.g. purna viram '।', '|', '.', '?', '!').
        """
        if not text:
            return []
            
        # Match Indic purna viram (। / \u0964), double purna viram (॥ / \u0965), standard pipe (|),
        # question mark (?), exclamation (!), and period (.)
        # Lookbehind assertions are used where practical, or simple splitting with filtering.
        sentence_endings = re.compile(r'(?<=[।॥\|!\?\.])\s+')
        sentences = sentence_endings.split(text.strip())
        
        # Filter out empty or whitespace-only elements
        return [s.strip() for s in sentences if s.strip()]

    def create_chunk(
        self,
        chunk_id: str,
        document_id: str,
        query_id: int,
        text: str,
        strategy: str,
        position: int,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Helper to construct a chunk dictionary matching the required internal schema."""
        return {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "query_id": query_id,
            "text": text,
            "strategy": strategy,
            "position": position,
            "metadata": metadata
        }
