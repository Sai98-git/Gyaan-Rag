from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseGenerator(ABC):
    """
    Abstract base class for all RAG generation providers.
    """
    
    @abstractmethod
    def generate(self, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates a text answer grounded in the supplied context.
        
        Args:
            query: The user query text.
            context: A list of retrieved chunk dictionaries (having 'chunk_id', 'text', 'score', etc.).
            
        Returns:
            A dictionary containing:
                "answer": str (the generated answer text)
                "sources": List[Dict[str, Any]] (the attributed chunk sources)
                "provider": str (the name of the generation provider used)
        """
        pass
