import logging
from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from backend.core.config import settings

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Handles local loading of Hugging Face embedding models and computes 
    L2-normalized sentence embeddings using PyTorch.
    
    Defaults to 'intfloat/multilingual-e5-small' which requires 
    'query: ' or 'passage: ' prefixes for optimal performance.
    """
    
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading embedding model '{self.model_name}' on device '{self.device}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()  # Set model to evaluation mode
        logger.info("Embedding model loaded successfully.")

    def _mean_pooling(self, model_output, attention_mask) -> torch.Tensor:
        """Performs mean pooling taking the attention mask into account."""
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed_texts(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """
        Embeds a list of texts. Adds the required prefix for E5 models 
        and returns L2-normalized embeddings.
        """
        if not texts:
            return []
            
        # Add E5 prefixes: 'query: ' for queries, 'passage: ' for index passages
        prefix = "query: " if is_query else "passage: "
        prefixed_texts = [prefix + text for text in texts]
        
        # Tokenize inputs
        encoded_input = self.tokenizer(
            prefixed_texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors='pt'
        ).to(self.device)
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            
        # Perform mean pooling
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings to L2 norm
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings.cpu().numpy().tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embeds a single query string."""
        return self.embed_texts([query], is_query=True)[0]

    def embed_passages(self, passages: List[str]) -> List[List[float]]:
        """Embeds a list of passage strings."""
        return self.embed_texts(passages, is_query=False)

# Singleton instance
_generator_instance = None

def get_embedding_generator() -> EmbeddingGenerator:
    """Helper to retrieve/instantiate the global embedding generator singleton."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = EmbeddingGenerator()
    return _generator_instance
