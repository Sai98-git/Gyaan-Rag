import os
import json
import logging
from typing import List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

class NumpyVectorStore:
    """
    A lightweight, pure-numpy vector store for dense retrieval.
    
    Persists embeddings as a `.npy` binary file and chunk metadata as a `.json` file.
    Performs cosine similarity search using normalized matrix dot product.
    """
    
    def __init__(self):
        self.embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.chunks_metadata: List[Dict[str, Any]] = []

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings_list: List[List[float]]):
        """Appends new chunks and their corresponding embeddings to the vector store."""
        if not chunks or not embeddings_list:
            return
            
        assert len(chunks) == len(embeddings_list), "Mismatch between chunks and embeddings list sizes"
        
        new_embeddings = np.array(embeddings_list, dtype=np.float32)
        
        if self.embeddings.size == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
        self.chunks_metadata.extend(chunks)
        logger.info(f"Added {len(chunks)} chunks to NumpyVectorStore. Total size: {len(self.chunks_metadata)}")

    def save(self, directory: str):
        """Persists the index (embeddings and metadata) to the given directory."""
        os.makedirs(directory, exist_ok=True)
        
        embeddings_path = os.path.join(directory, "embeddings.npy")
        metadata_path = os.path.join(directory, "metadata.json")
        
        # Save embeddings
        np.save(embeddings_path, self.embeddings)
        
        # Save metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks_metadata, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Vector index saved successfully to: {directory}")

    def load(self, directory: str) -> bool:
        """Loads the index from the given directory. Returns True if successful, False otherwise."""
        embeddings_path = os.path.join(directory, "embeddings.npy")
        metadata_path = os.path.join(directory, "metadata.json")
        
        if not os.path.exists(embeddings_path) or not os.path.exists(metadata_path):
            logger.warning(f"Vector index files not found in: {directory}")
            return False
            
        try:
            self.embeddings = np.load(embeddings_path)
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.chunks_metadata = json.load(f)
            logger.info(f"Loaded {len(self.chunks_metadata)} chunks from vector index: {directory}")
            return True
        except Exception as e:
            logger.error(f"Error loading vector index from {directory}: {e}")
            return False

    def search(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        """
        Queries the vector store and returns the top_k most similar chunks.
        
        Since both query and passage embeddings are L2-normalized during creation,
        cosine similarity is simply the dot product.
        """
        if self.embeddings.size == 0 or not self.chunks_metadata:
            return []
            
        q_vec = np.array(query_vector, dtype=np.float32)
        
        # Compute dot products (cosine similarities)
        similarities = np.dot(self.embeddings, q_vec)
        
        # Get top-k indices in descending order
        top_k = min(top_k, len(self.chunks_metadata))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks_metadata[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": float(similarities[idx]),
                "retrieval_method": "dense",
                "metadata": chunk["metadata"]
            })
            
        return results
