import os
import json
import re
import math
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class BM25Retriever:
    """
    A pure-python word-level BM25 lexical retriever.
    
    Includes tokenization for multilingual/Indic texts, corpus IDF calculation, 
    local serialization, and score-based retrieval.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lengths: List[int] = []
        self.doc_term_frequencies: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.chunks: List[Dict[str, Any]] = []

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes text by lowercasing and splitting on whitespace and punctuation.
        Works across both English and Indic texts.
        """
        if not text:
            return []
        # Replace punctuation marks with spaces and lowercase
        clean_text = re.sub(r'[।॥\|!\?\.,;:\(\)"\'\-\n\r\t]', ' ', text.lower())
        return [w.strip() for w in clean_text.split() if w.strip()]

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Builds the BM25 index parameters from the chunk corpus."""
        if not chunks:
            return
            
        self.chunks.extend(chunks)
        self.corpus_size = len(self.chunks)
        
        doc_terms = []
        total_len = 0
        term_doc_counts: Dict[str, int] = {}
        
        for chunk in chunks:
            tokens = self.tokenize(chunk["text"])
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_len += doc_len
            
            # Compute term frequencies for this chunk
            tf: Dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.doc_term_frequencies.append(tf)
            
            # Count document occurrences of each term
            for term in tf.keys():
                term_doc_counts[term] = term_doc_counts.get(term, 0) + 1
                
        self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 else 0.0
        
        # Compute IDF values
        # Formula: log((N + 1) / (df + 0.5)) - guarantees positive values
        for term, df in term_doc_counts.items():
            self.idf[term] = math.log((self.corpus_size + 1.0) / (df + 0.5))
            
        logger.info(f"Built BM25 index with {self.corpus_size} documents. Avg doc length: {self.avg_doc_len:.2f}")

    def save(self, directory: str):
        """Persists the BM25 index to the given directory."""
        os.makedirs(directory, exist_ok=True)
        index_path = os.path.join(directory, "bm25_index.json")
        
        data = {
            "corpus_size": self.corpus_size,
            "avg_doc_len": self.avg_doc_len,
            "doc_lengths": self.doc_lengths,
            "doc_term_frequencies": self.doc_term_frequencies,
            "idf": self.idf,
            "chunks": self.chunks
        }
        
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"BM25 index saved successfully to: {directory}")

    def load(self, directory: str) -> bool:
        """Loads the BM25 index from the given directory. Returns True if successful."""
        index_path = os.path.join(directory, "bm25_index.json")
        if not os.path.exists(index_path):
            logger.warning(f"BM25 index file not found in: {directory}")
            return False
            
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            self.corpus_size = data["corpus_size"]
            self.avg_doc_len = data["avg_doc_len"]
            self.doc_lengths = data["doc_lengths"]
            self.doc_term_frequencies = data["doc_term_frequencies"]
            self.idf = data["idf"]
            self.chunks = data["chunks"]
            
            logger.info(f"Loaded BM25 index with {self.corpus_size} chunks from: {directory}")
            return True
        except Exception as e:
            logger.error(f"Error loading BM25 index from {directory}: {e}")
            return False

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Queries the BM25 index and returns sorted chunks with lexical relevance scores."""
        if self.corpus_size == 0 or not self.chunks:
            return []
            
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []
            
        scores = []
        for idx in range(self.corpus_size):
            score = 0.0
            doc_len = self.doc_lengths[idx]
            tf = self.doc_term_frequencies[idx]
            
            for token in query_tokens:
                if token in tf:
                    # BM25 tf adjustment formula
                    tf_val = tf[token]
                    idf_val = self.idf.get(token, 0.0)
                    
                    numerator = tf_val * (self.k1 + 1.0)
                    denominator = tf_val + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                    
                    score += idf_val * (numerator / denominator)
            
            scores.append((idx, score))
            
        # Filter results with score > 0 and sort
        filtered_scores = [(idx, score) for idx, score in scores if score > 0]
        sorted_scores = sorted(filtered_scores, key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for idx, score in sorted_scores:
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": score,
                "retrieval_method": "sparse",
                "metadata": chunk["metadata"]
            })
            
        return results
