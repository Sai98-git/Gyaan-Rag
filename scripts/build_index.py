import sys
import os
import time
import logging

# Ensure UTF-8 output on Windows terminal
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("build_index")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.core.config import settings
# Force CORPUS_MODE to sample for controlled benchmarking
settings.CORPUS_MODE = "sample"
settings.MAX_RECORDS = 200  # We index 200 records (2,000 passages)

from backend.ingestion.dataset_loader import iterate_records
from backend.ingestion.chunkers import PassageChunker, SlidingWindowChunker, SemanticChunker
from backend.retrieval.embeddings import get_embedding_generator
from backend.retrieval.vector_store import NumpyVectorStore
from backend.retrieval.bm25 import BM25Retriever

def build_index_for_strategy(strategy_name: str, chunker, records, embedding_gen):
    logger.info(f"=== Building index for strategy: {strategy_name} ===")
    
    # 1. Generate Chunks
    start_time = time.perf_counter()
    chunks = []
    for r in records:
        record_chunks = chunker.chunk_record(r)
        for c in record_chunks:
            c["searchable_text"] = f"{c['text']} {r.Eng_Query} {r.query}"
            if "metadata" not in c:
                c["metadata"] = {}
            c["metadata"]["eng_query"] = r.Eng_Query
            c["metadata"]["hin_query"] = r.query
        chunks.extend(record_chunks)
    logger.info(f"Generated {len(chunks)} chunks from {len(records)} records in {time.perf_counter() - start_time:.4f}s.")
    
    if not chunks:
        logger.warning(f"No chunks generated for {strategy_name}. Skipping index creation.")
        return
        
    # 2. Compute Dense Embeddings (Batching to optimize CPU/GPU)
    logger.info("Computing dense embeddings for all chunks...")
    emb_start = time.perf_counter()
    chunk_texts = [c["text"] for c in chunks]
    
    batch_size = 64
    embeddings = []
    for i in range(0, len(chunk_texts), batch_size):
        batch_texts = chunk_texts[i:i+batch_size]
        batch_embeddings = embedding_gen.embed_passages(batch_texts)
        embeddings.extend(batch_embeddings)
        if (i // batch_size) % 5 == 0 or i + batch_size >= len(chunk_texts):
            logger.info(f"  Embedded {min(i + batch_size, len(chunk_texts))}/{len(chunk_texts)} chunks...")
            
    logger.info(f"Computed embeddings in {time.perf_counter() - emb_start:.4f}s.")
    
    # 3. Create and Save Dense Index
    dense_dir = f"data/indexes/{strategy_name}/dense"
    vector_store = NumpyVectorStore()
    vector_store.add_chunks(chunks, embeddings)
    vector_store.save(dense_dir)
    
    # 4. Create and Save BM25 Index
    bm25_dir = f"data/indexes/{strategy_name}/bm25"
    bm25_retriever = BM25Retriever()
    bm25_retriever.add_chunks(chunks)
    bm25_retriever.save(bm25_dir)
    
    logger.info(f"Completed index building for {strategy_name}.\n")

def main():
    logger.info("Starting Offline Index Construction...")
    
    # 1. Load the corpus records once to share across all strategies
    logger.info("Loading corpus records...")
    load_start = time.perf_counter()
    try:
        record_gen = iterate_records()
        records = list(record_gen)
    except Exception as e:
        logger.error(f"Error loading records: {e}")
        sys.exit(1)
        
    logger.info(f"Loaded {len(records)} records in {time.perf_counter() - load_start:.4f}s.")
    
    if not records:
        logger.error("No records found to index.")
        sys.exit(1)
        
    # Initialize embedding generator singleton once
    embedding_gen = get_embedding_generator()
    
    # 2. Run index creation for each chunker strategy
    strategies = {
        "passage": PassageChunker(),
        "sliding_window": SlidingWindowChunker(),
        "semantic": SemanticChunker()
    }
    
    overall_start = time.perf_counter()
    for name, chunker in strategies.items():
        build_index_for_strategy(name, chunker, records, embedding_gen)
        
    logger.info(f"All index construction completed in {time.perf_counter() - overall_start:.4f}s.")

if __name__ == "__main__":
    main()
