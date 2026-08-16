import sys
import time
import logging
import statistics

# Ensure UTF-8 output on Windows terminal
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Setup logging
logging.basicConfig(level=logging.WARNING)

from backend.ingestion.dataset_loader import iterate_records
from backend.ingestion.chunkers import PassageChunker, SlidingWindowChunker, SemanticChunker

def compare_chunking():
    print("=== Consuming Sample Records for Chunking Comparison ===")
    
    # Load 100 records
    records = []
    try:
        record_gen = iterate_records()
        for i in range(100):
            records.append(next(record_gen))
    except StopIteration:
        pass
    except Exception as e:
        print(f"Error loading records: {e}")
        sys.exit(1)
        
    print(f"Loaded {len(records)} records for benchmarking.")
    
    passage_chunker = PassageChunker()
    sliding_chunker = SlidingWindowChunker()
    semantic_chunker = SemanticChunker()
    
    chunkers = {
        "Passage-Aware": passage_chunker,
        "Sliding-Window": sliding_chunker,
        "Semantic/Structure-Aware": semantic_chunker
    }
    
    results = {}
    
    for name, chunker in chunkers.items():
        print(f"Benchmarking {name}...")
        
        start_time = time.perf_counter()
        
        all_chunks = []
        for record in records:
            chunks = chunker.chunk_record(record)
            all_chunks.extend(chunks)
            
        elapsed = time.perf_counter() - start_time
        
        lengths = [len(c["text"]) for c in all_chunks]
        
        # Calculate statistics
        num_chunks = len(all_chunks)
        if num_chunks > 0:
            avg_len = sum(lengths) / num_chunks
            med_len = statistics.median(lengths)
            min_len = min(lengths)
            max_len = max(lengths)
        else:
            avg_len, med_len, min_len, max_len = 0, 0, 0, 0
            
        # Extract metadata keys from the first chunk if available
        sample_meta_keys = list(all_chunks[0]["metadata"].keys()) if all_chunks else []
        
        results[name] = {
            "num_chunks": num_chunks,
            "avg_length": avg_len,
            "median_length": med_len,
            "min_length": min_len,
            "max_length": max_len,
            "processing_time": elapsed,
            "metadata_keys": ", ".join(sample_meta_keys)
        }
        
    # Print results table
    print("\n" + "="*80)
    print(f"{'Strategy':<26} | {'Chunks':<6} | {'Avg Len':<7} | {'Med Len':<7} | {'Min':<4} | {'Max':<5} | {'Time (s)':<8}")
    print("-"*80)
    for name, stats in results.items():
        print(f"{name:<26} | {stats['num_chunks']:<6} | {stats['avg_length']:<7.1f} | {stats['median_length']:<7.1f} | {stats['min_length']:<4} | {stats['max_length']:<5} | {stats['processing_time']:<8.4f}")
    print("="*80)
    
    print("\nMetadata Schema Preservation:")
    for name, stats in results.items():
        print(f"  {name}: {stats['metadata_keys']}")

if __name__ == "__main__":
    compare_chunking()
