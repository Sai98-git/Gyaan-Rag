import sys
import logging

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

def test_chunkers():
    print("=== Running Chunker Validation Tests ===")
    
    passage_chunker = PassageChunker()
    sliding_chunker = SlidingWindowChunker()
    semantic_chunker = SemanticChunker()
    
    # Load sample records
    records = []
    try:
        record_gen = iterate_records()
        for _ in range(3):
            records.append(next(record_gen))
    except StopIteration:
        pass
    except Exception as e:
        print(f"Error loading sample records: {e}")
        sys.exit(1)
        
    if not records:
        print("Error: No records loaded.")
        sys.exit(1)
        
    print(f"Loaded {len(records)} test records.")
    
    strategies = {
        "passage": passage_chunker,
        "sliding_window": sliding_chunker,
        "semantic": semantic_chunker
    }
    
    for name, chunker in strategies.items():
        print(f"\n--- Testing Strategy: {name} ---")
        
        all_chunk_ids = set()
        
        for rec_idx, record in enumerate(records):
            chunks = chunker.chunk_record(record)
            print(f"  Record {rec_idx+1} (Query ID: {record.query_id}) generated {len(chunks)} chunks.")
            
            for chunk_idx, chunk in enumerate(chunks):
                # 1. Check schema keys
                required_keys = {"chunk_id", "document_id", "query_id", "text", "strategy", "position", "metadata"}
                missing_keys = required_keys - set(chunk.keys())
                assert not missing_keys, f"Missing required keys: {missing_keys}"
                
                # 2. Check no empty chunks
                text = chunk["text"]
                assert len(text.strip()) > 0, f"Found empty chunk text at chunk_id {chunk['chunk_id']}"
                
                # 3. Check duplicate chunk_id
                cid = chunk["chunk_id"]
                assert cid not in all_chunk_ids, f"Duplicate chunk_id found: {cid}"
                all_chunk_ids.add(cid)
                
                # 4. Check traceability
                assert chunk["query_id"] == record.query_id, f"Traceability fail: query_id mismatch {chunk['query_id']} != {record.query_id}"
                assert chunk["document_id"] == str(record.query_id), "document_id must match str(query_id)"
                assert chunk["strategy"] == name, f"Strategy mismatch: {chunk['strategy']} != {name}"
                
                # 5. Check metadata
                metadata = chunk["metadata"]
                assert isinstance(metadata, dict), "metadata must be a dictionary"
                assert "language" in metadata, "metadata must contain 'language'"
                assert "dataset_split" in metadata, "metadata must contain 'dataset_split'"
                assert "dataset_language" in metadata, "metadata must contain 'dataset_language'"
                assert "parent_query_id" in metadata, "metadata must contain 'parent_query_id'"
                
                # Print a sample chunk from the first record
                if rec_idx == 0 and chunk_idx == 0:
                    print(f"    [Sample Chunk Details]")
                    print(f"      Chunk ID: {chunk['chunk_id']}")
                    print(f"      Text: {chunk['text'][:150]}...")
                    print(f"      Position: {chunk['position']}")
                    print(f"      Metadata keys: {list(metadata.keys())}")
                    
        print(f"  Validation for {name} passed successfully.")
        
    print("\nAll chunker validation checks PASSED successfully.")

if __name__ == "__main__":
    test_chunkers()
