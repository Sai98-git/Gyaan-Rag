import time
import logging
import sys

# Configure stdout/stderr to use UTF-8 encoding on Windows to prevent UnicodeEncodeErrors
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass  # Fallback for Python versions that don't support reconfigure

# Configure logging to display to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

from backend.ingestion.dataset_loader import iterate_records

def main():
    print("=== Starting Ingestion Verification Test ===")
    start_time = time.time()
    
    try:
        # Obtain generator
        record_generator = iterate_records()
        
        # Pull first 5 records
        print("\nRetrieving sample records from loader...")
        for i in range(5):
            print(f"\n--- Sample Record {i+1} ---")
            try:
                record = next(record_generator)
                # Print record fields
                print(f"Query ID: {record.query_id}")
                print(f"Query (Target): {record.query}")
                print(f"English Query:  {record.Eng_Query}")
                print(f"Answer (Target): {record.Answer}")
                print(f"English Answer:  {record.Eng_Answer}")
                print(f"Target Language: {record.target_lang}")
                print(f"Selected Passages: {sum(record.passages.is_selected)} of {len(record.passages.Translated_passages)}")
                if record.passages.Translated_passages:
                    print(f"First Translated Passage: {record.passages.Translated_passages[0][:150]}...")
            except StopIteration:
                print("StopIteration: Reached end of dataset shard.")
                break
                
        elapsed = time.time() - start_time
        print(f"\nVerification test completed in {elapsed:.4f} seconds.")
        print("Status: Ingestion module verified successfully.")
        
    except Exception as e:
        print(f"\nVerification FAILED with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
