import sys
from datasets import load_dataset_builder, load_dataset

def main():
    print("=== Step 1: Loading Dataset Builder ===")
    try:
        builder = load_dataset_builder("ai4bharat/MSMARCO-XI")
        print(f"Description:\n{builder.info.description}\n")
        print(f"Features/Schema:\n{builder.info.features}\n")
        print(f"Available Splits:\n{builder.info.splits}\n")
        print(f"Hugging Face Metadata:\n{builder.info}\n")
    except Exception as e:
        print(f"Error loading builder: {e}")

    print("\n=== Step 2-5: Streaming First 5 Records ===")
    try:
        ds = load_dataset(
            "ai4bharat/MSMARCO-XI",
            split="train",
            streaming=True
        )
        iterator = iter(ds)
        
        for i in range(5):
            print(f"\n--- Sample Record {i+1} ---")
            try:
                sample = next(iterator)
                for key, val in sample.items():
                    val_str = str(val)
                    if len(val_str) > 500:
                        val_str = val_str[:500] + "... [TRUNCATED]"
                    print(f"{key}: {val_str}")
            except StopIteration:
                print("StopIteration: Fewer than 5 records available in dataset.")
                break
    except Exception as e:
        print(f"Error during dataset streaming: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
