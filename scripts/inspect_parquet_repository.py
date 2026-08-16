import os
import collections
import pyarrow.parquet as pq
import fsspec
from huggingface_hub import HfApi

def inspect_repository():
    dataset_id = "ai4bharat/MSMARCO-XI"
    print(f"=== Inspecting Hugging Face Repository: {dataset_id} ===")
    
    api = HfApi()
    try:
        info = api.dataset_info(dataset_id, files_metadata=True)
    except Exception as e:
        print(f"Error fetching dataset info: {e}")
        return

    siblings = info.siblings
    all_files = [s.rfilename for s in siblings]
    parquet_siblings = [s for s in siblings if s.rfilename.endswith('.parquet')]
    
    print(f"Total number of files in repository: {len(all_files)}")
    print(f"Total number of Parquet files: {len(parquet_siblings)}\n")

    # Group files by split
    by_split = collections.defaultdict(list)
    for s in parquet_siblings:
        split = "train" if "train/" in s.rfilename else "validation" if "validation/" in s.rfilename else "other"
        by_split[split].append(s)

    # Print files for each split
    for split, items in sorted(by_split.items()):
        print(f"--- Split: {split} ({len(items)} files) ---")
        sorted_items = sorted(items, key=lambda x: x.size)
        for s in sorted(items, key=lambda x: x.rfilename):
            size_gb = s.size / (1024 * 1024 * 1024)
            print(f"  {s.rfilename} | Size: {size_gb:.4f} GB ({s.size} bytes)")
        
        # Display smallest/largest
        if sorted_items:
            print(f"  Smallest: {sorted_items[0].rfilename} ({sorted_items[0].size / (1024 * 1024 * 1024):.4f} GB)")
            print(f"  Largest:  {sorted_items[-1].rfilename} ({sorted_items[-1].size / (1024 * 1024 * 1024):.4f} GB)")
        print()

    # Determine schema for a representative file
    representative_file = "train/hintrain.parquet"
    url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{representative_file}"
    print(f"=== Reading Schema of Representative File: {representative_file} ===")
    try:
        f = fsspec.open(url).open()
        pf = pq.ParquetFile(f)
        meta = pf.metadata
        print("Schema:")
        print(pf.schema)
        print("\nMetadata Summary:")
        print(f"  Columns: {meta.num_columns}")
        print(f"  Rows: {meta.num_rows}")
        print(f"  Row Groups: {meta.num_row_groups}")
        print(f"  Serialized Metadata Size: {meta.serialized_size} bytes")
    except Exception as e:
        print(f"Failed to read parquet metadata: {e}")

if __name__ == "__main__":
    inspect_repository()
