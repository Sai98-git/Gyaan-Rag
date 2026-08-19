import os
import logging
from typing import Generator, Optional
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from backend.core.config import settings
from backend.ingestion.metadata import DatasetRecord
from backend.ingestion.cleaner import normalize_row

logger = logging.getLogger(__name__)

# Mapping 2-letter language codes to the 3-letter codes used in repository filenames
LANG_MAP_2_TO_3 = {
    "as": "asm",
    "bn": "ben",
    "gu": "guj",
    "hi": "hin",
    "kn": "kan",
    "ml": "mal",
    "mr": "mar",
    "ne": "nep",
    "or": "ori",
    "pa": "pan",
    "sa": "san",
    "ta": "tam",
    "te": "tel",
    "ur": "urd"
}

def resolve_language_code(lang: str) -> str:
    """Resolves language code to the 3-letter format used in filenames."""
    lang = lang.strip().lower()
    if lang in LANG_MAP_2_TO_3:
        return LANG_MAP_2_TO_3[lang]
    # Check if already a valid 3-letter code
    if len(lang) == 3 and lang in LANG_MAP_2_TO_3.values():
        return lang
    
    # Fallback to default if unrecognized
    logger.warning(f"Unrecognized language code '{lang}', defaulting to 'hin' (Hindi)")
    return "hin"

def get_target_filename(corpus_mode: str, language_3: str, data_split: str) -> str:
    """
    Returns the repository filename corresponding to the config settings.
    
    In CORPUS_MODE="sample", we use validation splits to keep downloads small (~430MB).
    """
    if corpus_mode == "sample":
        # Force validation split for sampling to prevent multi-GB train split downloads
        logger.info("CORPUS_MODE='sample': using validation split for development speed.")
        return f"validation/{language_3}val.parquet"
    
    # CORPUS_MODE="language" or "full" (which maps per split/language)
    split_dir = "train" if data_split == "train" else "validation"
    suffix = "train" if data_split == "train" else "val"
    
    # Critical Check: Telugu ('tel') is missing from the train split in the repository
    if split_dir == "train" and language_3 == "tel":
        raise ValueError(
            "Telugu ('tel'/'te') train split is not available in the ai4bharat/MSMARCO-XI repository. "
            "Please use DATA_SPLIT=validation or select a different language."
        )
        
    return f"{split_dir}/{language_3}{suffix}.parquet"

def download_dataset_shard() -> str:
    """
    Downloads the target Parquet shard from Hugging Face LFS to the local cache directory.
    
    Returns:
        str: Absolute local path to the downloaded file.
    """
    lang_3 = resolve_language_code(settings.DATA_LANGUAGE)
    filename = get_target_filename(settings.CORPUS_MODE, lang_3, settings.DATA_SPLIT)
    
    logger.info(f"Downloading dataset shard '{filename}' from repo '{settings.DATASET_NAME}'...")
    
    # Download using HF Hub downloader (handles caching automatically)
    local_path = hf_hub_download(
        repo_id=settings.DATASET_NAME,
        filename=filename,
        repo_type="dataset",
        cache_dir=settings.CACHE_DIR,
        token=settings.HF_TOKEN
    )
    
    logger.info(f"Dataset shard successfully downloaded/located at: {local_path}")
    return local_path

def iterate_records(max_records: Optional[int] = None) -> Generator[DatasetRecord, None, None]:
    """
    Downloads the required Parquet file, streams it in record batches,
    normalizes, and yields validated DatasetRecord objects.
    
    Limits memory usage by iterating in pyarrow RecordBatches.
    """
    try:
        local_path = download_dataset_shard()
    except Exception as e:
        logger.error(f"Failed to retrieve dataset shard: {e}")
        raise e
        
    logger.info("Initializing Parquet batch reader...")
    pf = pq.ParquetFile(local_path)
    
    limit = max_records if max_records is not None else (settings.MAX_RECORDS if settings.CORPUS_MODE == "sample" else None)
    count = 0
    
    # Read row-group in batches of 2000 rows to optimize memory and CPU
    for batch in pf.iter_batches(batch_size=2000):
        # Convert pyarrow RecordBatch directly to list of dictionaries
        rows = batch.to_pylist()
        for row in rows:
            record = normalize_row(row)
            if record:
                yield record
                count += 1
                if limit and count >= limit:
                    logger.info(f"Reached limit of {limit}. Stopping iteration.")
                    return
                    
    logger.info(f"Finished loading dataset shard. Total records yielded: {count}")
