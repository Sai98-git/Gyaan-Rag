import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    DATASET_NAME: str = "ai4bharat/MSMARCO-XI"
    CORPUS_MODE: str = "sample"  # "sample", "language", "full"
    MAX_RECORDS: int = 25000
    DATA_LANGUAGE: str = "hi"    # 2-letter (e.g., 'hi') or 3-letter (e.g., 'hin') language code
    DATA_SPLIT: str = "train"    # "train" or "validation"
    HF_TOKEN: Optional[str] = None
    CACHE_DIR: str = os.path.join(os.path.expanduser("~"), ".cache", "rag_in_goa")

    # Chunking Configuration Settings
    CHUNK_STRATEGY: str = "semantic"  # "passage", "sliding_window", "semantic"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    MIN_CHUNK_SIZE: int = 50
    MAX_CHUNK_SIZE: int = 1000

    # Retrieval Configuration Settings
    EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-small"
    RETRIEVAL_TOP_K: int = 10
    RRF_K: int = 60
    DENSE_WEIGHT: float = 0.5
    SPARSE_WEIGHT: float = 0.5

    # Generation Configuration Settings
    GENERATION_PROVIDER: str = "mock"  # "mock" or "sarvam"
    SARVAM_API_KEY: Optional[str] = None
    SARVAM_MODEL: str = "sarvam-105b"
    MIN_RETRIEVAL_SCORE: float = 0.78  # Grounding confidence threshold (below min observed positive 0.7987)

settings = Settings()
