import os
from typing import Optional, Any, Dict
from pydantic import model_validator
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
    GENERATION_PROVIDER: str = "sarvam"  # "mock" or "sarvam"
    SARVAM_API_KEY: Optional[str] = None
    SARVAM_MODEL: str = "sarvam-105b-conversations"
    MIN_RETRIEVAL_SCORE: float = 0.78  # Grounding confidence threshold (below min observed positive 0.7987)

    # Voice / Speech-to-Text Configuration Settings
    STT_PROVIDER: str = "sarvam"  # "sarvam" or "elevenlabs"
    SARVAM_STT_MODEL: str = "saaras:v3"
    ELEVENLABS_API_KEY: Optional[str] = None
    STT_TIMEOUT_SECONDS: float = 15.0

    @model_validator(mode="before")
    @classmethod
    def clean_empty_strings(cls, values: Any) -> Any:
        """
        Vercel and serverless environments frequently inject empty string values (e.g. KEY="")
        for unset optional environment variables. This validator filters out empty/whitespace
        strings so that type-coercion (int, float) doesn't fail and default values are preserved.
        """
        if isinstance(values, dict):
            cleaned: Dict[str, Any] = {}
            for k, v in values.items():
                if isinstance(v, str) and v.strip() == "":
                    continue  # Ignore empty string, use model field default
                cleaned[k] = v
            return cleaned
        return values

settings = Settings()
