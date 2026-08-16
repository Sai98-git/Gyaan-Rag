from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class TranslationMeta(BaseModel):
    model_name: str = Field(default="", description="Name of the translation model used")
    temperature: float = Field(default=0.0, description="Temperature parameter for translation generation")
    max_tokens: int = Field(default=0, description="Max tokens parameter for translation generation")
    top_p: float = Field(default=1.0, description="Top-p parameter for translation generation")
    frequency_penalty: float = Field(default=0.0, description="Frequency penalty parameter")
    presence_penalty: float = Field(default=0.0, description="Presence penalty parameter")

class PassagesGroup(BaseModel):
    is_selected: List[int] = Field(default_factory=list, description="Binary list indicating if a passage contains the answer")
    English_passages: List[str] = Field(default_factory=list, description="List of original English passages")
    Translated_passages: List[str] = Field(default_factory=list, description="List of translated passages")

class DatasetRecord(BaseModel):
    query_id: int = Field(..., description="Unique query identifier")
    query_type: str = Field(default="", description="Category of the query (e.g. description, numeric)")
    query: str = Field(..., description="Query translated into the target language")
    Answer: str = Field(default="", description="Answer translated into the target language")
    Eng_Query: str = Field(default="", description="Original English query")
    Eng_Answer: str = Field(default="", description="Original English answer")
    source_lang: str = Field(default="en", description="Source language code (usually 'en')")
    target_lang: str = Field(..., description="Target language code (e.g. 'hi')")
    meta: TranslationMeta = Field(default_factory=TranslationMeta, description="Translation execution metadata")
    passages: PassagesGroup = Field(default_factory=PassagesGroup, description="Groups of passages in English and target language")
