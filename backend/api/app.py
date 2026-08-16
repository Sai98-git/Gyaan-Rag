import time
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, status
from fastapi.staticfiles import StaticFiles

from backend.core.config import settings
from backend.retrieval.embeddings import get_embedding_generator
from backend.retrieval.vector_store import NumpyVectorStore
from backend.retrieval.bm25 import BM25Retriever
from backend.generation.mock import MockGenerator
from backend.generation.sarvam import SarvamGenerator
from backend.generation.guard import validate_generation

# Configure logging
logger = logging.getLogger("rag_api")

app = FastAPI(
    title="Indic RAG Subsystem API",
    description="Backend API layer for dense retrieval and grounded Indic LLM generation.",
    version="1.0.0"
)

# Global index variables
vector_store = NumpyVectorStore()
bm25_retriever = BM25Retriever()
embedding_gen = None

@app.on_event("startup")
def startup_event():
    global embedding_gen, vector_store, bm25_retriever
    
    logger.info("Initializing RAG resources and loading offline indexes...")
    logger.info(f"Active Chunking Strategy: '{settings.CHUNK_STRATEGY}'")
    
    dense_dir = f"data/indexes/{settings.CHUNK_STRATEGY}/dense"
    bm25_dir = f"data/indexes/{settings.CHUNK_STRATEGY}/bm25"
    
    # Load dense index
    if not vector_store.load(dense_dir):
        logger.error(f"Failed to load dense index from {dense_dir}. Ensure build_index script is run first.")
        
    # Load BM25 index
    if not bm25_retriever.load(bm25_dir):
        logger.error(f"Failed to load BM25 index from {bm25_dir}. Ensure build_index script is run first.")
        
    # Initialize embedding generator (triggers weights loading)
    embedding_gen = get_embedding_generator()
    logger.info("RAG resources initialized successfully.")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The query string to search and answer.")


class SourceItem(BaseModel):
    chunk_id: str
    score: float
    preview: Optional[str] = None
    metadata: Dict[str, Any]


class LatencyDetails(BaseModel):
    latency_ms: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    retrieval: LatencyDetails
    generation: LatencyDetails
    provider: str
    guard_triggered: Optional[bool] = False
    guard_reason: Optional[str] = None


@app.post("/api/query", response_model=QueryResponse)
def handle_query(payload: QueryRequest):
    """
    RAG endpoint that accepts a query, runs dense retrieval, constructs context,
    generates a grounded answer, applies safety guards, and returns sources.
    """
    query = payload.query.strip()
    req_start_time = time.perf_counter()
    
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty or whitespace."
        )
        
    logger.info(f"Received query request: '{query}'")
    
    # 1. Retrieval Phase
    t0 = time.perf_counter()
    try:
        query_emb = embedding_gen.embed_query(query)
        retrieved_chunks = vector_store.search(query_emb, top_k=settings.RETRIEVAL_TOP_K)
    except Exception as e:
        logger.error(f"Retrieval failure: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Subsystem retrieval failure: {e}"
        )
    retrieval_latency = (time.perf_counter() - t0) * 1000
    
    # 2. Generation Phase
    t0 = time.perf_counter()
    provider_name = settings.GENERATION_PROVIDER.lower()
    
    try:
        # Select Generator
        if provider_name == "sarvam":
            generator = SarvamGenerator()
        else:
            generator = MockGenerator()
            
        candidate_response = generator.generate(query, retrieved_chunks)
        
        # Apply Hallucination/Grounding Guards
        final_response = validate_generation(query, retrieved_chunks, candidate_response)
        
    except ValueError as e:
        logger.error(f"Bad Request Parameter: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except (TimeoutError, RuntimeError) as e:
        logger.error(f"LLM Provider unavailable: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM Provider failure: {e}"
        )
    except Exception as e:
        logger.error(f"Unexpected generation failure: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM Subsystem generation failure: {e}"
        )
    generation_latency = (time.perf_counter() - t0) * 1000
    total_latency = (time.perf_counter() - req_start_time) * 1000
    
    # Extract details for response
    answer = final_response["answer"]
    sources = final_response.get("sources", [])
    guard_triggered = final_response.get("guard_triggered", False)
    guard_reason = final_response.get("guard_reason", None)
    
    # 3. Structured Logging
    max_ret_score = max((c["score"] for c in retrieved_chunks), default=0.0)
    logger.info(
        f"RAG Request Completed: "
        f"query_id=None, "
        f"strategy='{settings.CHUNK_STRATEGY}', "
        f"chunks_retrieved={len(retrieved_chunks)}, "
        f"max_score={max_ret_score:.4f}, "
        f"provider='{provider_name}', "
        f"guard_triggered={guard_triggered}, "
        f"ret_lat={retrieval_latency:.2f}ms, "
        f"gen_lat={generation_latency:.2f}ms, "
        f"tot_lat={total_latency:.2f}ms"
    )
    
    return QueryResponse(
        answer=answer,
        sources=sources,
        retrieval=LatencyDetails(latency_ms=retrieval_latency),
        generation=LatencyDetails(latency_ms=generation_latency),
        provider=provider_name,
        guard_triggered=guard_triggered,
        guard_reason=guard_reason
    )

# Serve Frontend SPA Static files
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
