import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# Resolve project root as the directory two levels above this file (backend/api/app.py)
# Works regardless of the current working directory (important for Vercel)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from backend.core.config import settings
from backend.retrieval.vector_store import NumpyVectorStore
from backend.retrieval.bm25 import BM25Retriever
from backend.generation.mock import MockGenerator
from backend.generation.sarvam import SarvamGenerator
from backend.generation.guard import validate_generation

# NOTE: EmbeddingGenerator (torch/transformers) is imported lazily inside startup_event
# so the module can load on Vercel even without those heavy packages installed.

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
    """
    Resilient startup: loads indexes and embedding model when available.
    If anything is missing (e.g. on Vercel where indexes/model are not deployed),
    the server still starts so the frontend and /health endpoint work.
    /api/query will return a 503 explaining the situation.
    """
    global embedding_gen, vector_store, bm25_retriever

    logger.info("Initializing RAG resources and loading offline indexes...")
    logger.info(f"Active Chunking Strategy: '{settings.CHUNK_STRATEGY}'")

    dense_dir = str(PROJECT_ROOT / "data" / "indexes" / settings.CHUNK_STRATEGY / "dense")
    bm25_dir  = str(PROJECT_ROOT / "data" / "indexes" / settings.CHUNK_STRATEGY / "bm25")

    # Load dense index (non-fatal if missing)
    index_ok = vector_store.load(dense_dir)
    if not index_ok:
        logger.warning(
            f"Dense index not found at '{dense_dir}'. "
            "RAG retrieval will be unavailable. Run scripts/build_index.py to create it."
        )

    # Load BM25 index (non-fatal if missing)
    bm25_ok = bm25_retriever.load(bm25_dir)
    if not bm25_ok:
        logger.warning(f"BM25 index not found at '{bm25_dir}'.")

    # Load embedding model — only when the dense index was loaded successfully.
    # Importing torch/transformers is lazy to avoid import errors on environments
    # where they are not installed (e.g. Vercel lightweight Python runtime).
    if index_ok:
        try:
            from backend.retrieval.embeddings import get_embedding_generator
            embedding_gen = get_embedding_generator()
            logger.info("RAG resources initialized successfully.")
        except Exception as exc:
            logger.warning(
                f"Embedding model could not be loaded: {exc}. "
                "RAG retrieval will be unavailable on this deployment."
            )
    else:
        logger.warning(
            "Skipping embedding model load — no dense index present. "
            "The application will serve the frontend but RAG queries will return 503."
        )


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

    # Guard: reject queries when the RAG pipeline is not initialized.
    # This happens on Vercel (no indexes/model) or before build_index.py has been run.
    if embedding_gen is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "RAG pipeline not initialized: the dense index or embedding model is not available "
                "on this deployment. Run 'python -m scripts.build_index' locally, or deploy on a "
                "host that supports large model files (Railway / Render / Fly.io)."
            )
        )

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

# ─── Health endpoint ────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
def health_check():
    """Simple liveness probe. Returns 200 OK if the server is running."""
    return JSONResponse({"status": "ok"})


# ─── Frontend SPA (mounted for local dev; served statically on Vercel) ────────
_frontend_dir = PROJECT_ROOT / "frontend"
if _frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="static")
else:
    logger.info(f"Frontend directory '{_frontend_dir}' not mounted locally (handled by static hosting/CDN).")

