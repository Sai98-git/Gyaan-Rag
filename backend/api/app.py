import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# Resolve project root and ensure it is in sys.path before any local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from backend.core.config import settings
from backend.retrieval.vector_store import NumpyVectorStore
from backend.retrieval.bm25 import BM25Retriever
from backend.generation.mock import MockGenerator
from backend.generation.sarvam import SarvamGenerator
from backend.generation.guard import validate_generation

# Configure logging
logger = logging.getLogger("rag_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Indic RAG Subsystem API",
    description="Backend API layer for dense retrieval and grounded Indic LLM generation.",
    version="1.0.0"
)

# Global index variables
vector_store = NumpyVectorStore()
bm25_retriever = BM25Retriever()
embedding_gen = None
_initialized = False

def init_rag_resources():
    """Idempotent initialization of RAG indexes and embedding model."""
    global embedding_gen, vector_store, bm25_retriever, _initialized
    if _initialized:
        return

    logger.info("Initializing RAG resources and loading offline indexes...")
    logger.info(f"Active Chunking Strategy: '{settings.CHUNK_STRATEGY}'")

    dense_dir = str(PROJECT_ROOT / "data" / "indexes" / settings.CHUNK_STRATEGY / "dense")
    bm25_dir  = str(PROJECT_ROOT / "data" / "indexes" / settings.CHUNK_STRATEGY / "bm25")

    # Load dense index
    index_ok = vector_store.load(dense_dir)
    if not index_ok:
        logger.warning(f"Dense index not found at '{dense_dir}'.")

    # Load BM25 index
    bm25_ok = bm25_retriever.load(bm25_dir)
    if not bm25_ok:
        logger.warning(f"BM25 index not found at '{bm25_dir}'.")

    # Attempt to load embedding model (local dev / GPU environments)
    if index_ok:
        try:
            from backend.retrieval.embeddings import get_embedding_generator
            embedding_gen = get_embedding_generator()
            logger.info("Embedding model loaded successfully.")
        except Exception as exc:
            logger.warning(
                f"Embedding model could not be loaded ({exc}). "
                "Falling back to BM25 lexical retrieval on the local index."
            )

    _initialized = True
    logger.info(
        f"RAG resources ready (dense_chunks={len(vector_store.chunks_metadata)}, "
        f"bm25_chunks={len(bm25_retriever.chunks)}, dense_model={'loaded' if embedding_gen else 'none'})."
    )

@app.on_event("startup")
def startup_event():
    init_rag_resources()


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
    RAG endpoint that accepts a query, runs retrieval (Dense if model available,
    BM25 fallback on serverless), constructs context, generates a grounded answer,
    applies safety guards, and returns sources.
    """
    init_rag_resources()

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
    retrieved_chunks = []
    retrieval_method = "dense"

    try:
        if embedding_gen is not None:
            query_emb = embedding_gen.embed_query(query)
            retrieved_chunks = vector_store.search(query_emb, top_k=settings.RETRIEVAL_TOP_K)
            retrieval_method = "dense"
        elif len(bm25_retriever.chunks) > 0:
            retrieved_chunks = bm25_retriever.search(query, top_k=settings.RETRIEVAL_TOP_K)
            retrieval_method = "bm25"
        else:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": "INDEX_NOT_AVAILABLE",
                    "message": "Retrieval index is not loaded on this server instance.",
                    "detail": "Neither dense nor BM25 index files were loaded."
                }
            )
    except Exception as e:
        logger.error(f"Retrieval failure: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "RETRIEVAL_FAILED",
                "message": "Error occurred during the retrieval phase.",
                "detail": str(e)
            }
        )
    retrieval_latency = (time.perf_counter() - t0) * 1000
    
    # 2. Generation Phase
    t0 = time.perf_counter()
    provider_name = settings.GENERATION_PROVIDER.lower()
    
    try:
        if provider_name == "sarvam":
            generator = SarvamGenerator()
        else:
            generator = MockGenerator()
            
        candidate_response = generator.generate(query, retrieved_chunks)
        final_response = validate_generation(query, retrieved_chunks, candidate_response)
        
    except ValueError as e:
        logger.error(f"Bad Request Parameter: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "INVALID_REQUEST",
                "message": str(e),
                "detail": "Invalid parameter provided to generation subsystem."
            }
        )
    except (TimeoutError, RuntimeError) as e:
        logger.error(f"LLM Provider error: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "LLM_PROVIDER_ERROR",
                "message": "LLM Provider is currently unavailable.",
                "detail": str(e)
            }
        )
    except Exception as e:
        logger.error(f"Unexpected generation failure: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "GENERATION_FAILED",
                "message": "Internal error during answer generation.",
                "detail": str(e)
            }
        )

    generation_latency = (time.perf_counter() - t0) * 1000
    total_latency = (time.perf_counter() - req_start_time) * 1000
    
    # Extract details for response
    answer = final_response["answer"]
    sources = final_response.get("sources", [])
    guard_triggered = final_response.get("guard_triggered", False)
    guard_reason = final_response.get("guard_reason", None)
    
    # 3. Structured Logging
    max_ret_score = max((c.get("score", 0.0) for c in retrieved_chunks), default=0.0)
    logger.info(
        f"RAG Request Completed: "
        f"method='{retrieval_method}', "
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
    """Simple liveness & readiness probe with subsystem status."""
    init_rag_resources()
    return JSONResponse({
        "status": "ok",
        "dense_chunks": len(vector_store.chunks_metadata),
        "bm25_chunks": len(bm25_retriever.chunks),
        "dense_model_loaded": embedding_gen is not None,
        "generation_provider": settings.GENERATION_PROVIDER
    })


# ─── Frontend SPA (mounted for local dev; served statically on Vercel) ────────
_frontend_dir = PROJECT_ROOT / "public" if (PROJECT_ROOT / "public").is_dir() else (PROJECT_ROOT / "frontend")
if _frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="static")
else:
    logger.info(f"Frontend directory '{_frontend_dir}' not mounted locally (handled by static hosting/CDN).")
