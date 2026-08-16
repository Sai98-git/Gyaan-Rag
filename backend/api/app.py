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
    description="Backend API layer for dense/BM25 retrieval and grounded Indic LLM generation.",
    version="1.0.0"
)

# Global index variables
vector_store = NumpyVectorStore()
bm25_retriever = BM25Retriever()
embedding_gen = None
_initialized = False

def resolve_index_directory(strategy: str, index_type: str) -> Optional[Path]:
    """
    Robustly resolves the index directory across local dev and Vercel serverless.
    index_type: 'bm25' or 'dense'
    """
    target_filename = "bm25_index.json" if index_type == "bm25" else "metadata.json"
    
    # 1. Direct candidate directories
    candidate_dirs = [
        PROJECT_ROOT / "data" / "indexes" / strategy / index_type,
        PROJECT_ROOT / "data" / "indexes" / "semantic" / index_type,
        Path.cwd() / "data" / "indexes" / strategy / index_type,
        Path.cwd() / "data" / "indexes" / "semantic" / index_type,
        Path(__file__).resolve().parent.parent.parent / "data" / "indexes" / strategy / index_type,
        Path(__file__).resolve().parent.parent / "data" / "indexes" / strategy / index_type,
        Path("/var/task") / "data" / "indexes" / strategy / index_type,
        Path("/var/task") / "data" / "indexes" / "semantic" / index_type,
    ]
    
    for candidate in candidate_dirs:
        if (candidate / target_filename).exists():
            logger.info(f"Resolved {index_type} index at: {candidate}")
            return candidate
            
    # 2. Recursive fallback search
    search_roots = [PROJECT_ROOT, Path.cwd(), Path("/var/task"), Path(__file__).resolve().parent.parent.parent]
    for root in search_roots:
        if root.exists():
            try:
                for match in root.rglob(target_filename):
                    found_dir = match.parent
                    logger.info(f"Resolved {index_type} index via recursive search at: {found_dir}")
                    return found_dir
            except Exception as e:
                logger.debug(f"Search in {root} failed: {e}")
                
    logger.error(f"Could not locate {target_filename} in any candidate or search location.")
    return None

def init_rag_resources():
    """Idempotent initialization of RAG indexes and embedding model."""
    global embedding_gen, vector_store, bm25_retriever, _initialized
    if _initialized:
        return

    logger.info("Initializing RAG resources and loading offline indexes...")
    strategy = settings.CHUNK_STRATEGY or "semantic"
    logger.info(f"Active Chunking Strategy: '{strategy}'")

    dense_dir = resolve_index_directory(strategy, "dense")
    bm25_dir  = resolve_index_directory(strategy, "bm25")

    # Load dense index
    index_ok = False
    if dense_dir:
        index_ok = vector_store.load(str(dense_dir))
    if not index_ok:
        logger.warning(f"Dense index could not be loaded from '{dense_dir}'.")

    # Load BM25 index
    bm25_ok = False
    if bm25_dir:
        bm25_ok = bm25_retriever.load(str(bm25_dir))
    if not bm25_ok:
        logger.warning(f"BM25 index could not be loaded from '{bm25_dir}'.")

    # Attempt to load embedding model (local dev / GPU environments only)
    if index_ok:
        try:
            from backend.retrieval.embeddings import get_embedding_generator
            embedding_gen = get_embedding_generator()
            logger.info("Dense embedding model loaded successfully.")
        except Exception as exc:
            logger.info(
                f"Dense embedding model not loaded ({exc}). "
                "Running in serverless mode with local BM25 index."
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
@app.post("/query", response_model=QueryResponse)
def handle_query(payload: QueryRequest):
    """
    RAG endpoint executing the complete pipeline:
    [1] Request received -> [2] Resource init -> [3] Retrieval -> [4] Results ->
    [5] Context assembly -> [6] Provider -> [7] LLM request -> [8] LLM response ->
    [9] Grounding guard -> [10] Response returned
    """
    # [1] Request received
    query = payload.query.strip()
    req_start_time = time.perf_counter()
    logger.info(f"[1] Request received: query='{query}'")

    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty or whitespace."
        )

    # [2] RAG resources initialized
    init_rag_resources()
    logger.info(
        f"[2] RAG resources initialized: dense_chunks={len(vector_store.chunks_metadata)}, "
        f"bm25_chunks={len(bm25_retriever.chunks)}"
    )

    # [3] Retrieval started
    t0 = time.perf_counter()
    retrieved_chunks = []
    retrieval_method = "dense"
    logger.info("[3] Retrieval started...")

    try:
        if embedding_gen is not None:
            query_emb = embedding_gen.embed_query(query)
            retrieved_chunks = vector_store.search(query_emb, top_k=settings.RETRIEVAL_TOP_K)
            retrieval_method = "dense"
        elif len(bm25_retriever.chunks) > 0:
            retrieved_chunks = bm25_retriever.search(query, top_k=settings.RETRIEVAL_TOP_K)
            retrieval_method = "bm25"
        else:
            logger.error("[3.FAIL] Neither dense nor BM25 index files were loaded.")
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": "RETRIEVAL_FAILED",
                    "message": "The retrieval index could not be loaded on this server instance."
                }
            )
    except Exception as e:
        logger.error(f"[3.FAIL] Retrieval failure: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "RETRIEVAL_FAILED",
                "message": "The retrieval index could not be queried."
            }
        )
    retrieval_latency = (time.perf_counter() - t0) * 1000

    # [4] Retrieval completed
    logger.info(
        f"[4] Retrieval completed: method='{retrieval_method}', "
        f"chunks_found={len(retrieved_chunks)}, latency={retrieval_latency:.2f}ms"
    )

    # [5] Context assembly completed
    logger.info(f"[5] Context assembly completed for {len(retrieved_chunks)} chunks.")

    # [6] Generation provider selected
    provider_name = settings.GENERATION_PROVIDER.lower()
    logger.info(f"[6] Generation provider selected: '{provider_name}'")

    # [7] Generation request started
    t0 = time.perf_counter()
    logger.info(f"[7] Generation request started with provider='{provider_name}'...")

    try:
        if provider_name == "sarvam":
            # Check key configuration
            if not settings.SARVAM_API_KEY or settings.SARVAM_API_KEY == "your_sarvam_api_key_here":
                logger.error("[7.FAIL] SARVAM_API_KEY is not configured.")
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={
                        "error": "CONFIGURATION_ERROR",
                        "message": "The production RAG configuration is incomplete (missing SARVAM_API_KEY)."
                    }
                )
            generator = SarvamGenerator()
        else:
            generator = MockGenerator()

        candidate_response = generator.generate(query, retrieved_chunks)
        # [8] LLM response received
        logger.info(f"[8] LLM response received from provider='{provider_name}'.")

    except ValueError as e:
        logger.error(f"[7.FAIL] Configuration error in generator: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "CONFIGURATION_ERROR",
                "message": "The production RAG configuration is incomplete."
            }
        )
    except (TimeoutError, RuntimeError) as e:
        logger.error(f"[7.FAIL] LLM Provider failure: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "GENERATION_FAILED",
                "message": "The language model service could not generate a response."
            }
        )
    except Exception as e:
        logger.error(f"[7.FAIL] Unexpected generation failure: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "GENERATION_FAILED",
                "message": "The language model service could not generate a response."
            }
        )

    # [9] Grounding guard completed
    try:
        final_response = validate_generation(query, retrieved_chunks, candidate_response)
        guard_triggered = final_response.get("guard_triggered", False)
        guard_reason = final_response.get("guard_reason", None)
        logger.info(f"[9] Grounding guard completed: triggered={guard_triggered}, reason='{guard_reason}'")
    except Exception as e:
        logger.error(f"[9.FAIL] Grounding guard error: {e}", exc_info=True)
        final_response = candidate_response
        guard_triggered = False
        guard_reason = None

    generation_latency = (time.perf_counter() - t0) * 1000
    total_latency = (time.perf_counter() - req_start_time) * 1000

    # [10] Response returned
    answer = final_response["answer"]
    sources = final_response.get("sources", [])
    max_ret_score = max((c.get("score", 0.0) for c in retrieved_chunks), default=0.0)

    logger.info(
        f"[10] Response returned: method='{retrieval_method}', "
        f"chunks={len(retrieved_chunks)}, max_score={max_ret_score:.4f}, "
        f"guard={guard_triggered}, ret_lat={retrieval_latency:.2f}ms, "
        f"gen_lat={generation_latency:.2f}ms, tot_lat={total_latency:.2f}ms"
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

# ─── Health endpoints ────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
@app.get("/api/health", tags=["ops"])
def health_check():
    """Safe liveness & readiness probe with subsystem status diagnostics."""
    init_rag_resources()
    strategy = settings.CHUNK_STRATEGY or "semantic"
    dense_path = resolve_index_directory(strategy, "dense")
    bm25_path  = resolve_index_directory(strategy, "bm25")
    has_sarvam_key = bool(
        settings.SARVAM_API_KEY and settings.SARVAM_API_KEY != "your_sarvam_api_key_here"
    )
    is_ready = len(bm25_retriever.chunks) > 0 or len(vector_store.chunks_metadata) > 0

    return JSONResponse({
        "status": "ok",
        "service": "gyaan-rag",
        "retrieval": "ready" if is_ready else "degraded",
        "retrieval_backend": "dense" if embedding_gen is not None else "bm25",
        "bm25_loaded": len(bm25_retriever.chunks) > 0,
        "bm25_chunks": len(bm25_retriever.chunks),
        "dense_chunks": len(vector_store.chunks_metadata),
        "index_path_exists": dense_path is not None or bm25_path is not None,
        "generation_provider": settings.GENERATION_PROVIDER,
        "sarvam_configured": has_sarvam_key
    })


# ─── Frontend SPA (mounted for local dev; served statically on Vercel) ────────
_frontend_dir = PROJECT_ROOT / "public" if (PROJECT_ROOT / "public").is_dir() else (PROJECT_ROOT / "frontend")
if _frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="static")
else:
    logger.info(f"Frontend directory '{_frontend_dir}' not mounted locally (handled by static hosting/CDN).")
