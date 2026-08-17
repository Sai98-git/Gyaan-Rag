import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form, Request
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
from backend.voice import get_stt_provider, normalize_voice_query

# Configure logging
logger = logging.getLogger("rag_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="ज्ञान Gyaan RAG — Voice-Enabled Indic RAG Subsystem API",
    description="End-to-end Voice-to-Text, Dense/BM25 retrieval, and Grounded Indic Generation API.",
    version="2.0.0"
)

@app.middleware("http")
async def vercel_request_path_normalizer(request: Request, call_next):
    """
    Normalizes Vercel rewritten paths (x-matched-path / x-forwarded-uri)
    so that FastAPI router matches /api/query, /api/voice, /health, /api/health directly.
    """
    matched_path = request.headers.get("x-matched-path") or request.headers.get("x-forwarded-uri")
    if matched_path:
        clean_path = matched_path.split("?")[0]
        if clean_path and not clean_path.startswith("/api/index"):
            request.scope["path"] = clean_path
    elif request.scope.get("path", "").startswith("/api/index.py/"):
        request.scope["path"] = request.scope["path"][len("/api/index.py"):]
    elif request.scope.get("path", "").startswith("/api/index/"):
        request.scope["path"] = request.scope["path"][len("/api/index"):]
        
    return await call_next(request)

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


# ─── Pydantic Schemas ────────────────────────────────────────────────────────
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


class VoiceLatencyBreakdown(BaseModel):
    stt_ms: float
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class VoiceResponse(BaseModel):
    success: bool = True
    transcript: str
    normalized_query: str
    language: str
    answer: str
    sources: List[SourceItem]
    provider: str
    stt_provider: str
    guard_triggered: bool = False
    guard_reason: Optional[str] = None
    latency: VoiceLatencyBreakdown


# ─── Core RAG Pipeline Helper ────────────────────────────────────────────────
def _execute_rag_pipeline(query: str) -> Dict[str, Any]:
    """
    Core retrieval -> generation -> guard pipeline shared by text and voice.
    
    Returns:
        Dict containing answer, sources, retrieval_latency_ms, generation_latency_ms,
        provider, guard_triggered, guard_reason.
    """
    init_rag_resources()

    # 1. Retrieval
    t0 = time.perf_counter()
    retrieved_chunks = []
    retrieval_method = "dense"

    if embedding_gen is not None:
        query_emb = embedding_gen.embed_query(query)
        retrieved_chunks = vector_store.search(query_emb, top_k=settings.RETRIEVAL_TOP_K)
        retrieval_method = "dense"
    elif len(bm25_retriever.chunks) > 0:
        retrieved_chunks = bm25_retriever.search(query, top_k=settings.RETRIEVAL_TOP_K)
        retrieval_method = "bm25"
    else:
        raise RuntimeError("No retrieval index is loaded on this server instance.")

    retrieval_latency = (time.perf_counter() - t0) * 1000

    # 2. Fast Pre-Generation Abstention Guard Check
    from backend.generation.guard import check_pre_retrieval_guard
    pre_guard = check_pre_retrieval_guard(query, retrieved_chunks)
    if pre_guard is not None:
        logger.info(f"[Fast Guard] Abstaining before LLM invocation: {pre_guard['guard_reason']}")
        return {
            "answer": pre_guard["answer"],
            "sources": pre_guard.get("sources", []),
            "retrieval_latency": retrieval_latency,
            "generation_latency": 0.0,
            "retrieval_method": retrieval_method,
            "provider": settings.GENERATION_PROVIDER,
            "guard_triggered": True,
            "guard_reason": pre_guard.get("guard_reason"),
            "chunks_count": len(retrieved_chunks)
        }

    # 3. Generation Provider Selection
    provider_name = settings.GENERATION_PROVIDER.lower()
    t0 = time.perf_counter()

    if provider_name == "sarvam":
        if not settings.SARVAM_API_KEY or settings.SARVAM_API_KEY == "your_sarvam_api_key_here":
            raise ValueError("SARVAM_API_KEY is not configured for production generation.")
        generator = SarvamGenerator()
    else:
        generator = MockGenerator()

    candidate_response = generator.generate(query, retrieved_chunks)

    # 4. Grounding Guard
    try:
        final_response = validate_generation(query, retrieved_chunks, candidate_response)
        guard_triggered = final_response.get("guard_triggered", False)
        guard_reason = final_response.get("guard_reason", None)
    except Exception as e:
        logger.error(f"Grounding guard error: {e}", exc_info=True)
        final_response = candidate_response
        guard_triggered = False
        guard_reason = None

    generation_latency = (time.perf_counter() - t0) * 1000

    return {
        "answer": final_response["answer"],
        "sources": final_response.get("sources", []),
        "retrieval_latency": retrieval_latency,
        "generation_latency": generation_latency,
        "retrieval_method": retrieval_method,
        "provider": provider_name,
        "guard_triggered": guard_triggered,
        "guard_reason": guard_reason,
        "chunks_count": len(retrieved_chunks)
    }


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse)
@app.post("/query", response_model=QueryResponse)
@app.post("/api/index.py/api/query", response_model=QueryResponse)
@app.post("/api/index.py/query", response_model=QueryResponse)
def handle_query(payload: QueryRequest):
    """
    Text-based query endpoint executing retrieval -> generation -> guard.
    """
    query = payload.query.strip()
    req_start_time = time.perf_counter()
    logger.info(f"[Text Query] Received query='{query}'")

    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty or whitespace."
        )

    try:
        result = _execute_rag_pipeline(query)
    except ValueError as e:
        logger.error(f"Configuration error in pipeline: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "CONFIGURATION_ERROR", "message": str(e)}
        )
    except (TimeoutError, RuntimeError) as e:
        logger.error(f"Pipeline failure: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": "SERVICE_UNAVAILABLE", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"Unexpected query failure: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "QUERY_FAILED", "message": "An error occurred while processing the query."}
        )

    total_latency = (time.perf_counter() - req_start_time) * 1000
    logger.info(
        f"[Text Query] Complete: method='{result['retrieval_method']}', "
        f"chunks={result['chunks_count']}, guard={result['guard_triggered']}, "
        f"ret_lat={result['retrieval_latency']:.2f}ms, gen_lat={result['generation_latency']:.2f}ms, "
        f"tot_lat={total_latency:.2f}ms"
    )

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        retrieval=LatencyDetails(latency_ms=result["retrieval_latency"]),
        generation=LatencyDetails(latency_ms=result["generation_latency"]),
        provider=result["provider"],
        guard_triggered=result["guard_triggered"],
        guard_reason=result["guard_reason"]
    )


@app.post("/api/voice", response_model=VoiceResponse)
@app.post("/voice", response_model=VoiceResponse)
@app.post("/api/index.py/api/voice", response_model=VoiceResponse)
@app.post("/api/index.py/voice", response_model=VoiceResponse)
async def handle_voice(
    file: UploadFile = File(..., description="Recorded audio clip from microphone"),
    language_code: Optional[str] = Form(None, description="Optional language code (e.g., 'hi-IN')")
):
    """
    End-to-end Voice RAG endpoint:
    Microphone Audio -> Speech-To-Text (Sarvam/ElevenLabs) -> Query Normalization ->
    Dense/BM25 Retrieval -> Grounded Generation -> Grounding Guard -> Structured Response.
    """
    total_start_time = time.perf_counter()
    filename = file.filename or "recording.webm"
    content_type = file.content_type or "audio/webm"

    logger.info(f"[Voice Query] Audio received: filename='{filename}', content_type='{content_type}'")

    # 1. Read and validate audio bytes
    try:
        audio_bytes = await file.read()
    except Exception as e:
        logger.error(f"[Voice Query] Failed to read uploaded audio: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "INVALID_AUDIO", "message": "Could not read audio stream from request."}
        )

    if not audio_bytes or len(audio_bytes) == 0:
        logger.warning("[Voice Query] Empty audio stream (0 bytes).")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "EMPTY_AUDIO", "message": "Uploaded audio recording is empty (0 bytes)."}
        )

    # 2. Speech-to-Text Transcription
    try:
        stt_provider = get_stt_provider()
        stt_result = stt_provider.transcribe(
            audio_bytes=audio_bytes,
            filename=filename,
            mime_type=content_type,
            language_code=language_code
        )
    except ValueError as e:
        logger.error(f"[Voice Query] STT Configuration/Input error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "CONFIGURATION_ERROR", "message": str(e)}
        )
    except TimeoutError as e:
        logger.error(f"[Voice Query] STT Timeout: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": "STT_TIMEOUT", "message": "Speech transcription timed out. Please try speaking again."}
        )
    except Exception as e:
        err_msg = str(e)
        logger.error(f"[Voice Query] STT Failure: {e}", exc_info=True)
        if "HTTP 400" in err_msg or "400" in err_msg:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "STT_BAD_REQUEST", "message": f"Speech transcription rejected by STT provider: {err_msg}"}
            )
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"error": "STT_FAILED", "message": f"Speech transcription failed: {err_msg}"}
        )

    raw_transcript = stt_result.transcript.strip()
    if not raw_transcript:
        logger.warning("[Voice Query] No speech recognized in audio.")
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": False,
                "transcript": "",
                "normalized_query": "",
                "language": stt_result.language_code or "unknown",
                "answer": "आवाज़ स्पष्ट रूप से सुनाई नहीं दी। कृपया पुनः बोलें। (No clear speech detected. Please speak again.)",
                "sources": [],
                "provider": settings.GENERATION_PROVIDER,
                "stt_provider": stt_result.provider,
                "guard_triggered": True,
                "guard_reason": "No speech detected in audio clip",
                "latency": {
                    "stt_ms": stt_result.duration_ms,
                    "retrieval_ms": 0.0,
                    "generation_ms": 0.0,
                    "total_ms": (time.perf_counter() - total_start_time) * 1000
                }
            }
        )

    # 3. Query Normalization
    normalized_query = normalize_voice_query(raw_transcript)
    logger.info(f"[Voice Query] Transcribed: '{raw_transcript}' -> Normalized: '{normalized_query}'")

    # 4. Execute RAG Retrieval & Generation Pipeline
    try:
        rag_result = _execute_rag_pipeline(normalized_query)
    except ValueError as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "CONFIGURATION_ERROR", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"[Voice Query] RAG Subsystem Failure: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "RAG_PIPELINE_FAILED", "message": "Failed to retrieve evidence or generate answer."}
        )

    total_pipeline_ms = (time.perf_counter() - total_start_time) * 1000

    logger.info(
        f"[Voice Query Complete] STT={stt_result.duration_ms:.1f}ms, "
        f"Retrieval={rag_result['retrieval_latency']:.1f}ms, "
        f"Gen={rag_result['generation_latency']:.1f}ms, "
        f"Total={total_pipeline_ms:.1f}ms"
    )

    return VoiceResponse(
        success=True,
        transcript=raw_transcript,
        normalized_query=normalized_query,
        language=stt_result.language_code or "hi-IN",
        answer=rag_result["answer"],
        sources=rag_result["sources"],
        provider=rag_result["provider"],
        stt_provider=stt_result.provider,
        guard_triggered=rag_result["guard_triggered"],
        guard_reason=rag_result["guard_reason"],
        latency=VoiceLatencyBreakdown(
            stt_ms=stt_result.duration_ms,
            retrieval_ms=rag_result["retrieval_latency"],
            generation_ms=rag_result["generation_latency"],
            total_ms=total_pipeline_ms
        )
    )


# ─── Health & Diagnostics ────────────────────────────────────────────────────
@app.get("/", tags=["ops"])
@app.get("/api", tags=["ops"])
@app.get("/health", tags=["ops"])
@app.get("/api/health", tags=["ops"])
@app.get("/api/index.py/health", tags=["ops"])
@app.get("/api/index.py/api/health", tags=["ops"])
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
        "pipeline_type": "voice-enabled-rag",
        "retrieval": "ready" if is_ready else "degraded",
        "retrieval_backend": "dense" if embedding_gen is not None else "bm25",
        "bm25_loaded": len(bm25_retriever.chunks) > 0,
        "bm25_chunks": len(bm25_retriever.chunks),
        "dense_chunks": len(vector_store.chunks_metadata),
        "index_path_exists": dense_path is not None or bm25_path is not None,
        "chunk_strategy": settings.CHUNK_STRATEGY,
        "generation_provider": settings.GENERATION_PROVIDER,
        "stt_provider": settings.STT_PROVIDER,
        "stt_model": settings.SARVAM_STT_MODEL,
        "sarvam_configured": has_sarvam_key
    })


# ─── Universal Vercel API Dispatcher ──────────────────────────────────────────
@app.api_route("/api", methods=["GET", "POST", "HEAD", "OPTIONS"])
@app.api_route("/api/", methods=["GET", "POST", "HEAD", "OPTIONS"])
@app.api_route("/api/index", methods=["GET", "POST", "HEAD", "OPTIONS"])
@app.api_route("/api/index.py", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def vercel_universal_api_dispatcher(request: Request):
    """
    Universal dispatcher for Vercel serverless when requests are routed to /api.
    Inspects x-matched-path, content-type, or request method to invoke the exact handler.
    """
    target = (request.headers.get("x-matched-path") or request.headers.get("x-forwarded-uri") or request.url.path).lower()
    
    if request.method == "GET" or "health" in target:
        return health_check()
    elif request.method == "POST":
        content_type = request.headers.get("content-type", "")
        if "multipart" in content_type or "voice" in target:
            form = await request.form()
            file = form.get("file")
            lang = form.get("language_code")
            return await handle_voice(file=file, language_code=lang)
        else:
            try:
                body = await request.json()
                return handle_query(QueryRequest(**body))
            except Exception as e:
                logger.error(f"Failed to parse query payload in universal dispatcher: {e}")
                return JSONResponse(status_code=400, content={"error": "INVALID_JSON", "message": str(e)})
                
    return health_check()


# ─── Frontend SPA Mount (for local development only) ─────────────────────────
import os
if not os.getenv("VERCEL") and not os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
    _frontend_dir = PROJECT_ROOT / "public" if (PROJECT_ROOT / "public").is_dir() else (PROJECT_ROOT / "frontend")
    if _frontend_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(_frontend_dir), html=True), name="static")

