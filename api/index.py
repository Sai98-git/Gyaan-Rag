"""
Vercel Python serverless entry point for Gyaan RAG.
Exposes the FastAPI application object to Vercel ASGI runtime.
"""
import sys
import os
from pathlib import Path

# Robust sys.path configuration — ensures backend imports resolve in any deployment layout
_this_dir = Path(__file__).resolve().parent
_parent_dir = _this_dir.parent
_cwd = Path.cwd()

for p in [_this_dir, _parent_dir, _cwd, _parent_dir.parent]:
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

# Import the FastAPI app, with diagnostic fallback if import fails
try:
    from backend.api.app import app
except Exception as exc:
    import traceback
    _error_traceback = traceback.format_exc()
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI(title="Gyaan RAG (Diagnostic Mode)")

    @app.get("/health")
    def health_diagnostic():
        return JSONResponse(
            {
                "status": "startup_error",
                "error": str(exc),
                "traceback": _error_traceback,
                "sys_path": sys.path[:5],
                "cwd": str(Path.cwd()),
            },
            status_code=500
        )

    @app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE"])
    def catch_all_diagnostic(path_name: str):
        return JSONResponse(
            {
                "status": "startup_error",
                "error": str(exc),
                "traceback": _error_traceback,
                "sys_path": sys.path[:5],
                "cwd": str(Path.cwd()),
            },
            status_code=500
        )
