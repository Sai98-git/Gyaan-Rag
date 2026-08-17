import sys
from pathlib import Path

# Add project root to sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from backend.api.app import app as _fastapi_app

async def app(scope, receive, send):
    """
    ASGI middleware wrapper for Vercel Serverless.
    
    When Vercel rewrites /api/query, /api/voice, /health to /api/index.py,
    it may set ASGI scope['path'] to '/api/index.py' or include the serverless file prefix.
    This middleware extracts the original requested path from Vercel proxy headers
    (x-matched-path, x-forwarded-uri, x-real-path) or strips '/api/index.py' prefix
    so that FastAPI routes match with 100% precision.
    """
    if scope["type"] == "http":
        headers = dict(scope.get("headers", []))
        
        # Check for original requested path in Vercel proxy headers
        matched_path = None
        for header_name, header_val in headers.items():
            h_name = header_name.decode("latin1").lower() if isinstance(header_name, bytes) else str(header_name).lower()
            if h_name in ("x-matched-path", "x-forwarded-uri", "x-real-path", "x-envoy-original-path"):
                h_val = header_val.decode("latin1") if isinstance(header_val, bytes) else str(header_val)
                matched_path = h_val.split("?")[0]
                break

        current_path = scope.get("path", "")
        
        if current_path in ("/api/index.py", "/api/index", "/api/index.py/", "/api/index/"):
            if matched_path and not matched_path.startswith("/api/index"):
                scope["path"] = matched_path
        elif current_path.startswith("/api/index.py/"):
            scope["path"] = current_path[len("/api/index.py"):]
        elif current_path.startswith("/api/index/"):
            scope["path"] = current_path[len("/api/index"):]

    await _fastapi_app(scope, receive, send)
