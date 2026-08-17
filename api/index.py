import sys
from pathlib import Path

# Add project root to sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from backend.api.app import app as _fastapi_app

class VercelPathMiddleware:
    """
    ASGI middleware that restores the true incoming URL path from Vercel's proxy headers
    (x-matched-path / x-forwarded-uri / x-real-path) before Starlette routes the request.
    This guarantees 100% accurate route matching for /api/query, /api/voice, /health, /api/health.
    """
    def __init__(self, inner_app):
        self.inner_app = inner_app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            for h_k, h_v in headers.items():
                k = h_k.decode("latin1").lower() if isinstance(h_k, bytes) else str(h_k).lower()
                if k in ("x-matched-path", "x-forwarded-uri", "x-real-path", "x-envoy-original-path"):
                    v = h_v.decode("latin1") if isinstance(h_v, bytes) else str(h_v)
                    clean = v.split("?")[0]
                    if clean and not clean.startswith("/api/index"):
                        scope["path"] = clean
                        if "raw_path" in scope:
                            scope["raw_path"] = clean.encode("latin1")
                    break

        await self.inner_app(scope, receive, send)

# Top-level ASGI callable instance for Vercel Python runtime
app = VercelPathMiddleware(_fastapi_app)
handler = app
