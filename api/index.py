"""
Vercel Python serverless entry point for Gyaan RAG.

This file re-exports the FastAPI application object from backend/api/app.py.
Vercel looks for a callable named `app` (or `handler`) in api/index.py.

IMPORTANT — Deployment Constraints:
====================================
The full local RAG pipeline (dense retrieval + multilingual-e5-small) cannot
run on Vercel serverless functions because:

  1. The embedding model (intfloat/multilingual-e5-small) is ~470 MB.
     Vercel serverless functions have a 250 MB size limit.

  2. The architecture is stateful (model loaded at startup).
     Vercel functions are ephemeral and stateless.

What DOES work on Vercel:
  - Serving the frontend (HTML/CSS/JS)
  - /health endpoint
  - /api/query when GENERATION_PROVIDER=sarvam and indexes are small enough

For full dense retrieval in production, deploy the backend on:
  - Railway  (https://railway.app)
  - Render   (https://render.com)
  - Fly.io   (https://fly.io)

And point the frontend API calls to that backend's URL.
"""
import sys
import os
from pathlib import Path

# Ensure the project root is on sys.path so that `backend.*` imports resolve
# regardless of where Vercel invokes this module.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Re-export the FastAPI app — Vercel will call this as the ASGI handler.
from backend.api.app import app  # noqa: F401
