import sys
from pathlib import Path

# Add project root to sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from backend.api.app import app

# Explicit top-level FastAPI instance for Vercel AST parser
handler = app
