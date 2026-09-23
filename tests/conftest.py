from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
AI_DIR = REPO_ROOT / "AI"

for path in (BACKEND_DIR, AI_DIR):
    str_path = str(path)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)
