from pathlib import Path
import os

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_DIR / ".env")
PROJECT_ROOT = BACKEND_DIR.parent
AI_OUTPUTS_DIR = PROJECT_ROOT / "AI" / "outputs"
MODEL_DIR = BACKEND_DIR / "model"
MODEL_PATH = AI_OUTPUTS_DIR / "model_torchscript.pt"
MODEL_STATE_DICT_PATH = AI_OUTPUTS_DIR / "model_state_dict.pt"
PROCESSED_GRAPH_PATH = AI_OUTPUTS_DIR / "processed_graph.pt"
MODEL_METADATA_PATH = AI_OUTPUTS_DIR / "metrics.json"
BACKEND_MODEL_PATH = MODEL_DIR / "model.pkl"
BACKEND_MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")

SIMULATION_STEPS = 10
ALPHA = 0.7

APP_NAME = "ACWS Backend"
APP_VERSION = "1.0.0"
LOG_LEVEL = "INFO"
REQUEST_TIMEOUT_SECONDS = 60
RANDOM_SEED = 42
ALLOW_LLM_FALLBACK = os.getenv("ALLOW_LLM_FALLBACK", "true").strip().lower() in {"1", "true", "yes", "on"}

FEATURE_NAMES = ["velocity", "bot_ratio", "echo_density", "depth"]

DEFAULT_CORS_ORIGINS = [
	"http://localhost:3000",
	"http://127.0.0.1:3000",
]

_cors_origins_env = os.getenv("CORS_ORIGINS", "")
CORS_ORIGINS = [origin.strip() for origin in _cors_origins_env.split(",") if origin.strip()] or DEFAULT_CORS_ORIGINS