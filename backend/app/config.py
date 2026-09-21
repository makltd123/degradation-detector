"""Конфигурация сервиса: все настройки читаются из окружения / backend/.env."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DD_DATA_DIR", BACKEND_DIR / "data"))
DB_PATH = DATA_DIR / "detector.db"

# --- LLM (одна и та же модель может играть все роли: мутации / прод-модель / судья) ---
PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()  # openai | anthropic
API_KEY = (
    os.getenv("LLM_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("ANTHROPIC_API_KEY")
    or ""
)
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
WEAK_MODEL = os.getenv("LLM_WEAK_MODEL", "gpt-3.5-turbo")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.6"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1600"))

# --- пайплайн прогона ---
SOURCE_QUERY_SAMPLE = int(os.getenv("DD_SOURCE_QUERY_SAMPLE", "15"))
MUTATIONS_PER_QUERY = int(os.getenv("DD_MUTATIONS_PER_QUERY", "2"))
TOP_K_ARTICLES = int(os.getenv("DD_TOP_K_ARTICLES", "4"))
BASELINE_RUNS_LIMIT = int(os.getenv("DD_BASELINE_RUNS_LIMIT", "3"))

# Разделение стоимости оценки: semantic similarity считается для всех
# ответов, LLM-judge — в зависимости от режима:
#   all                 — судья оценивает каждый ответ (дефолт прототипа)
#   low_similarity_only — судья только там, где similarity < порога (экономия)
JUDGE_MODE = os.getenv("DD_JUDGE_MODE", "all")
JUDGE_SIMILARITY_THRESHOLD = float(os.getenv("DD_JUDGE_SIMILARITY_THRESHOLD", "0.85"))

EMBEDDING_MODEL = os.getenv(
    "DD_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
