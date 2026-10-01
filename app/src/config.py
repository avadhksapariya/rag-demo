import os

from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Base project root directory
APP_DIR = Path(__file__).resolve().parent.parent


# ── Embedding ─────────────────────────────────────────────────────────────────
# Local - Downloaded once from Hugging Face and then runs locally.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # to ingest and query both.
# gemini-embedding-2


# ── LLM ───────────────────────────────────────────────────────────────────────
# Primary: OpenRouter Free -> Fallback: Gemini
# to generate the final answer.
OPENROUTER_FREE_MODEL = "openrouter/free"
GEMINI_LLM_MODEL = "gemini-3.6-flash"


# Paths
DB_PATH = APP_DIR / "vector_db"
DATA_DIR = APP_DIR / "data"
PDF_PATH = DATA_DIR / "SamudraManthan-ChurningOfTheOcean.pdf"  # single-file fallback

# Multi-stage Retrieval Settings
INITIAL_RETRIEVAL_K = 15  # Stage 1: ChromaDB candidate retrieval
FINAL_RETRIEVAL_K = 5  # Stage 2: Reranked top candidates sent to LLM


# PROVIDER CONFIGURATION
OPENROUTER_CONFIG = {
    "api_key": os.getenv("OPENROUTER_API_KEY"),
    "base_url": "https://openrouter.ai/api/v1",
}


def get_llm():
    gemini_llm = ChatGoogleGenerativeAI(model=GEMINI_LLM_MODEL)

    if not OPENROUTER_CONFIG["api_key"]:
        return gemini_llm

    openrouter_llm = ChatOpenAI(
        model=OPENROUTER_FREE_MODEL,
        openai_api_key=OPENROUTER_CONFIG["api_key"],
        openai_api_base=OPENROUTER_CONFIG["base_url"],
        default_headers={
            "HTTP-Referer": "https://test-admin.zigzek.com/",  # http://localhost:8501
            "X-Title": "Zigzek AI",  # RAG Demo
        },
        max_retries=1,
    )

    # LangChain automatic failover
    return openrouter_llm.with_fallbacks([gemini_llm])
