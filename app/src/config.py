from pathlib import Path

# Base project root directory
APP_DIR = Path(__file__).resolve().parent.parent

# Models
EMBEDDING_MODEL = "gemini-embedding-2"  # Embedding model BOTH ingestion and querying.
LLM_MODEL = "gemini-3.6-flash"  # LLM used to generate the final answer.

# Paths
DB_PATH = APP_DIR / "vector_db"
DATA_DIR = APP_DIR / "data"
PDF_PATH = DATA_DIR / "SamudraManthan-ChurningOfTheOcean.pdf"  # single-file fallback

# Multi-stage Retrieval Settings
INITIAL_RETRIEVAL_K = 10  # Stage 1: ChromaDB candidate retrieval
FINAL_RETRIEVAL_K = 3  # Stage 2: Reranked top candidates sent to Gemini
