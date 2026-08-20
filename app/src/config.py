from pathlib import Path

# Base project root directory
APP_DIR = Path(__file__).resolve().parent.parent

# Models

# Local embedding model.
# Downloaded once from Hugging Face and then runs locally.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # to ingest and query both.
# gemini-embedding-2

LLM_MODEL = "gemini-3.6-flash"  # to generate the final answer.

# Paths
DB_PATH = APP_DIR / "vector_db"
DATA_DIR = APP_DIR / "data"
PDF_PATH = DATA_DIR / "SamudraManthan-ChurningOfTheOcean.pdf"  # single-file fallback

# Multi-stage Retrieval Settings
INITIAL_RETRIEVAL_K = 10  # Stage 1: ChromaDB candidate retrieval
FINAL_RETRIEVAL_K = 3  # Stage 2: Reranked top candidates sent to Gemini
