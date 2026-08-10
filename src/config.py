from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Embedding model used for BOTH ingestion and querying.
EMBEDDING_MODEL = "gemini-embedding-2"

# LLM used to generate the final answer.
LLM_MODEL = "gemini-3.6-flash"

DB_PATH = BASE_DIR / "vector_db"

PDF_PATH = BASE_DIR / "data" / "SamudraManthan-ChurningOfTheOcean.pdf"

# Stage 1: Fetch 10 initial candidates from ChromaDB
INITIAL_RETRIEVAL_K = 10

# Stage 2: Reranker selects top 3 best candidates for Gemini
FINAL_RETRIEVAL_K = 3
