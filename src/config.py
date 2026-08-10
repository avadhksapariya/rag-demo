# Embedding model used for BOTH ingestion and querying.
EMBEDDING_MODEL = "gemini-embedding-2"

# LLM used to generate the final answer.
LLM_MODEL = "gemini-3.6-flash"

DB_PATH = "./vector_db"

PDF_PATH = "data/SamudraManthan-ChurningOfTheOcean.pdf"

# Number of chunks retrieved for each question.
RETRIEVAL_K = 3
