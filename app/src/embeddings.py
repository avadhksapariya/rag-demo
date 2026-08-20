from functools import lru_cache
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

from app.src.config import EMBEDDING_MODEL


class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        print(f"Loading local embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> LocalSentenceTransformerEmbeddings:
    return LocalSentenceTransformerEmbeddings(EMBEDDING_MODEL)
