from typing import List, Optional, Dict
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


# BM25 retriever that searches child chunks and retrieves
# their corresponding Parent Documents from the KV docstore.
class ParentBM25Retriever:
    def __init__(self, vector_db: Chroma, store, k: int = 15):
        self.vector_db = vector_db
        self.store = store
        self.k = k
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._initialize_bm25()

    # Fetch all child documents from ChromaDB
    def _initialize_bm25(self):
        data = self.vector_db.get(include=["documents", "metadatas"])
        texts = data.get("documents", [])
        metadatas = data.get("metadatas", [])

        if not texts:
            return

        child_docs = [
            Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)
        ]

        # Initialize BM25 on child chunks
        self._bm25_retriever = BM25Retriever.from_documents(child_docs)
        self._bm25_retriever.k = (
            self.k * 2
        )  # Retrieve extra child chunks to account for duplicates

    def invoke(self, query: str, selected_file: Optional[str] = None) -> List[Document]:
        if not self._bm25_retriever:
            return []

        # Sparse keyword search on child chunks
        child_matches = self._bm25_retriever.invoke(query)

        # Filter by selected_file if applicable
        if selected_file and selected_file != "All Documents":
            child_matches = [
                doc
                for doc in child_matches
                if doc.metadata.get("source_file") == selected_file
            ]

        # Resolve parent IDs while preserving ranking order & avoiding duplicate parents
        parent_ids = []
        for doc in child_matches:
            pid = doc.metadata.get("doc_id")
            if pid and pid not in parent_ids:
                parent_ids.append(pid)
            if len(parent_ids) >= self.k:
                break

        # Fetch Parent Documents from docstore
        parent_docs = self.store.mget(parent_ids)
        return [doc for doc in parent_docs if doc is not None]


# Combines ranked results from Dense and Sparse retrievers using RRF.
# Score = 1 / (rrf_k + rank)
def reciprocal_rank_fusion(
    dense_docs: List[Document],
    sparse_docs: List[Document],
    rrf_k: int = 60,
    top_n: int = 15,
) -> List[Document]:
    fusion_scores: Dict[str, float] = {}
    doc_lookup: Dict[str, Document] = {}

    # Create a unique key based on source file and content snippet
    def _process_ranked_list(doc_list: List[Document]):
        for rank, doc in enumerate(doc_list):
            key = f"{doc.metadata.get('source_file', '')}_{doc.page_content[:100]}"
            if key not in doc_lookup:
                doc_lookup[key] = doc
                fusion_scores[key] = 0.0
            fusion_scores[key] += 1.0 / (rrf_k + rank + 1)

    _process_ranked_list(dense_docs)
    _process_ranked_list(sparse_docs)

    # Sort documents by accumulated RRF score descending
    sorted_keys = sorted(
        fusion_scores.keys(), key=lambda k: fusion_scores[k], reverse=True
    )
    return [doc_lookup[k] for k in sorted_keys[:top_n]]
