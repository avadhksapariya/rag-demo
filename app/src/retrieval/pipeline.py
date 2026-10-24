# app/src/retrieval/pipeline.py
from pathlib import Path
from typing import List, Tuple
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

from app.src.ingestion.embeddings import get_embeddings
from app.core.config import INITIAL_RETRIEVAL_K, FINAL_RETRIEVAL_K
from app.src.retrieval.web_search import search_web_fallback
from app.src.retrieval.hybrid import ParentBM25Retriever, reciprocal_rank_fusion
from app.src.retrieval.query_expansion import (
    generate_multi_queries,
    deduplicate_documents,
)
from app.src.context.token_budget import trim_documents_to_budget


class RetrievalGrade(BaseModel):
    is_relevant: bool = Field(
        description="True if the retrieved context contains sufficient facts or context to answer the user question. False if it is irrelevant, out-of-scope, or missing."
    )


# Initializes Chroma dense ParentDocumentRetriever and sparse ParentBM25Retriever.
def initialize_retrievers(db_path: Path, selected_file: str | None = None):
    chroma_path = db_path / "chroma"
    if not chroma_path.exists():
        raise FileNotFoundError(
            f"Vector store not found at '{chroma_path}'. Please run ingestion first!"
        )

    embeddings = get_embeddings()
    vector_db = Chroma(
        collection_name="child_chunks",
        persist_directory=str(chroma_path),
        embedding_function=embeddings,
    )

    fs = LocalFileStore(str(db_path / "doc_store"))
    store = create_kv_docstore(fs)

    search_kwargs = {"k": INITIAL_RETRIEVAL_K}
    if selected_file and selected_file != "All Documents":
        search_kwargs["filter"] = {"source_file": selected_file}

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

    dense_pdr = ParentDocumentRetriever(
        vectorstore=vector_db,
        docstore=store,
        child_splitter=child_splitter,
        search_kwargs=search_kwargs,
    )

    sparse_pdr = ParentBM25Retriever(
        vector_db=vector_db,
        store=store,
        k=INITIAL_RETRIEVAL_K,
    )

    compressor = FlashrankRerank(top_n=FINAL_RETRIEVAL_K)
    return dense_pdr, sparse_pdr, compressor


# Generates sub-queries, executes hybrid retrieval across branches, deduplicates, reranks, and budgets tokens.
def execute_multi_query_retrieval(
    standalone_query: str,
    selected_file: str | None,
    dense_pdr: ParentDocumentRetriever,
    sparse_pdr: ParentBM25Retriever,
    compressor: FlashrankRerank,
    llm,
) -> List[Document]:
    retrieval_queries = generate_multi_queries(standalone_query, llm)
    print(f"🔀 [Multi-Query Expansion]: Sub-queries: {retrieval_queries}")

    all_hybrid_candidates = []
    for q in retrieval_queries:
        dense_hits = dense_pdr.invoke(q)
        sparse_hits = sparse_pdr.invoke(q, selected_file=selected_file)
        fused = reciprocal_rank_fusion(
            dense_docs=dense_hits,
            sparse_docs=sparse_hits,
            top_n=INITIAL_RETRIEVAL_K,
        )
        all_hybrid_candidates.extend(fused)

    deduped_candidates = deduplicate_documents(all_hybrid_candidates)
    print(
        f"📚 [Retrieval Collator]: {len(all_hybrid_candidates)} chunks -> {len(deduped_candidates)} unique candidates."
    )

    if not deduped_candidates:
        return []

    # Cross-encoder rerank against the primary query
    reranked = compressor.compress_documents(
        documents=deduped_candidates, query=standalone_query
    )

    # Dynamic token budgeting (cap at 2500 context tokens)
    budgeted_docs, used_tokens = trim_documents_to_budget(reranked, max_tokens=2500)
    print(
        f"📦 [Context Budget]: Packed {len(budgeted_docs)} chunk(s) totaling ~{used_tokens} tokens."
    )
    return budgeted_docs


# CRAG: Evaluates retrieved document relevance; falls back to DuckDuckGo if insufficient.
def evaluate_and_fallback(
    standalone_query: str, retrieved_docs: List[Document], llm
) -> Tuple[List[Document], bool]:
    if not retrieved_docs:
        is_relevant = False
    else:
        context_preview = "\n".join([doc.page_content for doc in retrieved_docs])
        parser = JsonOutputParser(pydantic_object=RetrievalGrade)
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a strict factual retrieval evaluator.\n"
                    "Determine whether the retrieved context contains the actual, specific information "
                    "needed to answer the user's question.\n"
                    "- Mark 'is_relevant: false' if the context merely mentions related characters or themes "
                    "but DOES NOT actually state the answer to the question.\n"
                    "- Mark 'is_relevant: true' ONLY if the text directly provides the facts required to answer the question.\n"
                    "{format_instructions}",
                ),
                ("human", "Question: {question}\n\nRetrieved Context:\n{context}"),
            ]
        )
        grade_chain = grade_prompt | llm | parser
        try:
            grade_result = grade_chain.invoke(
                {
                    "question": standalone_query,
                    "context": context_preview,
                    "format_instructions": parser.get_format_instructions(),
                }
            )
            is_relevant = grade_result.get("is_relevant", True)
        except Exception:
            is_relevant = True

    if is_relevant:
        return retrieved_docs, False

    clean_search_query = (
        standalone_query.replace("**", "").replace('"', "").strip().split("\n")[0]
    )
    web_docs = search_web_fallback(clean_search_query, max_results=3)
    if web_docs:
        return web_docs, True
    return [], False
