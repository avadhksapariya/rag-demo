from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_classic.chains.history_aware_retriever import (
    create_history_aware_retriever,
)
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

from app.src.config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    DB_PATH,
    INITIAL_RETRIEVAL_K,
    FINAL_RETRIEVAL_K,
)


# Retrieves context using chat history and returns an answer with citations.
def answer_question(
    user_query: str,
    chat_history: list,
    db_path: str | Path = DB_PATH,
    selected_file: str | None = None,
) -> dict:
    db_path = Path(db_path)
    chroma_path = db_path / "chroma"

    if not chroma_path.exists():
        raise FileNotFoundError(
            f"Vector store not found at '{chroma_path}'. Please run ingestion first!"
        )

    # Load Chroma VectorDB & Local DocStore (Stage 1: Fetch 10 candidates)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
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

    # Base Parent Document Retriever
    base_pdr = ParentDocumentRetriever(
        vectorstore=vector_db,
        docstore=store,
        child_splitter=child_splitter,
        search_kwargs=search_kwargs,
    )

    # Two-Stage Reranker over Parent Documents
    compressor = FlashrankRerank(top_n=FINAL_RETRIEVAL_K)
    rerank_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_pdr
    )

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)

    # History-Aware Retriever Prompt
    # Reformulates follow-up queries into standalone search terms
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, rerank_retriever, contextualize_q_prompt
    )

    # Define how EACH document chunk is formatted before insertion into context
    document_prompt = PromptTemplate.from_template(
        "--- [{source_file} | Page {page}] ---\n{page_content}"
    )

    # Question-Answering Prompt (uses context and chat history)
    system_prompt = (
        "You are a helpful assistant for question-answering tasks.\n"
        "Use the following pieces of retrieved context to answer the question.\n"
        "If you don't know the answer or if it is not present in the context, say "
        '"I don\'t know based on the provided document."\n'
        "Do not use outside knowledge.\n"
        "Keep the answer concise and clear.\n\n"
        "CITATION INSTRUCTIONS:\n"
        "1. Include inline citations like [FileName | Page X] for facts stated in your answer.\n"
        "2. Use the exact source name and page number shown in the header markers.\n\n"
        "Retrieved context:\n{context}"
    )
    # "you don't know. Do not hallucinate.\n\n"

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Combine into final chain
    question_answer_chain = create_stuff_documents_chain(
        llm,
        prompt=qa_prompt,
        document_prompt=document_prompt,
        document_variable_name="context",
    )
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain.invoke({"input": user_query, "chat_history": chat_history})
