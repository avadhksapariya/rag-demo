from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
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

from src.config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    DB_PATH,
    INITIAL_RETRIEVAL_K,
    FINAL_RETRIEVAL_K,
)


# Formats retrieved chunks with clear page markers for inline citing.
def format_docs_with_page_number(docs):
    formatted_chunks = []
    for doc in docs:
        page = doc.metadata.get("page", 0) + 1
        chunk_text = f"--- [Page {page}] ---\n{doc.page_content}"
        formatted_chunks.append(chunk_text)
    return "\n\n".join(formatted_chunks)


# Retrieves context using chat history and returns an answer with citations.
def answer_question(
    user_query: str, chat_history: list, db_path: str | Path = DB_PATH
) -> dict:
    db_path = Path(db_path)

    if not db_path.exists():
        raise FileNotFoundError(
            f"Vector store not found at '{db_path}'. Please run ingestion first!"
        )

    # Load Vector Store & Base Retriever (Stage 1: Fetch 10 candidates)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vector_db = Chroma(
        persist_directory=str(db_path),
        embedding_function=embeddings,
    )
    base_retriever = vector_db.as_retriever(search_kwargs={"k": INITIAL_RETRIEVAL_K})

    compressor = FlashrankRerank(top_n=FINAL_RETRIEVAL_K)
    rerank_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
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
        "--- [Page {page}] ---\n{page_content}"
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
        "1. Whenever you state a fact from the context, include an inline citation with the page number, e.g., [Page X].\n"
        "2. Only cite pages that are explicitly provided in the context header markers (e.g., --- [Page X] ---).\n\n"
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
