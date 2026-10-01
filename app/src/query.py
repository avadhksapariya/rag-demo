import re

from pathlib import Path
from typing import Union, Generator
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

from app.src.embeddings import get_embeddings
from app.src.config import (
    get_llm,
    DB_PATH,
    INITIAL_RETRIEVAL_K,
    FINAL_RETRIEVAL_K,
)

pleasantry_words = {
    "thanks",
    "thank",
    "you",
    "perfect",
    "great",
    "ok",
    "okay",
    "got",
    "it",
    "cool",
    "understood",
    "awesome",
    "bye",
    "goodbye",
    "done",
    "alright",
    "clear",
    "helpful",
    "appreciated",
}


class RetrievalGrade(BaseModel):
    is_relevant: bool = Field(
        description="True if the retrieved context contains sufficient facts or context to answer the user question. False if it is irrelevant, out-of-scope, or missing."
    )


class UserIntent(BaseModel):
    intent: str = Field(
        description="Classify as 'chitchat' if the message is a greeting, reaction, compliment, pleasantry, acknowledgment (e.g., 'Amazing', 'Thanks', 'Cool', 'Haha wow', 'Understood'), or casual banter. Classify as 'rag_query' if the user is asking a factual question, seeking information, or following up on document content."
    )
    direct_response: str = Field(
        default="",
        description="If intent is 'chitchat', provide a short, warm, natural conversational reply (e.g. 'Glad you found it interesting! Let me know if you need anything else.'). If 'rag_query', leave this empty.",
    )


# Classifies user input dynamically using the LLM before any vector retrieval.
def _classify_intent(user_query: str, llm) -> UserIntent:
    parser = JsonOutputParser(pydantic_object=UserIntent)
    intent_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an input intent classifier for an intelligent assistant.\n"
                "Determine whether the user is casually interacting/acknowledging or asking a genuine question.\n"
                "{format_instructions}",
            ),
            ("human", "{input}"),
        ]
    )
    chain = intent_prompt | llm | parser
    try:
        res = chain.invoke(
            {
                "input": user_query,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        return UserIntent(**res)
    except Exception:
        # Default to standard RAG pipeline if parsing fails
        return UserIntent(intent="rag_query", direct_response="")


# Retrieves context using chat history and returns an answer with citations.
# Handles retrieval, CRAG evaluation, and answer generation.
# If stream=True: Yields {'type': 'token', 'content': '...'} and {'type': 'sources', 'content': [...]}.
# If stream=False: Returns {'answer': '...', 'context': [...]}.
def answer_question(
    user_query: str,
    chat_history: list,
    db_path: str | Path = DB_PATH,
    selected_file: str | None = None,
    stream: bool = False,
) -> Union[dict, Generator[dict, None, None]]:

    llm = get_llm()

    # ─────────────────────────────────────────────────────────────────────────
    # Dynamic Intent Classification flow
    # ─────────────────────────────────────────────────────────────────────────
    user_intent = _classify_intent(user_query, llm)

    if user_intent.intent == "chitchat":
        reply = (
            user_intent.direct_response
            or "Glad that helped! Let me know if you have any more questions."
        )
        if stream:

            def _chitchat_stream():
                yield {"type": "token", "content": reply}
                yield {"type": "sources", "content": []}

            return _chitchat_stream()
        return {"answer": reply, "context": []}

    # ─────────────────────────────────────────────────────────────────────────
    # Regular flow
    # ─────────────────────────────────────────────────────────────────────────

    db_path = Path(db_path)
    chroma_path = db_path / "chroma"

    if not chroma_path.exists():
        raise FileNotFoundError(
            f"Vector store not found at '{chroma_path}'. Please run ingestion first!"
        )

    # 1. Setup Retriever & Reranker
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

    # 2. History-Aware Standalone Query Reformulation
    standalone_query = user_query
    if chat_history:
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a query reformulator. Given a chat history and the latest user message "
                    "(which may contain conversational corrections like 'no not that' or follow-up details), "
                    "rephrase it into a complete, standalone search query that preserves all primary entities, "
                    "actions, and historical context. Do NOT answer the question. Return ONLY the reformulated query.",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_chain = contextualize_q_prompt | llm
        try:
            res = history_chain.invoke(
                {"input": user_query, "chat_history": chat_history}
            )
            standalone_query = res.content
        except Exception:
            standalone_query = user_query

    # 3. Retrieve Documents
    retrieved_docs = rerank_retriever.invoke(standalone_query)

    # 4. Corrective RAG (CRAG) Check
    if retrieved_docs:
        context_preview = "\n".join([doc.page_content for doc in retrieved_docs])
        parser = JsonOutputParser(pydantic_object=RetrievalGrade)
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a retrieval evaluator. Check if the retrieved documents share the same topic "
                    "or discuss the entities/actions asked about in the question.\n"
                    "Mark 'is_relevant: true' if the context contains relevant or related facts, even if the user "
                    "must infer the final count or answer.\n"
                    "Mark 'is_relevant: false' ONLY if the retrieved documents are completely unrelated.\n"
                    "{format_instructions}",
                ),
                ("human", "Question: {question}\n\nRetrieved Context:\n{context}"),
            ]
        )
        grade_chain = grade_prompt | llm | parser
        try:
            grade_result = grade_chain.invoke(
                {
                    "question": user_query,
                    "context": context_preview,
                    "format_instructions": parser.get_format_instructions(),
                }
            )
            is_relevant = grade_result.get("is_relevant", True)
        except Exception:
            is_relevant = True
    else:
        is_relevant = False

    # Short-circuit if out of scope / irrelevant
    if not is_relevant:
        refusal_msg = "I don't know based on the provided document."
        if stream:

            def _refusal_gen():
                yield {"type": "token", "content": refusal_msg}
                yield {"type": "sources", "content": retrieved_docs}

            return _refusal_gen()
        return {"answer": refusal_msg, "context": retrieved_docs}

    # 5. Build QA Chain
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

    formatted_chunks = [
        f"--- [{doc.metadata.get('source_file', 'Unknown')} | Page {doc.metadata.get('page', 0)}] ---\n{doc.page_content}"
        for doc in retrieved_docs
    ]
    context_str = "\n\n".join(formatted_chunks)
    qa_chain = qa_prompt | llm

    # 6. Return Streaming Generator OR Direct Dictionary
    if stream:

        def _stream_gen():
            for chunk in qa_chain.stream(
                {
                    "context": context_str,
                    "chat_history": chat_history,
                    "input": user_query,
                }
            ):
                token_text = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token_text:
                    yield {"type": "token", "content": token_text}
            yield {"type": "sources", "content": retrieved_docs}

        return _stream_gen()

    # Default synchronous invocation (Evaluation scripts, benchmarks, tests)
    response = qa_chain.invoke(
        {
            "context": context_str,
            "chat_history": chat_history,
            "input": user_query,
        }
    )
    final_text = response.content if hasattr(response, "content") else str(response)
    return {"answer": final_text, "context": retrieved_docs}
