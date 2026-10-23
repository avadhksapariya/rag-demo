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
from langchain_core.documents import Document
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

from app.src.ingest import get_indexed_documents
from app.src.embeddings import get_embeddings
from app.core.config import (
    get_llm,
    DB_PATH,
    INITIAL_RETRIEVAL_K,
    FINAL_RETRIEVAL_K,
)
from app.src.web_search import search_web_fallback
from app.src.hybrid import ParentBM25Retriever, reciprocal_rank_fusion
from app.src.history_compressor import compress_chat_history
from app.src.cache import semantic_cache

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


class DocumentFilterDecision(BaseModel):
    target_file: str | None = Field(
        default=None,
        description="The exact filename from the candidate list that the user specifically mentions or refers to. If the user does not specify a particular file, return None.",
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


# Detects if the user query references a specific document name or theme
# and maps it to an indexed file in ChromaDB.
def detect_document_filter(query: str, available_files: list[str], llm) -> str | None:
    if not available_files:
        return None

    if len(available_files) == 1:
        return available_files[0]

    parser = JsonOutputParser(pydantic_object=DocumentFilterDecision)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a document routing assistant.\n"
                "Analyze the user question and determine if they are asking about or referencing a specific document from this list:\n"
                "{file_list}\n\n"
                "RULES:\n"
                "1. If the user mentions a document title, topic, or file name matching an item in the list, set 'target_file' to that exact file name.\n"
                "2. If the user does not specify or asks across all knowledge, set 'target_file' to null.\n"
                "{format_instructions}",
            ),
            ("human", "{question}"),
        ]
    )

    chain = prompt | llm | parser
    try:
        res = chain.invoke(
            {
                "question": query,
                "file_list": "\n".join([f"- {f}" for f in available_files]),
                "format_instructions": parser.get_format_instructions(),
            }
        )
        return res.get("target_file")
    except Exception:
        return None


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
    running_summary: str = "",
) -> Union[dict, Generator[dict, None, None]]:

    llm = get_llm()

    if chat_history is None:
        chat_history = []

    # ─────────────────────────────────────────────────────────────────────────
    # 0. Context Window Compression
    # ─────────────────────────────────────────────────────────────────────────
    # Retain the last 2 interactions (4 messages) verbatim; compress the rest
    effective_history, updated_summary = compress_chat_history(
        chat_history=chat_history,
        llm=llm,
        keep_recent_pairs=2,
        existing_summary=running_summary,
    )

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
                yield {"type": "summary", "content": updated_summary}

            return _chitchat_stream()
        return {"answer": reply, "context": [], "summary": updated_summary}

    # ─────────────────────────────────────────────────────────────────────────
    # Regular flow
    # ─────────────────────────────────────────────────────────────────────────

    db_path = Path(db_path)
    chroma_path = db_path / "chroma"

    if not chroma_path.exists():
        raise FileNotFoundError(
            f"Vector store not found at '{chroma_path}'. Please run ingestion first!"
        )

    # 1. History-Aware Standalone Query Reformulation
    standalone_query = user_query
    if effective_history:
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a search query optimizer. Given a conversation history and a follow-up user message, "
                    "rephrase it into a concise, standalone keyword search query (3 to 8 words maximum).\n"
                    "CRITICAL RULES:\n"
                    "- DO NOT answer the question.\n"
                    "- DO NOT explain or add commentary.\n"
                    "- DO NOT use markdown bolding (**), quotes, or punctuation.\n"
                    "- Output ONLY the search query string.",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_chain = contextualize_q_prompt | llm
        try:
            res = history_chain.invoke(
                {"input": user_query, "chat_history": effective_history}
            )
            standalone_query = res.content.strip()
        except Exception:
            standalone_query = user_query

    # 2. Automated Metadata Filter Detection (Self-Querying)
    # If the user left scope as "All Documents" (or None), detect if the query targets a specific file
    if not selected_file or selected_file == "All Documents":
        available_files = get_indexed_documents(db_path)
        detected_file = detect_document_filter(user_query, available_files, llm)
        if detected_file:
            print(f"🎯 [Self-Querying]: Auto-scoped retrieval to '{detected_file}'.")
            selected_file = detected_file

    # 2.5 Semantic Cache Lookup
    cache_result = semantic_cache.lookup(
        query=standalone_query, selected_file=selected_file
    )

    if cache_result:
        cached_data, sim_score = cache_result
        try:
            score_val = float(sim_score)
            print(
                f"⚡ [Semantic Cache HIT]: Reusing response (Similarity: {score_val:.2f})"
            )
        except (ValueError, TypeError):
            print(f"⚡ [Semantic Cache HIT]: Reusing response (Score: {sim_score})")

        cached_answer = cached_data.get("answer", "")
        cached_sources = [
            Document(
                page_content=s.get("page_content", ""), metadata=s.get("metadata", {})
            )
            for s in cached_data.get("context", [])
        ]

        if stream:

            def _cached_stream():
                words = cached_answer.split(" ")
                for i, word in enumerate(words):
                    yield {
                        "type": "token",
                        "content": word + (" " if i < len(words) - 1 else ""),
                    }
                yield {"type": "sources", "content": cached_sources}
                yield {"type": "summary", "content": updated_summary}

            return _cached_stream()

        return {
            "answer": cached_answer,
            "context": cached_sources,
            "summary": updated_summary,
        }

    # 3. Setup Retriever & Reranker
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
    dense_pdr = ParentDocumentRetriever(
        vectorstore=vector_db,
        docstore=store,
        child_splitter=child_splitter,
        search_kwargs=search_kwargs,
    )

    # Sparse BM25 PDR
    sparse_pdr = ParentBM25Retriever(
        vector_db=vector_db,
        store=store,
        k=INITIAL_RETRIEVAL_K,
    )

    # Standalone Cross-Encoder Reranker
    compressor = FlashrankRerank(top_n=FINAL_RETRIEVAL_K)

    # 4. Retrieve Documents via Hybrid Search (Dense + Sparse) & Rerank
    dense_docs = dense_pdr.invoke(standalone_query)
    sparse_docs = sparse_pdr.invoke(standalone_query, selected_file=selected_file)

    hybrid_candidates = reciprocal_rank_fusion(
        dense_docs=dense_docs, sparse_docs=sparse_docs, top_n=INITIAL_RETRIEVAL_K
    )

    # Compress / Rerank top candidates using FlashRank
    if hybrid_candidates:
        retrieved_docs = compressor.compress_documents(
            documents=hybrid_candidates, query=standalone_query
        )
    else:
        retrieved_docs = []

    # 4. Corrective RAG (CRAG) Check: Evaluate relevance of retrieved context
    if retrieved_docs:
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
    else:
        is_relevant = False

    # 5. Corrective Action: Fallback to Web Search if Documents are Irrelevant
    is_web_fallback = False
    active_docs = retrieved_docs

    if not is_relevant:
        clean_search_query = (
            standalone_query.replace("**", "").replace('"', "").strip().split("\n")[0]
        )

        web_docs = search_web_fallback(clean_search_query, max_results=3)
        if web_docs:
            active_docs = web_docs
            is_web_fallback = True
        else:
            refusal_msg = "I don't know based on the provided document, and web search returned no relevant results"
            if stream:

                def _refusal_gen():
                    yield {"type": "token", "content": refusal_msg}
                    yield {"type": "sources", "content": []}
                    yield {"type": "summary", "content": updated_summary}

                return _refusal_gen()
            return {"answer": refusal_msg, "context": [], "summary": updated_summary}

    # 6. Build QA Chain: Choose prompt based on Document vs Web search
    summary_section = (
        f"\n\nContext from earlier in the conversation:\n{updated_summary}\n"
        if updated_summary
        else ""
    )

    if is_web_fallback:
        prefix_note = "*(Note: Information was not found in your indexed documents; retrieved via web search fallback)*\n\n"
        system_prompt = (
            "You are a helpful assistant.\n"
            "The requested information was not found in the user's uploaded documents, "
            "so you must answer based on the following web search results.\n"
            "Keep the answer concise, accurate, and cite web sources when possible.\n\n"
            "Web search context:\n{context}"
        )
        formatted_chunks = [
            f"--- [{doc.metadata.get('source_file', 'Web')} | Link: {doc.metadata.get('url', '')}] ---\n{doc.page_content}"
            for doc in active_docs
        ]
    else:
        prefix_note = ""
        system_prompt = (
            "You are a helpful assistant for question-answering tasks.\n"
            "Use ONLY the following pieces of retrieved context to answer the question.\n"
            "Do NOT infer or extrapolate facts not directly stated in the text.\n"
            "If the context does not explicitly state the answer, say:\n"
            '"I don\'t know based on the provided document."\n'
            "Do not use outside knowledge.\n"
            "Keep the answer concise and clear.\n\n"
            "CITATION INSTRUCTIONS:\n"
            "1. Include inline citations like [FileName | Page X] for facts stated in your answer.\n"
            "2. Use the exact source name and page number shown in the header markers.\n\n"
            "Retrieved context:\n{context}"
        )
        # "you don't know. Do not hallucinate.\n\n"
        formatted_chunks = [
            f"--- [{doc.metadata.get('source_file', 'Unknown')} | Page {doc.metadata.get('page', 0)}] ---\n{doc.page_content}"
            for doc in active_docs
        ]

    context_str = "\n\n".join(formatted_chunks)

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    qa_chain = qa_prompt | llm

    # 7. Return Streaming Generator OR Direct Dictionary

    # Helper to serialize Document objects so context can be stored in JSON cache
    def _serialize_docs(docs):
        return [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]

    if stream:

        def _stream_gen():
            accumulated_answer = []

            if prefix_note:
                yield {"type": "token", "content": prefix_note}
                accumulated_answer.append(prefix_note)

            for chunk in qa_chain.stream(
                {
                    "context": context_str,
                    "chat_history": effective_history,
                    "input": user_query,
                }
            ):
                token_text = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token_text:
                    accumulated_answer.append(token_text)
                    yield {"type": "token", "content": token_text}

            yield {"type": "sources", "content": active_docs}
            yield {"type": "summary", "content": updated_summary}

            full_text = "".join(accumulated_answer)
            semantic_cache.store(
                query=standalone_query,
                data={
                    "answer": full_text,
                    "context": _serialize_docs(active_docs),
                },
                selected_file=selected_file,
            )

        return _stream_gen()

    # Default synchronous invocation (Evaluation scripts, benchmarks, tests)
    response = qa_chain.invoke(
        {
            "context": context_str,
            "chat_history": effective_history,
            "input": user_query,
        }
    )
    final_text = prefix_note + (
        response.content if hasattr(response, "content") else str(response)
    )

    # Store newly generated response into Semantic Cache
    semantic_cache.store(
        query=standalone_query,
        data={
            "answer": final_text,
            "context": _serialize_docs(active_docs),
        },
        selected_file=selected_file,
    )

    return {
        "answer": final_text,
        "context": active_docs,
        "summary": updated_summary,
    }
