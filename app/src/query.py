# app/src/query.py
from pathlib import Path
from typing import Union, Generator, Tuple
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document

from app.core.config import get_llm, DB_PATH
from app.src.ingestion.ingest import get_indexed_documents
from app.storage.cache import semantic_cache
from app.src.context.history_compressor import compress_chat_history
from app.src.context.grounding import verify_grounding
from app.src.routing.intent_router import (
    classify_user_intent,
    detect_document_filter,
    reformulate_standalone_query,
)
from app.src.retrieval.pipeline import (
    initialize_retrievers,
    execute_multi_query_retrieval,
    evaluate_and_fallback,
)


def _serialize_docs(docs: list[Document]) -> list[dict]:
    return [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]


def _format_context(active_docs: list[Document], is_web: bool) -> Tuple[str, str, str]:
    if is_web:
        prefix = "*(Note: Information was not found in your indexed documents; retrieved via web search fallback)*\n\n"
        sys_prompt = (
            "You are a helpful assistant.\n"
            "{summary_section}"
            "The requested information was not found in the user's uploaded documents, "
            "so you must answer based on the following web search results.\n"
            "Keep the answer concise, accurate, and cite web sources when possible.\n\n"
            "Web search context:\n{context}"
        )
        chunks = [
            f"--- [{d.metadata.get('source_file', 'Web')} | Link: {d.metadata.get('url', '')}] ---\n{d.page_content}"
            for d in active_docs
        ]
    else:
        prefix = ""
        sys_prompt = (
            "You are a helpful assistant for question-answering tasks.\n"
            "{summary_section}"
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
        chunks = [
            f"--- [{d.metadata.get('source_file', 'Unknown')} | Page {d.metadata.get('page', 0)}] ---\n{d.page_content}"
            for d in active_docs
        ]
    return "\n\n".join(chunks), sys_prompt, prefix


def answer_question(
    user_query: str,
    chat_history: list = None,
    db_path: str | Path = DB_PATH,
    selected_file: str | None = None,
    stream: bool = False,
    running_summary: str = "",
) -> Union[dict, Generator[dict, None, None]]:

    llm = get_llm()
    chat_history = chat_history or []

    # 1. Context Window Compression
    effective_history, updated_summary = compress_chat_history(
        chat_history=chat_history,
        llm=llm,
        keep_recent_pairs=2,
        existing_summary=running_summary,
    )

    # 2. Intent Routing
    user_intent = classify_user_intent(user_query, llm)
    if user_intent.intent == "chitchat":
        reply = (
            user_intent.direct_response
            or "Glad that helped! Let me know if you have any questions."
        )
        if stream:

            def _chitchat_stream():
                yield {"type": "token", "content": reply}
                yield {"type": "sources", "content": []}
                yield {
                    "type": "grounding",
                    "content": {"is_grounded": True, "score": 1.0},
                }
                yield {"type": "summary", "content": updated_summary}

            return _chitchat_stream()
        return {
            "answer": reply,
            "context": [],
            "grounding": {"is_grounded": True, "score": 1.0},
            "summary": updated_summary,
        }

    # 3. Query Reformulation & Filter Detection
    standalone_query = reformulate_standalone_query(user_query, effective_history, llm)

    db_path = Path(db_path)
    if not selected_file or selected_file == "All Documents":
        available_files = get_indexed_documents(db_path)
        detected = detect_document_filter(user_query, available_files, llm)
        if detected:
            print(f"🎯 [Self-Querying]: Auto-scoped to '{detected}'.")
            selected_file = detected

    # 4. Semantic Cache
    cache_hit = semantic_cache.lookup(
        query=standalone_query, selected_file=selected_file
    )
    if cache_hit:
        cached_data, sim_score = cache_hit
        print(
            f"⚡ [Semantic Cache HIT]: Reusing response (Score: {float(sim_score):.2f})"
        )
        cached_answer = cached_data.get("answer", "")
        cached_sources = [
            Document(
                page_content=s.get("page_content", ""), metadata=s.get("metadata", {})
            )
            for s in cached_data.get("context", [])
        ]
        cached_grounding = cached_data.get(
            "grounding", {"is_grounded": True, "score": 1.0}
        )

        if stream:

            def _cached_stream():
                words = cached_answer.split(" ")
                for i, w in enumerate(words):
                    yield {
                        "type": "token",
                        "content": w + (" " if i < len(words) - 1 else ""),
                    }
                yield {"type": "sources", "content": cached_sources}
                yield {"type": "grounding", "content": cached_grounding}
                yield {"type": "summary", "content": updated_summary}

            return _cached_stream()
        return {
            "answer": cached_answer,
            "context": cached_sources,
            "grounding": cached_grounding,
            "summary": updated_summary,
        }

    # 5. Retrieval & Reranking Pipeline
    dense_pdr, sparse_pdr, compressor = initialize_retrievers(db_path, selected_file)
    retrieved_docs = execute_multi_query_retrieval(
        standalone_query=standalone_query,
        selected_file=selected_file,
        dense_pdr=dense_pdr,
        sparse_pdr=sparse_pdr,
        compressor=compressor,
        llm=llm,
    )

    # 6. CRAG & Web Fallback
    active_docs, is_web_fallback = evaluate_and_fallback(
        standalone_query, retrieved_docs, llm
    )

    if not active_docs:
        refusal_msg = "I don't know based on the provided document, and web search returned no relevant results."
        if stream:

            def _refusal_gen():
                yield {"type": "token", "content": refusal_msg}
                yield {"type": "sources", "content": []}
                yield {
                    "type": "grounding",
                    "content": {"is_grounded": True, "score": 1.0},
                }
                yield {"type": "summary", "content": updated_summary}

            return _refusal_gen()
        return {
            "answer": refusal_msg,
            "context": [],
            "grounding": {"is_grounded": True, "score": 1.0},
            "summary": updated_summary,
        }

    # 7. Prompt Assembly & QA Chain
    summary_section = (
        f"\n\nContext from earlier in the conversation:\n{updated_summary}\n"
        if updated_summary
        else ""
    )
    context_str, raw_prompt_tpl, prefix_note = _format_context(
        active_docs, is_web_fallback
    )
    system_prompt = raw_prompt_tpl.format(
        summary_section=summary_section, context=context_str
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    qa_chain = qa_prompt | llm

    # 8. Generation & Cache Persistence
    if stream:

        def _stream_gen():
            accumulated = []
            if prefix_note:
                yield {"type": "token", "content": prefix_note}
                accumulated.append(prefix_note)

            for chunk in qa_chain.stream(
                {
                    "chat_history": effective_history,
                    "input": user_query,
                }
            ):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    accumulated.append(token)
                    yield {"type": "token", "content": token}

            raw_answer = "".join(accumulated)

            # Post-generation grounding check (only for document retrieval, not web fallback)
            grounding_meta = {"is_grounded": True, "score": 1.0}
            if not is_web_fallback:
                grounding_report = verify_grounding(raw_answer, active_docs, llm)
                grounding_meta = {
                    "is_grounded": grounding_report["is_grounded"],
                    "score": grounding_report["score"],
                }

            yield {"type": "sources", "content": active_docs}
            yield {"type": "grounding", "content": grounding_meta}
            yield {"type": "summary", "content": updated_summary}

            semantic_cache.store(
                query=standalone_query,
                data={
                    "answer": raw_answer,
                    "context": _serialize_docs(active_docs),
                    "grounding": grounding_meta,
                },
                selected_file=selected_file,
            )

        return _stream_gen()

    response = qa_chain.invoke({"chat_history": effective_history, "input": user_query})
    raw_answer = prefix_note + (
        response.content if hasattr(response, "content") else str(response)
    )

    grounding_meta = {"is_grounded": True, "score": 1.0}
    final_answer = raw_answer

    if not is_web_fallback:
        grounding_report = verify_grounding(raw_answer, active_docs, llm)
        grounding_meta = {
            "is_grounded": grounding_report["is_grounded"],
            "score": grounding_report["score"],
        }
        if grounding_report.get("corrected_answer"):
            final_answer = prefix_note + grounding_report["corrected_answer"]

    semantic_cache.store(
        query=standalone_query,
        data={
            "answer": final_answer,
            "context": _serialize_docs(active_docs),
            "grounding": grounding_meta,
        },
        selected_file=selected_file,
    )
    return {
        "answer": final_answer,
        "context": active_docs,
        "grounding": grounding_meta,
        "summary": updated_summary,
    }
