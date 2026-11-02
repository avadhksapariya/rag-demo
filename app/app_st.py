import uuid
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from app.core.config import DB_PATH, get_llm
from app.storage.session_store import session_store, generate_chat_title
from app.src.query import answer_question
from app.src.retrieval.highlighter import extract_highlighted_snippet
from app.utils.typing_dots import TYPING_DOTS_LOADER_HTML
from app.utils.copy_btn import render_copy_button

from app.ui.citations import render_citations
from app.ui.sidebar_history import render_conversation_tab
from app.ui.sidebar_ingestion import render_ingestion_tab
from app.ui.chat_toolbar import render_chat_toolbar

# 1. Page Configuration
st.set_page_config(
    page_title="RAG Demo",
    page_icon="📚",
    layout="wide",
)

# 2. Session State Initialization
session_store.cleanup_empty_sessions()

existing_sessions = session_store.list_sessions()

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = (
        existing_sessions[0]["id"] if existing_sessions else str(uuid.uuid4())
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = session_store.get_messages(
        st.session_state.current_session_id
    )
    for msg in st.session_state.ui_messages:
        if msg["role"] == "user":
            st.session_state.chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            st.session_state.chat_history.append(AIMessage(content=msg["content"]))

if "renaming_session_id" not in st.session_state:
    st.session_state.renaming_session_id = None

if "selected_file" not in st.session_state:
    st.session_state.selected_file = "All Documents"

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "last_ingested_message" not in st.session_state:
    st.session_state.last_ingested_message = None

if "show_success_counter" not in st.session_state:
    st.session_state.show_success_counter = 0

if "retry_query" not in st.session_state:
    st.session_state.retry_query = None

# 3. Sidebar
with st.sidebar:
    st.subheader("📚 Retrieval-Augmented Generation")
    tab_chats, tab_docs = st.tabs(["💬 Conversations", "📄 Ingestion & Tools"])
    with tab_chats:
        render_conversation_tab()
    with tab_docs:
        render_ingestion_tab()

# 4. Main Chat Area
left_pad, main_chat_col, right_pad = st.columns([1, 6, 1])

with main_chat_col:
    render_chat_toolbar()

    # 5. Display Existing Chat Messages
    for msg_idx, msg in enumerate(st.session_state.ui_messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg.get("grounding"):
                score = msg["grounding"].get("score", 1.0)
                caption = (
                    f"🛡️ Grounding Confidence: **{int(score * 100)}%** (Factually verified)"
                    if score >= 0.8
                    else f"⚠️ Grounding Confidence: **{int(score * 100)}%** (Some claims could not be verified)"
                )
                st.caption(caption)

            if msg.get("diagnostics"):
                diag = msg["diagnostics"]
                with st.expander("⚡ Execution Diagnostics", expanded=False):
                    d_col1, d_col2, d_col3 = st.columns(3)
                    d_col1.metric("Total Latency", f"{diag.get('total_time_sec', 0)}s")
                    d_col2.metric(
                        "Retrieval Duration", f"{diag.get('retrieval_time_sec', 0)}s"
                    )
                    d_col3.metric("Chunks Evaluated", diag.get("chunks_retrieved", 0))

                    if diag.get("is_web_fallback"):
                        st.info("🌐 Mode: Corrective Web Search Fallback Triggered")
                    elif diag.get("cached"):
                        st.success("⚡ Mode: Semantic Cache Hit")
                    else:
                        st.success("📚 Mode: Local Parent-Document Hybrid Search")

            if msg.get("sources"):
                render_citations(msg["sources"], unique_prefix=f"hist_{msg_idx}")

            is_latest_assistant = (
                msg["role"] == "assistant"
                and msg_idx == len(st.session_state.ui_messages) - 1
            )

            # --- ACTION BAR: Retry & Copy ---
            is_latest_assistant = (
                msg["role"] == "assistant"
                and msg_idx == len(st.session_state.ui_messages) - 1
            )

            if msg["role"] == "assistant":
                st.markdown(
                    """
                    <style>
                    div[data-testid="stHorizontalBlock"]:has(iframe) {
                        align-items: center !important;
                    }
                    div[data-testid="stHorizontalBlock"]:has(iframe) iframe {
                        margin-top: 2px !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                col_copy, col_retry, col_spacer = st.columns(
                    [0.12, 0.12, 0.76], vertical_alignment="center"
                )

                with col_copy:
                    render_copy_button(
                        text_to_copy=msg["content"], key_suffix=f"msg_{msg_idx}"
                    )

                with col_retry:
                    if is_latest_assistant:
                        if st.button(
                            "🔄 Retry",
                            key=f"retry_{msg_idx}",
                            type="tertiary",
                            help="Regenerate this answer",
                        ):
                            st.session_state.ui_messages.pop()
                            last_user = st.session_state.ui_messages.pop()
                            st.session_state.chat_history.pop()
                            st.session_state.chat_history.pop()
                            st.session_state.retry_query = last_user["content"]
                            st.rerun()

# Chat Input Styling
st.markdown(
    """
    <style>
    div[data-testid="stChatInput"] {
        width: 75%;
        max-width: none;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 6. Live Chat Input Handling
user_input = st.chat_input("Ask a question about your uploaded document...")
active_query = user_input or st.session_state.get("retry_query")

if active_query:
    user_query = active_query
    st.session_state.retry_query = None

    current_sid = st.session_state.current_session_id

    if not st.session_state.ui_messages:
        auto_title = generate_chat_title(user_query, get_llm())
        session_store.create_session(current_sid, title=auto_title)

    with main_chat_col:
        st.chat_message("user").markdown(user_query)
        st.session_state.ui_messages.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        session_store.save_messages(current_sid, role="user", content=user_query)

        with st.chat_message("assistant"):
            try:
                placeholder = st.empty()
                placeholder.markdown(TYPING_DOTS_LOADER_HTML, unsafe_allow_html=True)

                stream_gen = answer_question(
                    user_query=user_query,
                    chat_history=st.session_state.chat_history,
                    db_path=DB_PATH,
                    selected_file=st.session_state.selected_file,
                    stream=True,
                )

                retrieved_sources = []
                grounding_info = {}
                diagnostics_info = {}

                def token_generator():
                    cleared_loader = False
                    for chunk in stream_gen:
                        chunk_type = chunk.get("type")
                        if chunk_type == "token":
                            if not cleared_loader:
                                placeholder.empty()
                                cleared_loader = True
                            yield chunk["content"]
                        elif chunk_type == "sources":
                            retrieved_sources.extend(chunk.get("content", []))
                        elif chunk_type == "grounding":
                            grounding_info.update(chunk.get("content", {}))
                        elif chunk_type == "diagnostics":
                            diagnostics_info.update(chunk.get("content", {}))
                        elif chunk_type == "summary":
                            st.session_state.running_summary = chunk.get("content", "")

                answer_text = st.write_stream(token_generator())

                sources_data = []
                for doc in retrieved_sources:
                    page = doc.metadata.get("page", 0) + 1
                    file_name = doc.metadata.get("source_file", "Document")
                    snippet_text = extract_highlighted_snippet(
                        doc_text=doc.page_content,
                        query=user_query,
                        max_sentences=2,
                    )
                    sources_data.append(
                        {
                            "file": file_name,
                            "page": page,
                            "text": snippet_text,
                        }
                    )

                # Persist turn
                st.session_state.ui_messages.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources_data,
                        "grounding": grounding_info,
                        "diagnostics": diagnostics_info,
                    }
                )
                st.session_state.chat_history.append(AIMessage(content=answer_text))
                session_store.save_messages(
                    current_sid, role="assistant", content=answer_text
                )

                # Rerun so Section 5 mounts completed turn with Retry button & expanders
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error generating response: {e}")
