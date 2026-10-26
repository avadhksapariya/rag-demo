import os
import uuid
import tempfile
import streamlit as st

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from app.src.ingestion.ingest import process_pdf_and_ingest, get_indexed_documents
from app.src.query import answer_question
from app.core.config import DB_PATH, get_llm
from app.storage.cache import semantic_cache
from app.storage.session_store import session_store, generate_chat_title
from app.utils.typing_dots import TYPING_DOTS_LOADER_HTML

# 1. Page Configuration
st.set_page_config(
    page_title="RAG Demo",
    page_icon="📚",
    layout="centered",
)

st.subheader("📚 RAG (Retrieval-Augmented Generation)")

# 2. Initialize Session State Variables
session_store.cleanup_empty_sessions()

existing_sessions = session_store.list_sessions()

if "current_session_id" not in st.session_state:
    if existing_sessions:
        # Re-attach to the most recent conversation
        st.session_state.current_session_id = existing_sessions[0]["id"]
    else:
        # Only assign a temporary in-memory ID without saving to SQLite yet
        new_id = str(uuid.uuid4())
        st.session_state.current_session_id = new_id

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Stores LangChain Human/AI message objects

if "ui_messages" not in st.session_state:
    # Load existing messages from DB on initial boot
    st.session_state.ui_messages = session_store.get_messages(
        st.session_state.current_session_id
    )

    # Reconstruct LangChain objects for conversation context
    for msg in st.session_state.ui_messages:
        if msg["role"] == "user":
            st.session_state.chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            st.session_state.chat_history.append(AIMessage(content=msg["content"]))

if "renaming_session_id" not in st.session_state:
    st.session_state.renaming_session_id = None


# Loads a selected session from SQLite into active memory.
def switch_session(target_session_id: str):
    st.session_state.current_session_id = target_session_id
    loaded_msgs = session_store.get_messages(target_session_id)
    st.session_state.ui_messages = loaded_msgs
    st.session_state.chat_history = []
    for m in loaded_msgs:
        if m["role"] == "user":
            st.session_state.chat_history.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            st.session_state.chat_history.append(AIMessage(content=m["content"]))


# Starts a fresh session in-memory without polluting the database.
def start_new_session():
    # If the current session has no messages, remain on it rather than creating a duplicate
    if not st.session_state.ui_messages:
        return
    new_id = str(uuid.uuid4())
    st.session_state.current_session_id = new_id
    st.session_state.ui_messages = []
    st.session_state.chat_history = []


# 3. Sidebar: Multiple PDF Upload & File Selector
with st.sidebar:
    st.header("💬 Conversations")
    if st.button("+ New Chat", use_container_width=True, type="primary"):
        start_new_session()
        st.rerun()

    all_sessions = session_store.list_sessions()

    # If the user is currently on an unsaved/blank chat, show it as active in the list
    is_unsaved_current = not any(
        s["id"] == st.session_state.current_session_id for s in all_sessions
    )
    if is_unsaved_current:
        all_sessions.insert(
            0, {"id": st.session_state.current_session_id, "title": "New Chat"}
        )

    raw_sessions = session_store.list_sessions()

    active_session_obj = next(
        (s for s in raw_sessions if s["id"] == st.session_state.current_session_id),
        None,
    )

    other_sessions = [
        s for s in raw_sessions if s["id"] != st.session_state.current_session_id
    ]

    if active_session_obj:
        all_sessions = [active_session_obj] + other_sessions
    else:
        all_sessions = [
            {"id": st.session_state.current_session_id, "title": "New Chat"}
        ] + other_sessions

    for s in all_sessions:
        s_id = s["id"]
        s_title = s["title"]
        is_active = s_id == st.session_state.current_session_id

        if st.session_state.renaming_session_id == s_id:
            col_input, col_save = st.columns([0.75, 0.25])
            with col_input:
                new_title_val = st.text_input(
                    "Rename",
                    value=s_title,
                    key=f"rename_input_{s_id}",
                    label_visibility="collapsed",
                )
            with col_save:
                if st.button("✓", key=f"save_{s_id}", help="Save title"):
                    if new_title_val.strip():
                        session_store.update_session_title(s_id, new_title_val.strip())
                        st.session_state.renaming_session_id = None
                        st.rerun()
        else:
            col_select, col_edit, col_del = st.columns([0.8, 0.1, 0.1])
            with col_select:
                label = f"● {s_title}" if is_active else s_title
                if st.button(
                    label,
                    key=f"sess_{s_id}",
                    use_container_width=True,
                    disabled=is_active,
                    type="tertiary",
                ):
                    switch_session(s_id)
                    st.rerun()

            with col_edit:
                if st.button(
                    "✏️",
                    key=f"edit_{s_id}",
                    help="Rename Chat",
                    type="tertiary",
                ):
                    st.session_state.renaming_session_id = s_id
                    st.rerun()

            with col_del:
                if st.button(
                    "🗑️",
                    key=f"del_{s_id}",
                    help="Delete this session",
                    type="tertiary",
                ):
                    session_store.delete_session(s_id)
                    if s_id == st.session_state.current_session_id:
                        remaining = [
                            x for x in session_store.list_sessions() if x["id"] != s_id
                        ]
                        if remaining:
                            switch_session(remaining[0]["id"])
                        else:
                            st.session_state.current_session_id = str(uuid.uuid4())
                            st.session_state.ui_messages = []
                            st.session_state.chat_history = []
                    st.rerun()

    st.divider()

    st.header("📄 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        max_upload_size=1,  # upto 1 MB
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("Process & Ingest PDF"):
            with st.spinner("Processing PDF(s) and updating vector store..."):
                for uploaded_file in uploaded_files:
                    # Save uploaded file to a temp directory
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file.flush()  # Forces Python to flush data to disk
                        temp_pdf_path = tmp_file.name

                    try:
                        process_pdf_and_ingest(
                            pdf_path=temp_pdf_path,
                            db_path=DB_PATH,
                            filename_override=uploaded_file.name,
                        )
                    except Exception as e:
                        st.error(f"❌ Error during ingestion: {e}")
                    finally:
                        if os.path.exists(temp_pdf_path):
                            os.remove(temp_pdf_path)

                st.success("✅ Ingestion complete! Vector store updated.")
                st.rerun()

    st.header("🔍 Document Filter")
    indexed_docs = get_indexed_documents(DB_PATH)

    options = ["All Documents"] + indexed_docs
    selected_file = st.selectbox(
        "Choose context scope:",
        options=options,
        index=0,
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.ui_messages = []
            st.rerun()

    with col2:
        if st.button("Flush Cache", use_container_width=True):
            semantic_cache.clear()
            st.toast("🧹 Semantic cache cleared successfully!")

# 4. Display Existing Chat Messages
for msg in st.session_state.ui_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display grounding trust badge if available
        if msg.get("grounding"):
            score = msg["grounding"].get("score", 1.0)
            if score >= 0.8:
                st.caption(
                    f"🛡️ Grounding Confidence: **{int(score * 100)}%** (Factually verified)"
                )
            else:
                st.caption(
                    f"⚠️ Grounding Confidence: **{int(score * 100)}%** (Some claims could not be verified)"
                )

        if "sources" in msg and msg["sources"]:
            with st.expander("📌 Retrieved Sources & Citations"):
                for idx, src in enumerate(msg["sources"]):
                    page_info = (
                        f" - Page {src['page']}" if src.get("page") is not None else ""
                    )
                    st.markdown(
                        f"**[{idx+1}] {src.get('file', 'Document')}{page_info}**"
                    )
                    if src.get("url"):
                        st.markdown(f"🔗 [Source Link]({src['url']})")
                    st.caption(src["text"])

# 5. Handle User Chat Input
if user_query := st.chat_input("Ask a question about your uploaded document..."):
    current_sid = st.session_state.current_session_id

    # If this is the first message in this thread, save the session to DB
    if not st.session_state.ui_messages:
        auto_title = generate_chat_title(user_query, get_llm())
        session_store.create_session(current_sid, title=auto_title)

    # Render and persist user question
    st.chat_message("user").markdown(user_query)
    st.session_state.ui_messages.append({"role": "user", "content": user_query})
    session_store.save_messages(current_sid, role="user", content=user_query)

    with st.chat_message("assistant"):
        try:
            placeholder = st.empty()
            placeholder.markdown(TYPING_DOTS_LOADER_HTML, unsafe_allow_html=True)

            stream_gen = answer_question(
                user_query=user_query,
                chat_history=st.session_state.chat_history,
                db_path=DB_PATH,
                selected_file=selected_file,
                stream=True,
            )

            # Container list for sources captured during the stream
            retrieved_sources = []
            grounding_info = {}

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
                        # In-place mutation: works across all scopes without nonlocal/global
                        retrieved_sources.extend(chunk.get("content", []))
                    elif chunk_type == "grounding":
                        grounding_info.update(chunk.get("content", {}))
                    elif chunk_type == "summary":
                        # Store updated running summary for next turns
                        st.session_state.running_summary = chunk.get("content", "")

            # Native Streamlit typewriter streaming
            answer_text = st.write_stream(token_generator())

            if grounding_info:
                score = grounding_info.get("score", 1.0)
                if score >= 0.8:
                    st.caption(
                        f"🛡️ Grounding Confidence: **{int(score * 100)}%** (Factually verified)"
                    )
                else:
                    st.caption(
                        f"⚠️ Grounding Confidence: **{int(score * 100)}%** (Some claims could not be verified)"
                    )

            # Format retrieved sources
            sources_data = []
            for doc in retrieved_sources:
                page = doc.metadata.get("page", 0) + 1
                file_name = doc.metadata.get("source_file", "Document")
                sources_data.append(
                    {
                        "file": file_name,
                        "page": page,
                        "text": doc.page_content[:200] + "...",
                    }
                )

            if sources_data:
                with st.expander("📌 Retrieved Sources & Citations"):
                    for idx, src in enumerate(sources_data):
                        st.markdown(f"**[{idx+1}] {src['file']} - Page {src['page']}**")
                        st.caption(src["text"])

            st.session_state.ui_messages.append(
                {
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources_data,
                }
            )

            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=answer_text))

            session_store.save_messages(
                session_id=current_sid, role="assistant", content=answer_text
            )

        except Exception as e:
            st.error(f"❌ Error generating response: {e}")
