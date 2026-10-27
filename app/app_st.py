import os
import uuid
import tempfile
import streamlit as st

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from app.src.ingestion.ingest import process_pdf_and_ingest, get_indexed_documents
from app.src.query import answer_question
from app.src.retrieval.highlighter import extract_highlighted_snippet
from app.core.config import DB_PATH, get_llm
from app.storage.cache import semantic_cache
from app.storage.session_store import session_store, generate_chat_title
from app.utils.typing_dots import TYPING_DOTS_LOADER_HTML

# 1. Page Configuration
st.set_page_config(
    page_title="RAG Demo",
    page_icon="📚",
    layout="wide",
)

# 2. Initialize Session State Variables
session_store.cleanup_empty_sessions()

existing_sessions = session_store.list_sessions()

if "current_session_id" not in st.session_state:
    if existing_sessions:
        st.session_state.current_session_id = existing_sessions[0]["id"]
    else:
        st.session_state.current_session_id = str(uuid.uuid4())

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

# Scope selection state (defaults to "All Documents")
if "selected_file" not in st.session_state:
    st.session_state.selected_file = "All Documents"


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


def start_new_session():
    if not st.session_state.ui_messages:
        return
    st.session_state.current_session_id = str(uuid.uuid4())
    st.session_state.ui_messages = []
    st.session_state.chat_history = []
    # Reset scope back to default on new chat if desired
    st.session_state.selected_file = "All Documents"


# 3. Sidebar
with st.sidebar:
    st.subheader("📚 Retrieval-Augmented Generation")
    tab_chats, tab_docs = st.tabs(["💬 Conversations", "📄 Ingestion & Tools"])

    # --- TAB 1: Conversations History ---
    with tab_chats:
        if st.button("＋ New Chat", use_container_width=True, type="primary"):
            start_new_session()
            st.rerun()

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

        with st.container(height=320, border=False):
            for s in all_sessions:
                s_id = s["id"]
                s_title = s["title"]
                is_active = s_id == st.session_state.current_session_id

                # Use an isolated row container to prevent widget re-layout flashes
                row = st.container()

                if st.session_state.renaming_session_id == s_id:
                    col_input, col_save, col_cancel = row.columns([0.76, 0.12, 0.12])

                    with col_input:
                        new_title_val = st.text_input(
                            "Rename",
                            value=s_title,
                            key=f"rename_input_{s_id}",
                            label_visibility="collapsed",
                        )

                    with col_save:
                        if st.button(
                            "✓", key=f"save_{s_id}", help="Save title", type="tertiary"
                        ):
                            if new_title_val.strip():
                                session_store.update_session_title(
                                    s_id, new_title_val.strip()
                                )
                            st.session_state.renaming_session_id = None
                            st.rerun()

                    with col_cancel:
                        if st.button(
                            "✕",
                            key=f"cancel_{s_id}",
                            help="Cancel rename",
                            type="tertiary",
                        ):
                            st.session_state.renaming_session_id = None
                            st.rerun()

                else:
                    col_select, col_edit, col_del = row.columns([0.76, 0.12, 0.12])

                    with col_select:
                        label = f"● {s_title}" if is_active else s_title
                        if st.button(
                            label,
                            key=f"sess_{s_id}",
                            use_container_width=True,
                            disabled=is_active,
                            type="secondary" if is_active else "tertiary",
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
                                    x
                                    for x in session_store.list_sessions()
                                    if x["id"] != s_id
                                ]
                                if remaining:
                                    switch_session(remaining[0]["id"])
                                else:
                                    st.session_state.current_session_id = str(
                                        uuid.uuid4()
                                    )
                                    st.session_state.ui_messages = []
                                    st.session_state.chat_history = []
                            st.rerun()

    # --- TAB 2: Document Ingestion & Utilities ---
    with tab_docs:
        st.markdown("#### 📄 Document Ingestion")
        uploaded_files = st.file_uploader(
            "Upload one or more PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            if st.button(
                "Process & Ingest PDF", use_container_width=True, type="primary"
            ):
                with st.spinner("Processing PDF(s) and updating vector store..."):
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".pdf"
                        ) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file.flush()
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

        st.divider()
        st.markdown("#### ⚡ Cache Operations")
        if st.button("Flush Cache", use_container_width=True):
            semantic_cache.clear()
            st.toast("🧹 Semantic cache cleared successfully!")


# 4. Main Chat Area: Centered layout
left_pad, main_chat_col, right_pad = st.columns([1, 6, 1])

with main_chat_col:
    # Resolve current active session title
    current_title = "New Chat"
    for s in raw_sessions:
        if s["id"] == st.session_state.current_session_id:
            current_title = s["title"]
            break

    # Top Toolbar: Title, In-Place Scope Selector Popover, Clear Button
    col_header, col_scope_popover, col_clear = st.columns([0.48, 0.34, 0.18])

    with col_header:
        st.markdown(
            f"<h4 style='margin: 0; padding-top: 4px; font-weight: 600;'>💬 {current_title}</h4>",
            unsafe_allow_html=True,
        )

    with col_scope_popover:
        # Load available indexed documents
        indexed_docs = get_indexed_documents(DB_PATH)
        scope_options = ["All Documents"] + indexed_docs

        # Ensure current selection exists in available options
        if st.session_state.selected_file not in scope_options:
            st.session_state.selected_file = "All Documents"

        current_selected = st.session_state.selected_file

        # Display label: compact with truncation if file name is long
        badge_label = (
            "🎯 All Documents"
            if current_selected == "All Documents"
            else (
                f"🎯 {current_selected[:18]}..."
                if len(current_selected) > 18
                else f"🎯 {current_selected}"
            )
        )

        # Clickable Scope Popover directly inside the chat top bar
        with st.popover(
            badge_label, use_container_width=True, help="Change document filter scope"
        ):
            st.caption("Select Context Scope")
            chosen_scope = st.selectbox(
                "Filter retrieval context:",
                options=scope_options,
                index=scope_options.index(st.session_state.selected_file),
                label_visibility="collapsed",
                key="scope_selector_widget",
            )
            if chosen_scope != st.session_state.selected_file:
                st.session_state.selected_file = chosen_scope
                st.rerun()

    with col_clear:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.ui_messages = []
            st.rerun()

    st.divider()

    # 5. Display Existing Chat Messages
    for msg in st.session_state.ui_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

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
                            f" - Page {src['page']}"
                            if src.get("page") is not None
                            else ""
                        )
                        st.markdown(
                            f"**[{idx+1}] {src.get('file', 'Document')}{page_info}**"
                        )
                        st.markdown(
                            f"<div style='font-size: 0.9em; line-height: 1.5; color: #636e72;'>{src['text']}</div>",
                            unsafe_allow_html=True,
                        )

    # 6. Handle User Chat Input
    if user_query := st.chat_input("Ask a question about your uploaded document..."):
        current_sid = st.session_state.current_session_id

        if not st.session_state.ui_messages:
            auto_title = generate_chat_title(user_query, get_llm())
            session_store.create_session(current_sid, title=auto_title)

        st.chat_message("user").markdown(user_query)
        st.session_state.ui_messages.append({"role": "user", "content": user_query})
        session_store.save_messages(current_sid, role="user", content=user_query)

        with st.chat_message("assistant"):
            try:
                placeholder = st.empty()
                placeholder.markdown(TYPING_DOTS_LOADER_HTML, unsafe_allow_html=True)

                # Pass the selected scope directly from session state
                stream_gen = answer_question(
                    user_query=user_query,
                    chat_history=st.session_state.chat_history,
                    db_path=DB_PATH,
                    selected_file=st.session_state.selected_file,
                    stream=True,
                )

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
                            retrieved_sources.extend(chunk.get("content", []))
                        elif chunk_type == "grounding":
                            grounding_info.update(chunk.get("content", {}))
                        elif chunk_type == "summary":
                            st.session_state.running_summary = chunk.get("content", "")

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

                if sources_data:
                    with st.expander("📌 Retrieved Sources & Citations"):
                        for idx, src in enumerate(sources_data):
                            page_info = (
                                f" - Page {src['page']}"
                                if src.get("page") is not None
                                else ""
                            )
                            st.markdown(f"**[{idx+1}] {src['file']}{page_info}**")
                            st.markdown(
                                f"<div style='font-size: 0.9em; line-height: 1.5; color: #636e72;'>{src['text']}</div>",
                                unsafe_allow_html=True,
                            )

                st.session_state.ui_messages.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources_data,
                        "grounding": grounding_info,
                    }
                )

                st.session_state.chat_history.append(HumanMessage(content=user_query))
                st.session_state.chat_history.append(AIMessage(content=answer_text))

                session_store.save_messages(
                    session_id=current_sid, role="assistant", content=answer_text
                )

            except Exception as e:
                st.error(f"❌ Error generating response: {e}")
