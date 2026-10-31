# app/ui/chat_toolbar.py
import streamlit as st
from app.core.config import DB_PATH
from app.storage.session_store import session_store
from app.src.ingestion.ingest import get_indexed_documents
from app.src.context.export_chat import format_chat_as_markdown


def render_chat_toolbar():
    # Resolve current active session title
    raw_sessions = session_store.list_sessions()
    current_title = "New Chat"
    for s in raw_sessions:
        if s["id"] == st.session_state.current_session_id:
            current_title = s["title"]
            break

    st.markdown(
        f"""
        <h4 style="margin: 0; padding-top: 4px; padding-bottom: 12px; font-weight: 600;">
            💬 {current_title}
        </h4>
        """,
        unsafe_allow_html=True,
    )

    col_scope, col_clear, col_export = st.columns([0.38, 0.28, 0.34])

    with col_scope:
        indexed_docs = get_indexed_documents(DB_PATH)
        scope_options = ["All Documents"] + indexed_docs

        if st.session_state.selected_file not in scope_options:
            st.session_state.selected_file = "All Documents"

        current_selected = st.session_state.selected_file
        badge_label = (
            "🎯 All Documents"
            if current_selected == "All Documents"
            else (
                f"🎯 {current_selected[:18]}..."
                if len(current_selected) > 18
                else f"🎯 {current_selected}"
            )
        )

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
        if st.button(
            "🗑️ Clear Chat", use_container_width=True, key="clear_chat_button"
        ):
            st.session_state.chat_history = []
            st.session_state.ui_messages = []
            st.rerun()

    with col_export:
        if st.session_state.ui_messages:
            md_content = format_chat_as_markdown(
                messages=st.session_state.ui_messages,
                session_title=f"Chat-{st.session_state.current_session_id[:8]}",
            )
            st.download_button(
                label="📥 Export Chat",
                data=md_content,
                file_name=f"rag_conversation_{st.session_state.current_session_id[:8]}.md",
                mime="text/markdown",
                use_container_width=True,
                key="export_chat_button",
            )
        else:
            st.button(
                "📥 Export Chat",
                disabled=True,
                use_container_width=True,
                key="export_chat_disabled",
            )

    st.divider()
