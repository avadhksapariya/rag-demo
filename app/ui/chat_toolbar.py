import json
import streamlit as st

from datetime import datetime

from app.core.config import DB_PATH
from app.storage.session_store import session_store
from app.src.ingestion.ingest import get_indexed_documents
from app.src.context.export_chat import format_chat_as_markdown


# Formats chat messages into clean readable Markdown.
def generate_markdown_export(messages: list, session_title: str) -> str:
    lines = [
        f"# Chat Export: {session_title}",
        f"*Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "\n---\n",
    ]

    for msg in messages:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        lines.append(f"### 🧑‍💻 {role}" if role == "User" else f"### 🤖 {role}")
        lines.append(content)

        sources = msg.get("sources", [])
        if sources:
            lines.append("\n**Sources Cited:**")
            for s in sources:
                lines.append(f"- `{s.get('file', 'Doc')}` (Page {s.get('page', '?')})")

        lines.append("\n---\n")

    return "\n".join(lines)


# Packages the complete conversation turn-by-turn with grounding scores,
# diagnostics, citations, and metadata ready for Ragas / benchmark evaluation.
def generate_json_export(messages: list, session_title: str, selected_file: str) -> str:
    payload = {
        "export_metadata": {
            "session_title": session_title,
            "exported_at": datetime.now().isoformat(),
            "scope_file": selected_file,
            "total_turns": len(messages),
        },
        "messages": messages,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


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

    col_scope, col_clear, col_export = st.columns(
        [0.38, 0.28, 0.34], vertical_alignment="center"
    )

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
        if not st.session_state.ui_messages:
            # Disabled button when no chat exists
            st.button("📥 Export Chat", disabled=True, use_container_width=True)
        else:
            with st.popover("📥 Export Chat", use_container_width=True):
                st.markdown(
                    "<p style='font-size: 0.85em; color: #8892b0; margin-bottom: 8px;'>Choose export format:</p>",
                    unsafe_allow_html=True,
                )

                current_title = st.session_state.get(
                    "current_session_title", "Chat_Export"
                ).replace(" ", "_")
                selected_file = st.session_state.get("selected_file", "All Documents")

                # 1. Human-Readable Markdown
                md_content = generate_markdown_export(
                    messages=st.session_state.ui_messages,
                    session_title=current_title,
                )
                st.download_button(
                    label="📄 Export as Markdown (.md)",
                    data=md_content,
                    file_name=f"{current_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

                # 2. Evaluation / Machine-Readable JSON
                json_content = generate_json_export(
                    messages=st.session_state.ui_messages,
                    session_title=current_title,
                    selected_file=selected_file,
                )
                st.download_button(
                    label="📊 Export as Evaluation JSON (.json)",
                    data=json_content,
                    file_name=f"{current_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                )

    st.divider()
