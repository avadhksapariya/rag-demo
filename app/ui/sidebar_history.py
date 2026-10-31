import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from app.storage.session_store import session_store


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


def render_conversation_tab():
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

    all_sessions = (
        [active_session_obj] + other_sessions
        if active_session_obj
        else [{"id": st.session_state.current_session_id, "title": "New Chat"}]
        + other_sessions
    )

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
                    if st.button("✓", key=f"save_{s_id}", help="Save", type="tertiary"):
                        if new_title_val.strip():
                            session_store.update_session_title(
                                s_id, new_title_val.strip()
                            )
                        st.session_state.renaming_session_id = None
                        st.rerun()

                with col_cancel:
                    if st.button(
                        "✕", key=f"cancel_{s_id}", help="Cancel", type="tertiary"
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
                        "✏️", key=f"edit_{s_id}", help="Rename Chat", type="tertiary"
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
                                st.session_state.current_session_id = str(uuid.uuid4())
                                st.session_state.ui_messages = []
                                st.session_state.chat_history = []
                        st.rerun()
