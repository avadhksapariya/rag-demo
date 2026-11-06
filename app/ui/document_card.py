import json
import streamlit as st
from pathlib import Path

from app.core.config import DB_PATH


def load_document_profile(filename: str, db_path: Path = DB_PATH) -> dict:
    registry_file = Path(db_path) / "document_metadata.json"
    if registry_file.exists():
        try:
            with open(registry_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if filename in data:
                    return data[filename]
        except Exception:
            pass

    # Fallback placeholder if ingested before this feature
    return {
        "filename": filename,
        "summary": "Full overview available upon next re-indexing.",
        "entities": ["Document", "PDF"],
        "prompts": [],
        "total_pages": "-",
        "total_chunks": "-",
        "file_size_kb": "-",
        "indexed_at": "Earlier session",
    }


# Renders an executive summary card inside a clean modal dialog.
@st.dialog("📄 Document Summary & Quick Profile", width="medium")
def show_document_summary_dialog(filename: str):
    profile = load_document_profile(filename)

    # 1. Header / Title
    st.subheader(filename)
    st.caption(f"Indexed on {profile.get('indexed_at', 'N/A')}")

    # 2. Key Metrics Row
    m1, m2, m3 = st.columns(3)
    m1.metric("Pages", profile.get("total_pages", "-"))
    m2.metric("Indexed Chunks", profile.get("total_chunks", "-"))
    m3.metric("File Size", f"{profile.get('file_size_kb', '—')} KB")

    st.markdown("---")

    # 3. Executive Synopsis
    st.markdown("**Executive Synopsis**")
    st.info(profile.get("summary", "No synopsis recorded."))

    # 4. Key Entities / Themes (rendered as pill badges)
    entities = profile.get("entities", [])
    if entities:
        st.markdown("**Key Entities & Core Themes**")
        pills_html = " ".join(
            [
                f'<span style="background: rgba(255, 75, 75, 0.12); color: #ff6b6b; padding: 3px 8px; border-radius: 12px; font-size: 0.82em; margin-right: 6px; display: inline-block; margin-bottom: 4px;">🏷️ {e}</span>'
                for e in entities
            ]
        )
        st.markdown(pills_html, unsafe_allow_html=True)
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 5. Suggested Inquiries
    prompts = profile.get("prompts", [])
    if prompts:
        st.markdown("**Starter Questions**")
        for q in prompts:
            st.markdown(f"- *{q}*")

    st.markdown("---")

    # 6. Action: Set as active scope & close
    col_scope, col_close = st.columns([0.65, 0.35], vertical_alignment="center")
    with col_scope:
        if st.button(
            "🎯 Focus Chat on this Document", use_container_width=True, type="primary"
        ):
            st.session_state.selected_file = filename
            st.toast(f"Scoped chat to {filename}")
            st.rerun()

    with col_close:
        if st.button("Close", use_container_width=True):
            st.rerun()
