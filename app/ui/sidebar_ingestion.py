import os
import hashlib
import tempfile
import streamlit as st
from pathlib import Path

from app.core.config import DB_PATH, DATA_DIR
from app.storage.cache import semantic_cache
from app.src.ingestion.ingest import (
    process_pdf_and_ingest,
    get_indexed_documents,
    get_indexed_file_hashes,
    delete_document_from_index,
)


def render_ingestion_tab():
    st.markdown("#### 📄 Document Ingestion")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdf_uploader_{st.session_state.uploader_key}",
    )

    feedback_slot = st.empty()
    if st.session_state.show_success_counter > 0:
        feedback_slot.success(st.session_state.last_ingested_message)
        # Decrement: message lives only for this immediate rerun cycle
        st.session_state.show_success_counter -= 1
    else:
        st.session_state.last_ingested_message = None

    if uploaded_files:
        existing_hashes = get_indexed_file_hashes(DB_PATH)
        files_to_process = []
        duplicate_warnings = []

        for f in uploaded_files:
            file_bytes = f.getvalue()
            f_hash = hashlib.sha256(file_bytes).hexdigest()

            if f_hash in existing_hashes:
                orig_name = existing_hashes[f_hash]
                if orig_name == f.name:
                    duplicate_warnings.append(f"**{f.name}** is already indexed.")
                else:
                    duplicate_warnings.append(
                        f"**{f.name}** has identical content to **{orig_name}**."
                    )
            else:
                files_to_process.append(f)

        for warn in duplicate_warnings:
            st.warning(f"⚠️ {warn}")

        if files_to_process:
            btn_label = f"Process & Ingest ({len(files_to_process)} New File{'s' if len(files_to_process) > 1 else ''})"
            if st.button(btn_label, use_container_width=True, type="primary"):
                ingestion_failed = False
                with st.spinner("Processing PDF(s) and updating vector store..."):
                    # Persist copies to DATA_DIR so the PDF preview viewer can render pages
                    data_dir_path = Path(DATA_DIR)
                    data_dir_path.mkdir(parents=True, exist_ok=True)

                    for uploaded_file in files_to_process:
                        persisted_pdf_path = data_dir_path / uploaded_file.name
                        with open(persisted_pdf_path, "wb") as f_out:
                            f_out.write(uploaded_file.getvalue())

                        try:
                            process_pdf_and_ingest(
                                pdf_path=persisted_pdf_path,
                                db_path=DB_PATH,
                                filename_override=uploaded_file.name,
                            )
                        except Exception as e:
                            ingestion_failed = True
                            st.error(
                                f"❌ Error during ingestion of {uploaded_file.name}: {e}"
                            )

                if not ingestion_failed:
                    file_names = ", ".join([f.name for f in files_to_process])
                    st.session_state.last_ingested_message = (
                        f"✅ Ingested: **{file_names}**"
                    )
                    st.session_state.show_success_counter = 1
                    st.session_state.uploader_key += 1
                    st.rerun()

    st.divider()

    # --- SECTION: Indexed Documents List & Delete Controls ---
    st.markdown("#### 📚 Indexed Documents")
    current_indexed = get_indexed_documents(DB_PATH)

    if not current_indexed:
        st.caption("No documents indexed yet.")
    else:
        with st.container(height=220, border=True):
            for doc_name in current_indexed:
                col_name, col_btn = st.columns([0.82, 0.18])
                with col_name:
                    st.markdown(
                        f"<div style='padding-top: 6px; font-size: 0.9em; word-break: break-all;'>📄 {doc_name}</div>",
                        unsafe_allow_html=True,
                    )
                with col_btn:
                    if st.button(
                        "🗑️",
                        key=f"del_doc_{doc_name}",
                        help=f"Delete {doc_name} from vector store",
                        type="tertiary",
                    ):
                        with st.spinner(f"Deleting {doc_name}..."):
                            delete_document_from_index(doc_name, DB_PATH)
                            # Remove raw file from DATA_DIR as well
                            doc_path = Path(DATA_DIR) / doc_name
                            if doc_path.exists():
                                os.remove(doc_path)

                            # If current active chat filter was set to this file, reset to All
                            if st.session_state.get("selected_file") == doc_name:
                                st.session_state.selected_file = "All Documents"

                        st.toast(f"🗑️ Deleted **{doc_name}** successfully!")
                        st.rerun()

    st.divider()
    st.markdown("#### ⚡ Cache Operations")
    if st.button("Flush Cache", use_container_width=True):
        semantic_cache.clear()
        st.toast("🧹 Semantic cache cleared successfully!")
