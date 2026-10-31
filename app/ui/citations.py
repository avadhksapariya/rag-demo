import fitz
import streamlit as st

from pathlib import Path
from app.core.config import DATA_DIR


@st.dialog("📄 Source Document Preview", width="large")
def show_pdf_page_model(file_name: str, page_number: int, snippet: str):
    st.markdown(f"**Document:** `{file_name}` — **Page:** `{page_number}`")

    st.markdown(
        f"""
        <div style="
            background-color: rgba(28, 131, 225, 0.1);
            border-left: 4px solid #1c83e1;
            padding: 12px 16px;
            border-radius: 4px;
            margin-bottom: 16px;
            font-size: 0.95em;
            line-height: 1.6;
        ">
            <strong style="color: #58a6ff; display: block; margin-bottom: 6px;">
                📌 Retrieved Context Snippet:
            </strong>
            <div>{snippet}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pdf_file_path = Path(DATA_DIR) / file_name
    if not pdf_file_path.exists():
        st.warning(f"Original PDF file not found at `{pdf_file_path}`.")
        return

    try:
        doc = fitz.open(str(pdf_file_path))
        target_page_index = max(0, page_number - 1)

        if target_page_index < len(doc):
            page = doc[target_page_index]
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            st.image(
                img_bytes,
                caption=f"Page {page_number} preview",
                use_container_width=True,
            )
        else:
            st.error(f"Page {page_number} exceeds total document pages ({len(doc)}).")
    except Exception as e:
        st.error(f"Failed to render PDF page preview: {e}")


# Single reusable renderer for both historical messages and live turn outputs.
def render_citations(sources: list, unique_prefix: str):
    if not sources:
        return

    with st.expander("📌 Retrieved Sources & Citations"):
        for idx, src in enumerate(sources):
            page = src.get("page")
            page_info = f" - Page {page}" if page is not None else ""
            file_name = src.get("file", "Document")
            snippet = src.get("text", "")

            c_label, c_btn = st.columns([0.82, 0.18])
            with c_label:
                st.markdown(f"**[{idx+1}] {file_name}{page_info}**")
            with c_btn:
                if st.button(
                    "👁️ View",
                    key=f"cite_view_{unique_prefix}_{idx}",
                    type="tertiary",
                    help="View exact page in PDF",
                ):
                    show_pdf_page_model(
                        file_name=file_name,
                        page_number=page if page else 1,
                        snippet=snippet,
                    )
            st.markdown(
                f"<div style='font-size: 0.9em; line-height: 1.5; color: #636e72;'>{src['text']}</div>",
                unsafe_allow_html=True,
            )
