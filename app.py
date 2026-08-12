import os
import tempfile
import streamlit as st

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from src.ingest import process_pdf_and_ingest, get_indexed_documents
from src.query import answer_question
from src.config import DB_PATH

# 1. Page Configuration
st.set_page_config(
    page_title="RAG Demo",
    page_icon="📚",
    layout="centered",
)

st.title("📚 RAG (Gemini + ChromaDB)")

# 2. Initialize Session State Variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Stores LangChain Human/AI message objects

if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []  # Stores display dictionary for UI rendering

# 3. Sidebar: Multiple PDF Upload & File Selector
with st.sidebar:
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
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.ui_messages = []
        st.rerun()

# 4. Display Existing Chat Messages
for msg in st.session_state.ui_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📌 Retrieved Sources & Citations"):
                for idx, src in enumerate(msg["sources"]):
                    st.markdown(f"**[{idx+1}] Page {src['page']}**")
                    st.caption(src["text"])

# 5. Handle User Chat Input
if user_query := st.chat_input("Ask a question about your uploaded document..."):
    # Render user prompt in UI
    st.chat_message("user").markdown(user_query)
    st.session_state.ui_messages.append({"role": "user", "content": user_query})

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Call backend RAG pipeline with history
                response = answer_question(
                    user_query=user_query,
                    chat_history=st.session_state.chat_history,
                    db_path=DB_PATH,
                    selected_file=selected_file,
                )

                answer_text = response["answer"]
                st.markdown(answer_text)

                # Format retrieved sources
                sources_data = []
                for doc in response.get("context", []):
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
                            st.markdown(f"**[{idx+1}] Page {src['page']}**")
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

            except Exception as e:
                st.error(f"❌ Error generating response: {e}")
