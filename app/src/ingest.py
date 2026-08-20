import time
import base64
import pymupdf
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_classic.indexes import SQLRecordManager, index

from app.src.embeddings import get_embeddings
from app.src.config import EMBEDDING_MODEL, LLM_MODEL, DB_PATH, DATA_DIR, PDF_PATH


# Uses Gemini Vision to summarize an extracted image or diagram.
def generate_image_summary(image_bytes: bytes) -> str:
    try:
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL)
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Describe this image, diagram, or table in detail. Extract any relevant text, data, or structural information. If it is just a decorative background, say 'Decorative image'.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                },
            ]
        )
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        print(f"⚠️ Vision API Error: {e}")
        return "Image could not be summarized."


# Scans the data/ folder and ingests all .pdf files found.
def ingest_all_pdfs_in_directory(
    data_dir: str | Path = DATA_DIR, db_path: str | Path = DB_PATH
):
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Directory not found: {data_path}")

    pdf_files = list(data_path.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️ No PDF files found in '{data_path}'.")
        return

    print(f"📂 Found {len(pdf_files)} PDF(s) in '{data_path}':")
    for pdf in pdf_files:
        print(f"  • {pdf.name}")
    print()

    for pdf_path in pdf_files:
        try:
            process_pdf_and_ingest(
                pdf_path=pdf_path, db_path=db_path, filename_override=pdf_path.name
            )
        except Exception as e:
            print(f"❌ Error ingesting {pdf_path.name}: {e}\n")


# Reads a PDF, chunks it, generates embeddings, and saves to ChromaDB.
def process_pdf_and_ingest(
    pdf_path: str | Path = PDF_PATH,
    db_path: str | Path = DB_PATH,
    filename_override: str | None = None,
):
    # Ensure pdf_path and db_path are Path objects
    pdf_path = Path(pdf_path)
    db_path = Path(db_path)

    # .exists() works reliably on both str and Path inputs
    if not pdf_path.exists():
        raise FileNotFoundError(f"Please place a PDF file at: {pdf_path}")

    display_name = filename_override if filename_override else pdf_path.name

    print(f"📄 Loading PDF: {pdf_path}...")
    loader = PyMuPDFLoader(str(pdf_path))
    documents = loader.load()

    # MULTIMODAL EXTRACTION STAGE
    print(f"🖼️ Scanning '{display_name}' for images and diagrams...")
    pdf_document = pymupdf.open(str(pdf_path))

    for i, doc in enumerate(documents):
        page_num = doc.metadata.get("page", i)
        page = pdf_document[page_num]
        images = page.get_images(full=True)

        if images:
            print(
                f"   -> Found {len(images)} image(s) on Page {page_num + 1}. Summarizing..."
            )
            image_summaries = []

            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]

                summary = generate_image_summary(image_bytes)

                if "Decorative image" not in summary:
                    image_summaries.append(
                        f"\n\n--- [Image/Diagram Summary] ---\n{summary}"
                    )

                time.sleep(1)  # Prevent rate limiting on Vision API

            # Append visual summaries directly to the page's text content
            if image_summaries:
                doc.page_content += "".join(image_summaries)

    # Normalize metadata
    for doc in documents:
        doc.metadata["source_file"] = display_name

    # Define Splitters
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

    # Split into Parent Documents & Child Chunks
    parent_docs = parent_splitter.split_documents(documents)

    # Generate Parent IDs & Link Child Chunks to Parent IDs
    child_docs = []
    parent_id_key = "doc_id"
    parent_records = {}  # {parent_id: parent_doc}

    for i, p_doc in enumerate(parent_docs):
        # Create a deterministic parent ID based on file and index
        parent_id = f"{display_name}_parent_{i}"
        parent_records[parent_id] = p_doc

        # Split parent into child chunks
        sub_docs = child_splitter.split_documents([p_doc])
        for c_doc in sub_docs:
            c_doc.metadata[parent_id_key] = parent_id
            c_doc.metadata["source_file"] = display_name
            child_docs.append(c_doc)

    print(
        f"✂️ Created {len(parent_docs)} Parent Docs and {len(child_docs)} Child Chunks."
    )

    # Store Parents in Local Doc Store
    doc_store_path = db_path / "doc_store"
    doc_store_path.mkdir(parents=True, exist_ok=True)
    fs = LocalFileStore(str(doc_store_path))
    store = create_kv_docstore(fs)
    store.mset(list(parent_records.items()))

    # Initialize Embeddings & Vector Store
    embeddings = get_embeddings()  # GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    chroma_path = db_path / "chroma"
    vector_db = Chroma(
        collection_name="child_chunks",
        persist_directory=str(chroma_path),
        embedding_function=embeddings,
    )

    # Initialize SQL Record Manager for Child Chunks (Stores metadata & chunk hashes locally)
    namespace = f"chroma/{display_name}"
    record_manager = SQLRecordManager(
        namespace,
        db_url=f"sqlite:///{db_path}/record_manager.db",
    )
    record_manager.create_schema()

    # Incremental Indexing with Batching Safeguards
    print("🧠 Checking for changes and updating vector DB...")

    batch_size = 100
    if len(child_docs) > batch_size:
        print(
            f"⚠️ Large document detected ({len(child_docs)} child chunks). Indexing in batches of {batch_size}..."
        )
        total_added = 0
        total_skipped = 0

        for i in range(0, len(child_docs), batch_size):
            batch = child_docs[i : i + batch_size]
            res = index(
                docs_source=batch,
                record_manager=record_manager,
                vector_store=vector_db,
                cleanup="incremental",  # Options: 'incremental', 'full', or None
                source_id_key="source_file",
                key_encoder="sha256",
            )
            total_added += res.get("num_added", 0)
            total_skipped += res.get("num_skipped", 0)
            time.sleep(1)  # Brief pause between batches for API stability

            indexing_result = {"num_added": total_added, "num_skipped": total_skipped}

    else:
        indexing_result = index(
            docs_source=child_docs,
            record_manager=record_manager,
            vector_store=vector_db,
            cleanup="incremental",
            source_id_key="source_file",
            key_encoder="sha256",
        )

    print(f"✅ Indexing complete for '{display_name}'! Results: {indexing_result}\n")

    print(f"✅ Vector database created successfully at '{db_path}'.\n")

    return indexing_result


# Retrieves a list of all unique file names indexed in ChromaDB.
def get_indexed_documents(db_path: str | Path = DB_PATH) -> list[str]:
    db_path = Path(db_path)
    chroma_path = db_path / "chroma"
    if not chroma_path.exists():
        return []

    embeddings = get_embeddings()

    try:
        vector_db = Chroma(
            collection_name="child_chunks",
            persist_directory=str(chroma_path),
            embedding_function=embeddings,
        )
        data = vector_db.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
        files = {
            meta.get("source_file")
            for meta in metadatas
            if meta and "source_file" in meta
        }
        return sorted(list(files))
    except Exception as e:
        print(f"Error fetching documents: {e}")
        return []
