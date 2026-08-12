import time
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.indexes import SQLRecordManager, index

from src.config import EMBEDDING_MODEL, DB_PATH, PDF_PATH


# Scans the data/ folder and ingests all .pdf files found.
def ingest_all_pdfs_in_directory(
    data_dir: str | Path = "data", db_path: str | Path = DB_PATH
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

    # Now .exists() works reliably on both str and Path inputs
    if not pdf_path.exists():
        raise FileNotFoundError(f"Please place a PDF file at: {pdf_path}")

    display_name = filename_override if filename_override else pdf_path.name

    print(f"📄 Loading PDF: {pdf_path}...")
    loader = PyMuPDFLoader(str(pdf_path))
    documents = loader.load()

    for doc in documents:
        doc.metadata["source_file"] = display_name

    print("✂️  Chunking document...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # ~250 tokens
        chunk_overlap=200,  # 20% duplicate text
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # Initialize Embeddings & Vector Store
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vector_db = Chroma(
        persist_directory=str(db_path),
        embedding_function=embeddings,
    )

    # Initialize Record Manager (Stores metadata & chunk hashes locally)
    namespace = "chroma/pdf_ingest"
    record_manager = SQLRecordManager(
        namespace,
        db_url=f"sqlite:///{db_path}/record_manager.db",
    )
    record_manager.create_schema()

    # Perform Incremental Indexing
    # handles embedding generation, saving, updating, and cleanup internally.
    print("🧠 Checking for changes and updating vector DB...")

    batch_size = 100
    if len(chunks) > batch_size:
        print(
            f"⚠️ Large document detected ({len(chunks)} chunks). Indexing in batches of {batch_size}..."
        )
        total_added = 0
        total_skipped = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            res = index(
                docs_source=chunks,
                record_manager=record_manager,
                vector_store=vector_db,
                cleanup="incremental",  # Options: 'incremental', 'full', or None
                source_id_key="source",
                key_encoder="sha256",
            )
            total_added += res.get("num_added", 0)
            total_skipped += res.get("num_skipped", 0)
            time.sleep(1)  # Brief pause between batches for API stability

            indexing_result = {"num_added": total_added, "num_skipped": total_skipped}

    else:
        indexing_result = index(
            docs_source=chunks,
            record_manager=record_manager,
            vector_store=vector_db,
            cleanup="incremental",  # Options: 'incremental', 'full', or None
            source_id_key="source",
            key_encoder="sha256",
        )

    print(f"✅ Indexing completefor '{display_name}'! Results: {indexing_result}\n")

    print(f"✅ Vector database created successfully at '{db_path}'.\n")

    return indexing_result


# Retrieves a list of all unique file names indexed in ChromaDB.
def get_indexed_documents(db_path: str | Path = DB_PATH) -> list[str]:
    db_path = Path(db_path)
    if not db_path.exists():
        return []

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    try:
        vector_db = Chroma(
            persist_directory=str(db_path), embedding_function=embeddings
        )
        data = vector_db.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
        files = {
            meta.get("source_file")
            for meta in metadatas
            if meta and "source_file" in meta
        }
        return sorted(list(files))
    except Exception:
        return []
