from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.indexes import SQLRecordManager, index

from src.config import EMBEDDING_MODEL, DB_PATH, PDF_PATH


# Reads a PDF, chunks it, generates embeddings, and saves to ChromaDB.
def process_pdf_and_ingest(
    pdf_path: Path = PDF_PATH,
    db_path: Path = DB_PATH,
):
    if not pdf_path.exists():
        raise FileNotFoundError(f"Please place a PDF file at: {pdf_path}")

    print(f"📄 Loading PDF: {pdf_path}...")
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    print("✂️  Chunking document...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # 1. Initialize Embeddings & Vector Store
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vector_db = Chroma(
        persist_directory=str(db_path),
        embedding_function=embeddings,
    )

    # 2. Initialize Record Manager (Stores metadata & chunk hashes locally)
    db_path.mkdir(parents=True, exist_ok=True)
    record_manager_path = db_path / "record_manager.db"

    namespace = "chroma/pdf_ingest"

    record_manager = SQLRecordManager(
        namespace,
        db_url=f"sqlite:///{record_manager_path.as_posix()}",
    )
    record_manager.create_schema()

    # 3. Perform Incremental Indexing
    # handles embedding generation, saving, updating, and cleanup internally.
    print("🧠 Checking for changes and updating vector DB...")
    indexing_result = index(
        docs_source=chunks,
        record_manager=record_manager,
        vector_store=vector_db,
        cleanup="incremental",  # Options: 'incremental', 'full', or None
        source_id_key="source",
        key_encoder="sha256",
    )

    print(f"✅ Indexing complete! Results: {indexing_result}\n")

    print(f"✅ Vector database created successfully at '{db_path}'.\n")
