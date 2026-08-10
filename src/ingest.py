import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import EMBEDDING_MODEL, DB_PATH, PDF_PATH


# Reads a PDF, chunks it, generates embeddings, and saves to ChromaDB.
def process_pdf_and_ingest(
    pdf_path: str = PDF_PATH,
    db_path: str = DB_PATH,
):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Please place a PDF file at: {pdf_path}")

    print(f"📄 Loading PDF: {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print("✂️  Chunking document...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # Remove the old vector database.
    if os.path.exists(db_path):
        print(f"🗑️ Removing existing vector database: {db_path}")
        shutil.rmtree(db_path)

    print("🧠 Generating embeddings and saving to local vector DB...")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=db_path
    )
    print(f"✅ Vector database created successfully at '{db_path}'.\n")
