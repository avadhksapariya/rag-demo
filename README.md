# RAG Demo – Gemini & ChromaDB

A simple **Retrieval-Augmented Generation (RAG)** application that uses a PDF document as the knowledge source.

The project uses **Gemini embeddings**, **ChromaDB** for vector storage and retrieval, and a **Gemini LLM** to generate answers based on the retrieved document context.

## Tech Stack

- Python
- LangChain
- Google Gemini
- ChromaDB

## How It Works

```text
PDF → Chunking → Embeddings → ChromaDB
                              ↓
Question → Retrieval → Gemini LLM → Answer
```

## Run

Add your Gemini API key to `.env` and run:

`py main.py`

The application provides options to ingest the PDF, ask questions, or exit.

> **Note:** Run the ingestion step before asking questions.
