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

## Necessary Commands

- Virtual environment:  
    - Create : `python -m venv .venv`  
    - Activate : `.\venv\Scripts\Activate`

- Install dependencies:  
    `pip install -r requirements.txt`

- Add Gemini API key to `.env` and run:  
    - If only in *Terminal* : `py main.py`  
    or  
    - else *Streamlit Web Interface* : `streamlit run app.py`

The application provides options to ingest the PDF, ask questions, or exit.

> **Note:** Run the ingestion step before asking questions.
