# RAG Demo – Gemini & ChromaDB

A simple **Retrieval-Augmented Generation (RAG)** application that uses a PDF document as the knowledge source.

The project uses **Hugging Face** (earlier Gemini) embedding model, **ChromaDB** for vector storage and retrieval, and a **OpenRouter Free LLM** with default fallback of **Gemini LLM** to generate answers based on the retrieved document context.  

The application provides options to ingest the PDF, ask questions, or exit.

## Tech Stack

- Python
- LangChain
- ChromaDB
- Streamlit

## Embedding & LLMs

- Hugging Face
- Google Gemini
- OpenRouter

## How It Works

```text
PDF → Chunking → Embeddings → ChromaDB
                              ↓
Question → Retrieval → LLM → Answer
```

## Necessary Commands

- Virtual environment:  
    - Create : `python -m venv .venv`  
    - Activate : `.\venv\Scripts\Activate`

- Install dependencies:  
    `pip install -r requirements.txt`

- Add Gemini API key to `.env` and run:  
    - If only in *Terminal* : `py -m app.main`  
    or  
    - else *Streamlit Web Interface* : `py -m streamlit run app/app_st.py`

- Automated Evaluation Benchmarks:  
    - `py -m tests.evaluate_rag`

### *Notes:*  
- Run the ingestion step before asking questions.

- when changing embedding models...  
    - remove old vector database : `Remove-Item -Recurse -Force app\vector_db`
    - rebuild the vector database by running ingestion.


