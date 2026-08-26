# Study Assistant

A RAG (Retrieval-Augmented Generation) chatbot that answers chemistry questions strictly from your own course notes ("Skript"), PDFs (typed or scanned), and Word documents, in German.

Built for Studienkolleg-style exam prep, where notes often lack problem sets and exams require full reasoning chains, not just final answers.

## How it works

1. Upload your notes (PDF or `.docx`) through the Streamlit UI.
2. Each file is loaded, split into chunks, and embedded locally with a multilingual sentence-transformer model.
3. Chunks are stored in a persistent Chroma vector store on disk, keyed by a content hash — re-running with the same files skips re-embedding, so nothing is lost between sessions and nothing is redone unnecessarily.
4. Ask a question in the chat. The most relevant chunks are retrieved and passed to Claude, which answers **only** from that context and cites the source file and page.

## Setup

### 1. Python environment

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. Tesseract OCR (for scanned/photographed pages)

Scanned-page ingestion needs the Tesseract binary installed at the OS level (not just the `pytesseract` Python package), with the German language pack (`deu`). Install it separately and make sure it's on your `PATH` before uploading scanned PDFs.

### 3. API key

Copy `.env.example` to `.env` and add your Anthropic API key and Langsmith API key:

```
ANTHROPIC_API_KEY=your-api-key-here
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-api-key-here
```

## Running

```bash
streamlit run ui/app.py
```

Upload files, then ask questions in the chat box. Click "Quellen anzeigen" under an answer to see which file/page it was grounded in.

## Project structure

```
study-assistant/
├── data/
│   ├── raw/            # uploaded source files (gitignored)
│   └── vectorstore/    # persisted Chroma DB (gitignored)
├── logic/               # ingestion, chunking, embeddings, vectorstore, chat — no UI code
├── ui/
│   └── app.py           # Streamlit UI only
└── scripts/              # manual test scripts used during development
```

## Status

**v1 complete** — multi-format ingestion (text PDF, scanned PDF via OCR, `.docx`), incremental embedding, Streamlit chat UI with citations, and a vector store that survives restarts.

Planned next: a web-search fallback for topics notes don't cover well, and a problem-set generator that grades full reasoning chains rather than final answers.
