"""
Milestone 2 manual check: embed one PDF's chunks into Chroma, run a known
German query, print top-k results with similarity scores.

Run from the project root with the venv active:
    python scripts/manual_test_milestone2.py
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(str(Path(__file__).resolve().parent.parent))

from logic.loaders import load_pdf_text
from logic.chunking import chunk_documents
from logic.embeddings import get_embedding_model
from logic.vectorstore import get_vectorstore, query_vectorstore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "raw" / "Thermodynamik_RAG_Notizen.pdf"
PERSIST_DIRECTORY = PROJECT_ROOT / "data" / "vectorstore"
COLLECTION_NAME = "chemistry_notes"

QUERY = "Was besagt der erste Hauptsatz der Thermodynamik?"
K = 3


def main():
    pages = load_pdf_text(str(PDF_PATH))
    chunks = chunk_documents(pages)
    print(f"Prepared {len(chunks)} chunk(s) from {PDF_PATH.name}.\n")

    print("Loading embedding model (first run downloads the model)...")
    embedding_model = get_embedding_model()

    print("Building/loading vector store (first run embeds all chunks)...\n")
    vectorstore = get_vectorstore(
        documents=chunks,
        embedding_model=embedding_model,
        persist_directory=str(PERSIST_DIRECTORY),
        collection_name=COLLECTION_NAME,
    )

    print(f"Query: {QUERY}\n")
    results = query_vectorstore(vectorstore, QUERY, k=K)

    for i, (doc, score) in enumerate(results, start=1):
        preview = doc.page_content[:300].replace("\n", " ")
        print(f"--- Result {i} (score: {score:.4f}) ---")
        print(f"source_file: {doc.metadata['source_file']}")
        print(f"page: {doc.metadata['page']}")
        print(f"preview: {preview}...")
        print()


if __name__ == "__main__":
    main()
