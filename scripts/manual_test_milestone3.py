"""
Milestone 3 manual check: retrieve context for a real question, then generate
a Claude Sonnet 5 answer from it. First milestone that costs API money.

Run from the project root with the venv active:
    python scripts/manual_test_milestone3.py
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(str(Path(__file__).resolve().parent.parent))

load_dotenv()

from logic.loaders import load_pdf_text
from logic.chunking import chunk_documents
from logic.embeddings import get_embedding_model
from logic.vectorstore import get_vectorstore, query_vectorstore
from logic.chat import get_chat_model, generate_answer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "raw" / "Thermodynamik_RAG_Notizen.pdf"
PERSIST_DIRECTORY = PROJECT_ROOT / "data" / "vectorstore"
COLLECTION_NAME = "chemistry_notes"

QUESTION = "Was besagt der erste Hauptsatz der Thermodynamik?"
K = 3


def main():
    pages = load_pdf_text(str(PDF_PATH))
    chunks = chunk_documents(pages)

    embedding_model = get_embedding_model()
    vectorstore = get_vectorstore(
        documents=chunks,
        embedding_model=embedding_model,
        persist_directory=str(PERSIST_DIRECTORY),
        collection_name=COLLECTION_NAME,
    )

    results = query_vectorstore(vectorstore, QUESTION, k=K)
    retrieved_docs = [doc for doc, score in results]

    print(f"Frage: {QUESTION}\n")
    print(f"Retrieved {len(retrieved_docs)} chunk(s) as context:")
    for doc in retrieved_docs:
        print(f"  - {doc.metadata['source_file']}, Seite {doc.metadata['page']}")
    print()

    chat_model = get_chat_model()
    answer = generate_answer(QUESTION, retrieved_docs, chat_model)

    print("--- Antwort ---")
    print(answer)


if __name__ == "__main__":
    main()
