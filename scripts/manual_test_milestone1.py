"""
Milestone 1 manual check: load one PDF, chunk it, print the results.

Run from the project root with the venv active:
    python scripts/manual_test_milestone1.py
"""

import sys
from pathlib import Path

# Windows terminals default to cp1252, which can't print German umlauts or
# symbols like Δ that show up in chemistry/thermodynamics notes.
sys.stdout.reconfigure(encoding="utf-8")

# Allow running this script directly without installing the project as a package.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from logic.loaders import load_pdf_text
from logic.chunking import chunk_documents

PDF_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "Thermodynamik_RAG_Notizen.pdf"


def main():
    pages = load_pdf_text(str(PDF_PATH))
    print(f"Loaded {len(pages)} page(s) with extractable text.\n")

    chunks = chunk_documents(pages)
    print(f"Split into {len(chunks)} chunk(s).\n")

    for i, chunk in enumerate(chunks, start=1):
        preview = chunk.page_content[:200].replace("\n", " ")
        print(f"--- Chunk {i} ---")
        print(f"source_file: {chunk.metadata['source_file']}")
        print(f"page: {chunk.metadata['page']}")
        print(f"length: {len(chunk.page_content)} chars")
        print(f"preview: {preview}...")
        print()


if __name__ == "__main__":
    main()
