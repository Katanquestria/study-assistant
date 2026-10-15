"""
Milestone 5 manual check: web-search fallback when local notes don't cover a question.
Run from the project root with the venv active: python scripts/manual_test_milestone5.py
Adjust COVERED_QUESTION / UNCOVERED_QUESTION to match what's actually in your ingested notes.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(str(Path(__file__).resolve().parent.parent))

load_dotenv()

from logic.embeddings import get_embedding_model
from logic.vectorstore import get_vectorstore, query_vectorstore
from logic.chat import get_chat_model, build_messages, extract_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIRECTORY = PROJECT_ROOT / "data" / "vectorstore"
COLLECTION_NAME = "Documents"
K = 3

COVERED_QUESTION = "Was besagt der erste Hauptsatz der Thermodynamik?"
UNCOVERED_QUESTION = "Welche Neuigkeiten gab es diese Woche zur Kernfusionsforschung?"


def used_web_search(content) -> bool:
    if isinstance(content, str):
        return False
    return any(
        isinstance(block, dict) and block.get("type") in ("server_tool_use", "web_search_tool_result")
        for block in content
    )


def run_case(label: str, question: str, vectorstore, chat_model) -> bool:
    results = query_vectorstore(vectorstore, question, k=K)
    retrieved_docs = [doc for doc, score in results]

    print(f"--- {label} ---")
    print(f"Frage: {question}")
    print(f"Retrieved {len(retrieved_docs)} chunk(s):")
    for doc, score in results:
        print(f"  - {doc.metadata['source_file']}, Seite {doc.metadata['page']} (score={score:.4f})")

    response = chat_model.invoke(build_messages(question, retrieved_docs))
    searched = used_web_search(response.content)
    answer = extract_text(response.content)

    print(f"Web search fired: {searched}")
    print("Antwort:")
    print(answer)
    print()
    return searched


def main():
    embedding_model = get_embedding_model()
    vectorstore = get_vectorstore(embedding_model, str(PERSIST_DIRECTORY), COLLECTION_NAME)
    chat_model = get_chat_model()

    covered_searched = run_case("Fall 1: Frage aus den Notizen (Regressionstest)", COVERED_QUESTION, vectorstore, chat_model)
    uncovered_searched = run_case("Fall 2: Frage NICHT in den Notizen (Fallback-Test)", UNCOVERED_QUESTION, vectorstore, chat_model)

    print("=== Zusammenfassung ===")
    print(f"Fall 1 (erwartet: False) -> {covered_searched}")
    print(f"Fall 2 (erwartet: True)  -> {uncovered_searched}")


if __name__ == "__main__":
    main()
