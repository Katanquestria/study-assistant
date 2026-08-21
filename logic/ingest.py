from logic.chunking import chunk_documents
from logic.loaders import load_file
from logic.vectorstore import add_documents, delete_by_source_file
from pathlib import Path
import hashlib
import json
import os





def compute_file_hash(path: str) -> str:
    with open(path, "rb") as file:
        file_data = file.read()
        return hashlib.sha256(file_data).hexdigest()


def load_manifest(manifest_path: str) -> dict:
    if not Path(manifest_path).exists():
        return {}
    else:
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            return json.load(manifest_file)


def save_manifest(manifest_path: str, manifest: dict) -> None:
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent = 4)


def ingest_files(paths: list[str], vectorstore, manifest_path: str) -> None:
    manifest = load_manifest(manifest_path)

    for path in paths:
        filename = os.path.basename(path)
        file_hash = compute_file_hash(path)

        if filename in manifest and manifest[filename] == file_hash:
            continue
        elif filename in manifest and manifest[filename] != file_hash:
            delete_by_source_file(vectorstore, filename)

        docs = load_file(path)
        chunks = chunk_documents(docs)

        for chunk in chunks:
            chunk.metadata["file_hash"] = file_hash
            chunk.metadata["format"] = Path(path).suffix.lower().lstrip(".")

        add_documents(vectorstore, chunks)
        manifest[filename] = file_hash

    save_manifest(manifest_path, manifest)