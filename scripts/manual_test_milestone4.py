import sys
from pathlib import Path

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(str(Path(__file__).resolve().parent.parent))

load_dotenv()

from logic.embeddings import get_embedding_model
from logic.ingest import ingest_files, load_manifest
from logic.vectorstore import get_vectorstore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
COLLECTION_NAME = "Documents"
PERSIST_DIRECTORY = PROJECT_ROOT / "data" / "vectorstore"
MANIFEST_PATH = PERSIST_DIRECTORY / "manifest.json"

paths = [str(p) for p in RAW_DIR.iterdir() if p.suffix.lower() in (".pdf", ".docx")]
embedding_model = get_embedding_model()
vectorstore = get_vectorstore(embedding_model, str(PERSIST_DIRECTORY), COLLECTION_NAME)

ingest_files(paths, vectorstore, str(MANIFEST_PATH))
