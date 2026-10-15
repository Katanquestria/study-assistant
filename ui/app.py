import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from logic.chat import get_chat_model, generate_answer
from logic.embeddings import get_embedding_model
from logic.ingest import ingest_files
from logic.vectorstore import get_vectorstore, query_vectorstore
import streamlit as st

load_dotenv()


RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PERSIST_DIRECTORY = Path(__file__).resolve().parent.parent / "data" / "vectorstore"
MANIFEST_PATH = PERSIST_DIRECTORY / "manifest.json"
COLLECTION_NAME = "Documents"

@st.cache_resource
def load_vectorstore():
    embedding_model = get_embedding_model()
    return get_vectorstore(embedding_model, str(PERSIST_DIRECTORY), COLLECTION_NAME)

@st.cache_resource
def load_chat_model():
    return get_chat_model()

st.session_state.setdefault("chat_history", [])

uploaded_files = st.file_uploader("Dateien hochladen", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        save_path = RAW_DIR / uploaded_file.name
        save_path.write_bytes(uploaded_file.getvalue())

    paths = [str(RAW_DIR / f.name) for f in uploaded_files]
    ingest_files(paths, load_vectorstore(), str(MANIFEST_PATH))

if question := st.chat_input("Ask AI"):
    st.session_state["chat_history"].append({"role": "user", "content": question})

    vectorstore = load_vectorstore()
    results = query_vectorstore(vectorstore, question, k=6)
    retrieved_docs = [doc for doc, score in results]
    answer = generate_answer(question, retrieved_docs, load_chat_model())
    
    st.session_state["chat_history"].append({
        "role": "assistant",
        "content": answer,
        "sources": [(doc.metadata["source_file"], doc.metadata["page"]) for doc in retrieved_docs]
    })


for chat in st.session_state["chat_history"]:
    if chat["role"] == "user":
        with st.chat_message("user"):
            st.write(chat["content"])
    else:
        with st.chat_message("assistant"):
            st.write(chat["content"])
            with st.popover("Quellen anzeigen"):
                source_placeholder = []
                for source_file, page in chat["sources"]:
                    if f"Quelle: {source_file}, Seite: {page}" not in source_placeholder:
                        st.write(f"Quelle: {source_file}, Seite: {page}")
                        source_placeholder.append(f"Quelle: {source_file}, Seite: {page}")
                        
