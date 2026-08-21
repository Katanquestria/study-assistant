from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_documents(documents: list[Document]) -> list[Document]:
    # Splits each input Document independently, so chunk_overlap never spans
    # a page boundary — text cut mid-derivation across two pages stays cut.

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    return chunks