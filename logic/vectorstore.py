from langchain_chroma import Chroma

def get_vectorstore(embedding_model, persist_directory, collection_name):
    return Chroma(persist_directory=persist_directory, embedding_function=embedding_model, collection_name=collection_name)


def query_vectorstore(vectorstore, query, k):
    return vectorstore.similarity_search_with_score(query, k=k)


def add_documents(vectorstore, chunks):
    return vectorstore.add_documents(chunks)


def delete_by_source_file(vectorstore, source_file: str) -> None:
    vectorstore.delete(where={"source_file": source_file})
