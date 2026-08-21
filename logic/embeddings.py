from langchain_huggingface import HuggingFaceEmbeddings

class E5Embeddings(HuggingFaceEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = ["passage: " + text for text in texts]
        return super().embed_documents(prefixed)

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query("query: " + text)

def get_embedding_model():
    return E5Embeddings(
        model = "intfloat/multilingual-e5-large"
    )