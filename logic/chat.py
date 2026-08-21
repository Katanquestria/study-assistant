from langchain_core.messages import SystemMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document


def get_chat_model():
    return ChatAnthropic(
        model_name="claude-sonnet-5",
        )

def generate_answer(question: str, documents: list[Document], chat_model) -> str:

    context = "\n\n".join(
        f"[Quelle: {doc.metadata['source_file']}, Seite {doc.metadata['page']}]\n{doc.page_content}"
        for doc in documents
    )

    system_message = SystemMessage(
        content=(
        "Beantworte die Frage ausschließlich auf Basis des gegebenen Kontexts, "
        "ohne externes Wissen. Antworte auf Deutsch."
        )
    )

    human_message = HumanMessage(
        content= f"Kontext:\n{context}\n\nFrage:\n{question}"
    )

    response = chat_model.invoke([system_message, human_message])
    content = response.content

    if isinstance(content, str):
        return content
    return "".join(
        block["text"] for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )
