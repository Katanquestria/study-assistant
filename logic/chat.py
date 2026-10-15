from langchain_core.messages import SystemMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document

WEB_SEARCH_TOOL = {
    "type": "web_search_20260209",
    "name": "web_search",
    "max_uses": 3,
}


def get_chat_model():
    model = ChatAnthropic(model_name="claude-sonnet-5", timeout=None, stop=None)
    return model.bind_tools([WEB_SEARCH_TOOL])


def build_messages(question: str, documents: list[Document]) -> list:
    context = "\n\n".join(
        f"[Quelle: {doc.metadata['source_file']}, Seite {doc.metadata['page']}]\n{doc.page_content}"
        for doc in documents
    )

    system_message = SystemMessage(
        content=(
            "Beantworte die Frage bevorzugt auf Basis des gegebenen Kontexts aus den Kursnotizen. "
            "Wenn der Kontext nicht ausreicht, um die Frage vollständig und korrekt zu beantworten, "
            "nutze die Websuche, um fehlende Informationen zu ergänzen. "
            "Gib in der Antwort immer an, ob eine Information aus den Notizen oder aus einer Websuche "
            "stammt (bei Websuche-Ergebnissen nenne die Quelle im Antworttext, z. B. Website-Name oder URL). "
            "Antworte auf Deutsch, unabhängig von der Sprache der Quelle."
        )
    )

    human_message = HumanMessage(content=f"Kontext:\n{context}\n\nFrage:\n{question}")
    return [system_message, human_message]


def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        block["text"] for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def generate_answer(question: str, documents: list[Document], chat_model) -> str:
    response = chat_model.invoke(build_messages(question, documents))
    return extract_text(response.content)
