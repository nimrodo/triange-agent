from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI

from triange_agent.tools import build_vector_store

CHAT_MODEL = "gpt-4o-mini"


def build_llm() -> ChatOpenAI:
    return ChatOpenAI(model=CHAT_MODEL)


def build_retriever() -> InMemoryVectorStore:
    return build_vector_store()
