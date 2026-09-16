from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from triange_agent.tools import build_vector_store


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: SecretStr | None = None
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 400
    chunk_overlap: int = 50
    retrieval_k: int = 4


def build_llm(settings: Settings | None = None) -> ChatOpenAI:
    settings = settings or Settings()
    return ChatOpenAI(model=settings.chat_model, api_key=settings.openai_api_key)


def build_retriever(settings: Settings | None = None) -> InMemoryVectorStore:
    settings = settings or Settings()
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model, api_key=settings.openai_api_key
    )
    return build_vector_store(
        embeddings=embeddings,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
