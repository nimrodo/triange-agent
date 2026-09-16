from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from triange_agent.tools import build_vector_store

Provider = Literal["openai", "gemini", "ollama"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    chat_provider: Provider = "openai"
    embedding_provider: Provider = "openai"

    openai_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"

    openai_chat_model: str = "gpt-4o-mini"
    gemini_chat_model: str = "gemini-2.0-flash"
    ollama_chat_model: str = "llama3.2"

    openai_embedding_model: str = "text-embedding-3-small"
    gemini_embedding_model: str = "models/text-embedding-004"
    ollama_embedding_model: str = "nomic-embed-text"

    chunk_size: int = 400
    chunk_overlap: int = 50
    retrieval_k: int = 4


def build_llm(settings: Settings | None = None) -> BaseChatModel:
    settings = settings or Settings()
    match settings.chat_provider:
        case "openai":
            return ChatOpenAI(
                model=settings.openai_chat_model, api_key=settings.openai_api_key
            )
        case "gemini":
            return ChatGoogleGenerativeAI(
                model=settings.gemini_chat_model, api_key=settings.gemini_api_key
            )
        case "ollama":
            return ChatOllama(
                model=settings.ollama_chat_model, base_url=settings.ollama_base_url
            )


def _build_embeddings(settings: Settings) -> Embeddings:
    match settings.embedding_provider:
        case "openai":
            return OpenAIEmbeddings(
                model=settings.openai_embedding_model, api_key=settings.openai_api_key
            )
        case "gemini":
            return GoogleGenerativeAIEmbeddings(
                model=settings.gemini_embedding_model, api_key=settings.gemini_api_key
            )
        case "ollama":
            return OllamaEmbeddings(
                model=settings.ollama_embedding_model, base_url=settings.ollama_base_url
            )


def build_retriever(settings: Settings | None = None) -> InMemoryVectorStore:
    settings = settings or Settings()
    return build_vector_store(
        embeddings=_build_embeddings(settings),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
