from pathlib import Path
from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from tax_qa.tools import build_vector_store

Provider = Literal["openai", "gemini", "ollama"]

# src/tax_qa/dependencies.py -> src/tax_qa -> src -> repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PDF_PATH = _REPO_ROOT / ".research-data" / "ordinance.pdf"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    chat_provider: Provider = "openai"
    embedding_provider: Provider = "ollama"

    openai_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"

    openai_chat_model: str = "gpt-4o-mini"
    gemini_chat_model: str = "gemini-2.0-flash"
    ollama_chat_model: str = "llama3.2"

    openai_embedding_model: str = "text-embedding-3-small"
    gemini_embedding_model: str = "models/text-embedding-004"
    ollama_embedding_model: str = "bge-m3"

    pdf_path: Path = Field(default_factory=lambda: DEFAULT_PDF_PATH)
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
    return build_vector_store(settings.pdf_path, _build_embeddings(settings))
