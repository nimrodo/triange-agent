import re
from pathlib import Path
from typing import Literal

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from tax_qa.tools import build_vector_store

Provider = Literal["openai", "gemini", "ollama"]

# src/tax_qa/dependencies.py -> src/tax_qa -> src -> repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PDF_PATH = _REPO_ROOT / ".research-data" / "ordinance.pdf"
DEFAULT_VECTOR_STORE_DIR = _REPO_ROOT / ".vector-store"

_COLLECTION_NAME_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_-]+")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    chat_provider: Provider = "ollama"
    embedding_provider: Provider = "ollama"

    openai_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GATEWAY_GEMINI_UPSTREAM_KEY"),
    )
    ollama_base_url: str = "http://localhost:11434"

    openai_chat_model: str = "gpt-4o-mini"
    gemini_chat_model: str = "gemini-3.6-flash"
    ollama_chat_model: str = "llama3.2"

    openai_embedding_model: str = "text-embedding-3-small"
    gemini_embedding_model: str = "models/gemini-embedding-001"
    ollama_embedding_model: str = "bge-m3"

    pdf_path: Path = Field(default_factory=lambda: DEFAULT_PDF_PATH)
    vector_store_dir: Path = Field(default_factory=lambda: DEFAULT_VECTOR_STORE_DIR)
    retrieval_k: int = 4

    # Free-tier embedding APIs (e.g. Gemini) can cap both tokens-per-request
    # and requests-per-minute far below what a local Ollama server allows --
    # these let indexing fit under either without touching the default path.
    embedding_batch_size: int = 100
    embedding_request_delay_seconds: float = 0.0


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


def _embedding_model_name(settings: Settings) -> str:
    match settings.embedding_provider:
        case "openai":
            return settings.openai_embedding_model
        case "gemini":
            return settings.gemini_embedding_model
        case "ollama":
            return settings.ollama_embedding_model


def _collection_name(settings: Settings) -> str:
    # Different providers/models produce incompatible embedding vectors, so
    # each gets its own persisted collection rather than overwriting a shared
    # one on every provider switch.
    raw = f"{settings.embedding_provider}_{_embedding_model_name(settings)}"
    return _COLLECTION_NAME_SANITIZE_RE.sub("_", raw).strip("_")[:63]


def build_retriever(settings: Settings | None = None) -> Chroma:
    settings = settings or Settings()
    return build_vector_store(
        settings.pdf_path,
        _build_embeddings(settings),
        batch_size=settings.embedding_batch_size,
        persist_directory=settings.vector_store_dir,
        collection_name=_collection_name(settings),
        request_delay_seconds=settings.embedding_request_delay_seconds,
    )
