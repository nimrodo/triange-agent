"""Standalone sandbox for comparing chunking and embedding choices against a
real KnowledgeBase doc. Never imported by tools.py or nodes/ -- run directly:

    uv run python scripts/compare_embeddings.py
"""

from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

KB_DOC = Path(__file__).parent.parent / "src/triange_agent/knowledge_base/billing.md"
QUERY = "How long do I have to request a refund?"


@dataclass
class Config:
    label: str
    chunk_size: int
    chunk_overlap: int
    embeddings: Embeddings


CONFIGS = [
    Config(
        "decided (400/50, 3-small)",
        400,
        50,
        OpenAIEmbeddings(model="text-embedding-3-small"),
    ),
    Config(
        "smaller chunks (200/50, 3-small)",
        200,
        50,
        OpenAIEmbeddings(model="text-embedding-3-small"),
    ),
    Config(
        "larger chunks (800/100, 3-small)",
        800,
        100,
        OpenAIEmbeddings(model="text-embedding-3-small"),
    ),
    Config(
        "no overlap (400/0, 3-small)",
        400,
        0,
        OpenAIEmbeddings(model="text-embedding-3-small"),
    ),
    Config(
        "larger model (400/50, 3-large)",
        400,
        50,
        OpenAIEmbeddings(model="text-embedding-3-large"),
    ),
]


def run(config: Config) -> None:
    document = Document(
        page_content=KB_DOC.read_text(), metadata={"source": KB_DOC.name}
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
    )
    chunks = splitter.split_documents([document])
    vector_store = InMemoryVectorStore.from_documents(chunks, config.embeddings)

    print(f"\n=== {config.label} ({len(chunks)} chunks) ===")
    for doc, score in vector_store.similarity_search_with_score(QUERY, k=3):
        preview = doc.page_content.replace("\n", " ")[:160]
        print(f"  score={score:.4f}  {preview}...")


def main() -> None:
    for config in CONFIGS:
        run(config)


if __name__ == "__main__":
    main()
