from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from triange_agent.state import RetrievedSnippet

KNOWLEDGE_BASE_DIR = Path(__file__).parent / "knowledge_base"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


def load_knowledge_base(directory: Path = KNOWLEDGE_BASE_DIR) -> list[Document]:
    return [
        Document(page_content=path.read_text(), metadata={"source": path.name})
        for path in sorted(directory.glob("*.md"))
    ]


def build_vector_store(embeddings: Embeddings | None = None) -> InMemoryVectorStore:
    embeddings = embeddings or OpenAIEmbeddings(model=EMBEDDING_MODEL)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(load_knowledge_base())

    return InMemoryVectorStore.from_documents(chunks, embeddings)


def retrieve_snippets(
    vector_store: InMemoryVectorStore, query: str, k: int = 4
) -> list[RetrievedSnippet]:
    results = vector_store.similarity_search(query, k=k)
    return [
        RetrievedSnippet(content=doc.page_content, source=doc.metadata["source"])
        for doc in results
    ]
