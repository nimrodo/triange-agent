from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool, tool
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


def build_vector_store(
    embeddings: Embeddings | None = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> InMemoryVectorStore:
    embeddings = embeddings or OpenAIEmbeddings(model=EMBEDDING_MODEL)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(load_knowledge_base())

    return InMemoryVectorStore.from_documents(chunks, embeddings)


def retrieve_snippets(
    retriever: InMemoryVectorStore, query: str, k: int = 4
) -> list[RetrievedSnippet]:
    results = retriever.similarity_search(query, k=k)
    return [
        RetrievedSnippet(content=doc.page_content, source=doc.metadata["source"])
        for doc in results
    ]


def format_snippets(snippets: list[RetrievedSnippet]) -> str:
    return "\n\n".join(f"[{s.source}] {s.content}" for s in snippets)


def make_search_tool(retriever: InMemoryVectorStore, k: int = 4) -> BaseTool:
    @tool(response_format="content_and_artifact")
    def search_knowledge_base(query: str) -> tuple[str, list[RetrievedSnippet]]:
        """Search the KnowledgeBase for policy content relevant to the query."""
        snippets = retrieve_snippets(retriever, query, k=k)
        return format_snippets(snippets), snippets

    return search_knowledge_base
