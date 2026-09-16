from langchain_ollama import OllamaEmbeddings

from triange_agent.tools import (
    build_vector_store,
    load_knowledge_base,
    retrieve_snippets,
)


def test_load_knowledge_base_returns_one_document_per_category_doc() -> None:
    documents = load_knowledge_base()

    sources = {doc.metadata["source"] for doc in documents}
    assert sources == {"account.md", "billing.md", "general.md", "technical.md"}


def test_retrieve_snippets_returns_relevant_chunks_for_real_query() -> None:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = build_vector_store(embeddings)

    snippets = retrieve_snippets(
        vector_store, "How long do I have to request a refund?"
    )

    assert snippets
    assert any(snippet.source == "billing.md" for snippet in snippets)
    assert any("refund" in snippet.content.lower() for snippet in snippets)
