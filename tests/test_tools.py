from langchain_ollama import OllamaEmbeddings

from triange_agent.tools import (
    build_vector_store,
    load_knowledge_base,
    make_search_tool,
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


def test_make_search_tool_returns_content_and_snippet_artifact_for_real_query() -> None:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = build_vector_store(embeddings)
    search_tool = make_search_tool(vector_store)

    tool_message = search_tool.invoke(
        {
            "name": "search_knowledge_base",
            "args": {"query": "How long do I have to request a refund?"},
            "id": "call_1",
            "type": "tool_call",
        }
    )

    assert isinstance(tool_message.content, str)
    assert tool_message.content
    assert tool_message.artifact
    assert any(snippet.source == "billing.md" for snippet in tool_message.artifact)
    assert any("refund" in snippet.content.lower() for snippet in tool_message.artifact)
