from pathlib import Path

from langchain_ollama import OllamaEmbeddings

from tax_qa.tools import (
    build_vector_store,
    extract_clauses,
    make_search_tool,
    retrieve_clauses,
)

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "ordinance_excerpt.pdf"
FIXTURE_PAGE_RANGE = range(4)


def test_extract_clauses_returns_one_document_per_real_clause() -> None:
    clauses = extract_clauses(FIXTURE_PDF, FIXTURE_PAGE_RANGE)

    sources = {clause.metadata["source"] for clause in clauses}
    assert "סעיף 121" in sources


def test_extract_clauses_excludes_repealed_clauses() -> None:
    clauses = extract_clauses(FIXTURE_PDF, FIXTURE_PAGE_RANGE)

    assert not any("(בוטל)" in clause.page_content[:40] for clause in clauses)


def test_extract_clauses_content_is_the_real_clause_text() -> None:
    clauses = extract_clauses(FIXTURE_PDF, FIXTURE_PAGE_RANGE)

    clause_121 = next(c for c in clauses if c.metadata["source"] == "סעיף 121")
    assert "המס על הכנסתו החייבת של יחיד" in clause_121.page_content


def test_retrieve_clauses_returns_relevant_clause_for_real_hebrew_query() -> None:
    embeddings = OllamaEmbeddings(model="bge-m3")
    vector_store = build_vector_store(FIXTURE_PDF, embeddings, FIXTURE_PAGE_RANGE)

    clauses = retrieve_clauses(
        vector_store, "מה שיעורי המס החלים על הכנסתו החייבת של יחיד?"
    )

    assert clauses
    assert any(c.source == "סעיף 121" for c in clauses)
    assert any("המס על הכנסתו החייבת של יחיד" in c.content for c in clauses)
    assert all(isinstance(c.score, float) for c in clauses)


def test_make_search_tool_returns_content_and_clause_artifact_for_real_query() -> None:
    embeddings = OllamaEmbeddings(model="bge-m3")
    vector_store = build_vector_store(FIXTURE_PDF, embeddings, FIXTURE_PAGE_RANGE)
    search_tool = make_search_tool(vector_store)

    tool_message = search_tool.invoke(
        {
            "name": "search_tax_ordinance",
            "args": {"query": "מה שיעורי המס החלים על הכנסתו החייבת של יחיד?"},
            "id": "call_1",
            "type": "tool_call",
        }
    )

    assert isinstance(tool_message.content, str)
    assert tool_message.content
    assert tool_message.artifact
    assert any(c.source == "סעיף 121" for c in tool_message.artifact)
