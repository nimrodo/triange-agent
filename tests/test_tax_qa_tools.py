from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from tax_qa.tools import (
    MAX_CLAUSE_CHARS,
    build_vector_store,
    extract_clauses,
    make_search_tool,
    retrieve_clauses,
    split_oversized_clauses,
)

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "ordinance_excerpt.pdf"
FIXTURE_PAGE_RANGE = range(4)


class BatchLimitedFakeEmbeddings(Embeddings):
    """Rejects any single embed_documents call larger than max_batch --
    mirrors the local Ollama server refusing oversized batch requests
    against the full real ordinance (500+ Clauses)."""

    def __init__(self, max_batch: int) -> None:
        self.max_batch = max_batch
        self.batch_sizes: list[int] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if len(texts) > self.max_batch:
            raise ValueError(f"batch of {len(texts)} exceeds max {self.max_batch}")
        self.batch_sizes.append(len(texts))
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0]


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


def test_extract_clauses_preserves_the_containing_part_structure() -> None:
    clauses = extract_clauses(FIXTURE_PDF, FIXTURE_PAGE_RANGE)

    clause_121 = next(c for c in clauses if c.metadata["source"] == "סעיף 121")
    assert clause_121.metadata["part"] == "חלק ז': שיעורי המס"


def test_retrieve_clauses_returns_relevant_clause_for_real_hebrew_query(
    tmp_path: Path,
) -> None:
    embeddings = OllamaEmbeddings(model="bge-m3")
    vector_store = build_vector_store(
        FIXTURE_PDF, embeddings, FIXTURE_PAGE_RANGE, persist_directory=tmp_path
    )

    clauses = retrieve_clauses(
        vector_store, "מה שיעורי המס החלים על הכנסתו החייבת של יחיד?"
    )

    assert clauses
    assert any(c.source == "סעיף 121" for c in clauses)
    assert any("המס על הכנסתו החייבת של יחיד" in c.content for c in clauses)
    assert all(isinstance(c.score, float) for c in clauses)


def test_split_oversized_clauses_leaves_small_clauses_untouched() -> None:
    small = Document(page_content="קצר", metadata={"source": "סעיף 1"})

    result = split_oversized_clauses([small])

    assert result == [small]


def test_split_oversized_clauses_splits_a_clause_over_the_size_cap() -> None:
    huge = Document(
        page_content="א" * (MAX_CLAUSE_CHARS * 3), metadata={"source": "סעיף 9"}
    )

    result = split_oversized_clauses([huge])

    assert len(result) > 1
    assert all(len(chunk.page_content) <= MAX_CLAUSE_CHARS for chunk in result)
    assert all(chunk.metadata["source"] == "סעיף 9" for chunk in result)


def test_build_vector_store_embeds_documents_in_batches(tmp_path: Path) -> None:
    embeddings = BatchLimitedFakeEmbeddings(max_batch=1)

    vector_store = build_vector_store(
        FIXTURE_PDF,
        embeddings,
        FIXTURE_PAGE_RANGE,
        batch_size=1,
        persist_directory=tmp_path,
    )

    clauses = extract_clauses(FIXTURE_PDF, FIXTURE_PAGE_RANGE)
    assert len(embeddings.batch_sizes) == len(clauses)
    assert all(size <= 1 for size in embeddings.batch_sizes)
    assert len(vector_store.get(include=[])["ids"]) == len(clauses)


class FlakyEmbeddings(Embeddings):
    """Raises once a fixed number of documents have been embedded -- mirrors a
    provider rate limit killing a build partway through."""

    def __init__(self, fail_after: int) -> None:
        self.fail_after = fail_after
        self.embedded = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.embedded >= self.fail_after:
            raise ValueError("rate limited")
        self.embedded += len(texts)
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0]


def test_build_vector_store_resumes_after_a_partial_build(tmp_path: Path) -> None:
    clauses = extract_clauses(FIXTURE_PDF, FIXTURE_PAGE_RANGE)
    fail_after = len(clauses) // 2

    try:
        build_vector_store(
            FIXTURE_PDF,
            FlakyEmbeddings(fail_after=fail_after),
            FIXTURE_PAGE_RANGE,
            batch_size=1,
            persist_directory=tmp_path,
        )
    except ValueError:
        pass

    resuming_embeddings = BatchLimitedFakeEmbeddings(max_batch=1)
    vector_store = build_vector_store(
        FIXTURE_PDF,
        resuming_embeddings,
        FIXTURE_PAGE_RANGE,
        batch_size=1,
        persist_directory=tmp_path,
    )

    assert len(resuming_embeddings.batch_sizes) == len(clauses) - fail_after
    assert len(vector_store.get(include=[])["ids"]) == len(clauses)


def test_make_search_tool_returns_content_and_clause_artifact_for_real_query(
    tmp_path: Path,
) -> None:
    embeddings = OllamaEmbeddings(model="bge-m3")
    vector_store = build_vector_store(
        FIXTURE_PDF, embeddings, FIXTURE_PAGE_RANGE, persist_directory=tmp_path
    )
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
