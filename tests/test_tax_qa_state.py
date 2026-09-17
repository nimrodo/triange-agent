import pytest
from pydantic import ValidationError

from tax_qa.state import RetrievedClause


def test_retrieved_clause_round_trips() -> None:
    clause = RetrievedClause(
        content="המס על הכנסתו החייבת של יחיד בשנת המס יהיה כלהלן",
        source="סעיף 121",
        score=0.82,
    )

    assert clause.source == "סעיף 121"
    assert clause.score == 0.82
    assert clause.content.startswith("המס")


def test_retrieved_clause_requires_all_fields() -> None:
    with pytest.raises(ValidationError):
        RetrievedClause(content="c", source="s")  # ty: ignore[missing-argument]
