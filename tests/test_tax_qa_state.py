import operator

import pytest
from pydantic import ValidationError

from tax_qa.state import Answer, Citation, Exchange, RetrievedClause, TaxQAState


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


def test_citation_round_trips() -> None:
    citation = Citation(source="סעיף 121", excerpt="המס על הכנסתו החייבת של יחיד")

    assert citation.source == "סעיף 121"
    assert citation.excerpt.startswith("המס")


def test_answer_confidence_rejects_values_outside_fixed_set() -> None:
    with pytest.raises(ValidationError):
        Answer(
            text="t",
            citations=[],
            confidence="maybe",  # ty: ignore[invalid-argument-type]
        )


def test_answer_accepts_each_confidence_state() -> None:
    for confidence in ("answered", "uncertain", "not_found"):
        answer = Answer(text="t", citations=[], confidence=confidence)
        assert answer.confidence == confidence


def test_exchange_pairs_question_with_answer() -> None:
    answer = Answer(text="30 יום", citations=[], confidence="answered")
    exchange = Exchange(question="מה התקופה?", answer=answer)

    assert exchange.question == "מה התקופה?"
    assert exchange.answer is answer


def test_tax_qa_state_requires_only_question() -> None:
    state = TaxQAState(question="מה שיעור המס?")

    assert state.question == "מה שיעור המס?"
    assert state.answer is None
    assert state.history == []


def test_tax_qa_state_requires_question_field() -> None:
    with pytest.raises(ValidationError):
        TaxQAState()  # ty: ignore[missing-argument]


def test_tax_qa_history_reducer_appends_via_operator_add() -> None:
    answer = Answer(text="t", citations=[], confidence="answered")
    existing = [Exchange(question="q1", answer=answer)]
    delta = [Exchange(question="q2", answer=answer)]

    merged = operator.add(existing, delta)

    assert merged == existing + delta


def test_thread_id_is_not_a_tax_qa_state_field() -> None:
    assert "thread_id" not in TaxQAState.model_fields
