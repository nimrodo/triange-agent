import operator

import pytest
from pydantic import ValidationError

from triange_agent.state import AttemptRecord, RetrievedSnippet, Ticket, TriageState


def test_triage_state_requires_only_ticket() -> None:
    state = TriageState(
        ticket=Ticket(subject="Can't log in", body="Password reset loop")
    )

    assert state.ticket.subject == "Can't log in"
    assert state.category is None
    assert state.retrieved_snippets == []
    assert state.history == []
    assert state.needs_escalation is False
    assert state.escalation_reason is None
    assert state.review_decision is None
    assert state.review_feedback is None
    assert state.retry_count == 0


def test_triage_state_requires_ticket_field() -> None:
    with pytest.raises(ValidationError):
        TriageState()  # ty: ignore[missing-argument]


def test_category_rejects_values_outside_fixed_set() -> None:
    with pytest.raises(ValidationError):
        TriageState(
            ticket=Ticket(subject="s", body="b"),
            category="not-a-real-category",  # ty: ignore[invalid-argument-type]
        )


def test_category_accepts_fixed_set_value() -> None:
    state = TriageState(
        ticket=Ticket(subject="s", body="b"),
        category="billing",
    )

    assert state.category == "billing"


def test_history_accumulates_full_construction() -> None:
    attempt = AttemptRecord(
        answer="Here's how to reset your password.", attempt_number=1
    )

    state = TriageState(
        ticket=Ticket(subject="s", body="b"),
        history=[attempt],
    )

    assert state.history == [attempt]
    assert state.history[0].reviewer_feedback is None


def test_ticket_requires_subject_and_body() -> None:
    with pytest.raises(ValidationError):
        Ticket(subject="s")  # ty: ignore[missing-argument]


def test_attempt_record_requires_answer_and_attempt_number() -> None:
    with pytest.raises(ValidationError):
        AttemptRecord(answer="a")  # ty: ignore[missing-argument]


def test_history_reducer_appends_via_operator_add() -> None:
    existing = [AttemptRecord(answer="a", attempt_number=1)]
    delta = [AttemptRecord(answer="b", attempt_number=2)]

    merged = operator.add(existing, delta)

    assert merged == existing + delta


def test_retrieved_snippet_round_trips() -> None:
    snippet = RetrievedSnippet(
        content="Refunds are processed within 5 business days.", source="billing.md"
    )

    state = TriageState(
        ticket=Ticket(subject="s", body="b"),
        retrieved_snippets=[snippet],
    )

    assert state.retrieved_snippets[0].source == "billing.md"
    assert state.retrieved_snippets[0].content.startswith("Refunds")


def test_thread_id_is_not_a_state_field() -> None:
    assert "thread_id" not in TriageState.model_fields
