from triange_agent.nodes.review import (
    ReviewRequest,
    ReviewResponse,
    _apply_review_response,
    _build_review_request,
)
from triange_agent.state import AttemptRecord, Ticket, TriageState


def _state_with_attempt(retry_count: int = 0) -> TriageState:
    return TriageState(
        ticket=Ticket(subject="Angry customer", body="I demand a refund now."),
        category="billing",
        needs_escalation=True,
        escalation_reason="Customer is escalating to a manager.",
        history=[AttemptRecord(answer="Here is your refund policy.", attempt_number=1)],
        retry_count=retry_count,
    )


def test_build_review_request_reflects_latest_attempt() -> None:
    state = _state_with_attempt()

    request = _build_review_request(state)

    assert request == ReviewRequest(
        category="billing",
        answer="Here is your refund policy.",
        escalation_reason="Customer is escalating to a manager.",
        attempt_number=1,
    )


def test_apply_review_response_records_approval() -> None:
    state = _state_with_attempt()
    response = ReviewResponse(decision="approved")

    result = _apply_review_response(state, response)

    assert result == {
        "review_decision": "approved",
        "review_feedback": None,
        "retry_count": 0,
    }


def test_apply_review_response_records_rejection_and_increments_retry_count() -> None:
    state = _state_with_attempt(retry_count=1)
    response = ReviewResponse(decision="rejected", feedback="Mention the deadline.")

    result = _apply_review_response(state, response)

    assert result == {
        "review_decision": "rejected",
        "review_feedback": "Mention the deadline.",
        "retry_count": 2,
    }
