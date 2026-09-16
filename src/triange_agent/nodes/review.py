from typing import Literal

from langgraph.types import interrupt
from pydantic import BaseModel

from triange_agent.state import Category, TriageState


class ReviewRequest(BaseModel):
    category: Category | None
    answer: str
    escalation_reason: str | None
    attempt_number: int


class ReviewResponse(BaseModel):
    decision: Literal["approved", "rejected"]
    feedback: str | None = None


def _build_review_request(state: TriageState) -> ReviewRequest:
    latest_attempt = state.history[-1]
    return ReviewRequest(
        category=state.category,
        answer=latest_attempt.answer,
        escalation_reason=state.escalation_reason,
        attempt_number=latest_attempt.attempt_number,
    )


def _apply_review_response(state: TriageState, response: ReviewResponse) -> dict:
    retry_count = state.retry_count
    if response.decision == "rejected":
        retry_count += 1

    return {
        "review_decision": response.decision,
        "review_feedback": response.feedback,
        "retry_count": retry_count,
    }


def review(state: TriageState) -> dict:
    request = _build_review_request(state)
    response = interrupt(request)
    return _apply_review_response(state, response)
