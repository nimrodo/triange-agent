import operator
from typing import Annotated, Literal

from pydantic import BaseModel, Field

Category = Literal["billing", "technical", "account", "general"]


class Ticket(BaseModel):
    subject: str
    body: str


class AttemptRecord(BaseModel):
    answer: str
    reviewer_feedback: str | None = None
    attempt_number: int


class RetrievedSnippet(BaseModel):
    content: str
    source: str


class TriageState(BaseModel):
    ticket: Ticket
    category: Category | None = None
    retrieved_snippets: list[RetrievedSnippet] = Field(default_factory=list)
    history: Annotated[list[AttemptRecord], operator.add] = Field(default_factory=list)
    needs_escalation: bool = False
    escalation_reason: str | None = None
    review_decision: Literal["approved", "rejected"] | None = None
    review_feedback: str | None = None
    retry_count: int = 0
