import operator
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class RetrievedClause(BaseModel):
    content: str
    source: str
    score: float


class Citation(BaseModel):
    source: str
    excerpt: str
    score: float | None = None


class Answer(BaseModel):
    text: str
    citations: list[Citation]
    confidence: Literal["answered", "uncertain", "not_found"]
    considered: list[RetrievedClause] = Field(default_factory=list)


class Exchange(BaseModel):
    question: str
    answer: Answer


class TaxQAState(BaseModel):
    question: str
    answer: Answer | None = None
    history: Annotated[list[Exchange], operator.add] = Field(default_factory=list)
