from typing import Protocol

from pydantic import BaseModel

from triange_agent.state import Category, TriageState


class CategoryAssignment(BaseModel):
    category: Category


class StructuredOutputRunnable(Protocol):
    def invoke(self, input: str, /) -> CategoryAssignment: ...


class ClassifierLLM(Protocol):
    def with_structured_output(
        self, schema: type[CategoryAssignment], /
    ) -> StructuredOutputRunnable: ...


def classify(state: TriageState, llm: ClassifierLLM) -> dict:
    structured_llm = llm.with_structured_output(CategoryAssignment)
    assignment = structured_llm.invoke(
        f"Subject: {state.ticket.subject}\n\n{state.ticket.body}"
    )
    return {"category": assignment.category}
