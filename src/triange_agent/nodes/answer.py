from typing import Protocol

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from triange_agent.state import AttemptRecord, RetrievedSnippet, TriageState
from triange_agent.tools import format_snippets


class AnswerDraft(BaseModel):
    answer: str
    needs_escalation: bool
    escalation_reason: str | None = None


class ToolCallingRunnable(Protocol):
    def invoke(self, input: list, /) -> AIMessage: ...


class StructuredOutputRunnable(Protocol):
    def invoke(self, input: str, /) -> AnswerDraft: ...


class AnswerLLM(Protocol):
    def bind_tools(
        self, tools: list[BaseTool], /, *, tool_choice: str
    ) -> ToolCallingRunnable: ...

    def with_structured_output(
        self, schema: type[AnswerDraft], /
    ) -> StructuredOutputRunnable: ...


def answer(state: TriageState, llm: AnswerLLM, search_tool: BaseTool) -> dict:
    question = f"Subject: {state.ticket.subject}\n\n{state.ticket.body}"
    messages = [HumanMessage(question)]
    if state.review_feedback:
        messages.append(HumanMessage(f"Reviewer feedback: {state.review_feedback}"))

    tool_calling_llm = llm.bind_tools([search_tool], tool_choice="auto")
    response = tool_calling_llm.invoke(messages)

    snippets: list[RetrievedSnippet] = []
    for tool_call in response.tool_calls:
        tool_message = search_tool.invoke(tool_call)
        snippets.extend(tool_message.artifact or [])

    draft_prompt = question
    if snippets:
        draft_prompt = (
            f"{question}\n\nRetrieved policy context:\n{format_snippets(snippets)}"
        )

    structured_llm = llm.with_structured_output(AnswerDraft)
    draft = structured_llm.invoke(draft_prompt)

    attempt = AttemptRecord(
        answer=draft.answer,
        reviewer_feedback=state.review_feedback,
        attempt_number=len(state.history) + 1,
    )

    return {
        "retrieved_snippets": snippets,
        "needs_escalation": draft.needs_escalation,
        "escalation_reason": draft.escalation_reason,
        "history": [attempt],
    }
