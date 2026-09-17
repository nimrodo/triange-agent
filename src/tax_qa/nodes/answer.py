from typing import Literal, Protocol

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from tax_qa.state import Answer, Citation, Exchange, RetrievedClause, TaxQAState
from tax_qa.tools import format_clauses

DEFAULT_SIMILARITY_FLOOR = 0.5


class AnswerDraft(BaseModel):
    text: str
    confidence: Literal["answered", "uncertain"]


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


def answer(
    state: TaxQAState,
    llm: AnswerLLM,
    search_tool: BaseTool,
    similarity_floor: float = DEFAULT_SIMILARITY_FLOOR,
) -> dict:
    tool_calling_llm = llm.bind_tools([search_tool], tool_choice="auto")
    response = tool_calling_llm.invoke([HumanMessage(state.question)])

    retrieved: list[RetrievedClause] = []
    for tool_call in response.tool_calls:
        tool_message = search_tool.invoke(tool_call)
        retrieved.extend(tool_message.artifact or [])

    qualifying = [clause for clause in retrieved if clause.score >= similarity_floor]

    if not qualifying:
        result = Answer(text="", citations=[], confidence="not_found")
    else:
        prompt = f"{state.question}\n\nRetrieved clauses:\n{format_clauses(qualifying)}"
        draft = llm.with_structured_output(AnswerDraft).invoke(prompt)
        result = Answer(
            text=draft.text,
            citations=[
                Citation(source=clause.source, excerpt=clause.content)
                for clause in qualifying
            ],
            confidence=draft.confidence,
        )

    exchange = Exchange(question=state.question, answer=result)
    return {"answer": result, "history": [exchange]}
