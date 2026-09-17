from pathlib import Path
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import OllamaEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from tax_qa.graph import build_graph
from tax_qa.nodes.answer import AnswerDraft
from tax_qa.tools import build_vector_store

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "ordinance_excerpt.pdf"
FIXTURE_PAGE_RANGE = range(4)
RELEVANT_QUESTION = "מה שיעורי המס החלים על הכנסתו החייבת של יחיד?"
IRRELEVANT_QUESTION = "מהי בירת צרפת?"


class FakeToolCallingLLM:
    def __init__(self, query: str) -> None:
        self._query = query

    def invoke(self, _messages: list) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_tax_ordinance",
                    "args": {"query": self._query},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        )


class FakeQueuedStructuredLLM:
    def __init__(self, results: list[AnswerDraft]) -> None:
        self._results = iter(results)

    def invoke(self, _input: str) -> AnswerDraft:
        return next(self._results)


class FakeLLM:
    """Serves one scripted (search query, AnswerDraft) pair per turn,
    mirroring the triage agent's queued-fake-LLM pattern."""

    def __init__(self, turns: list[tuple[str, AnswerDraft]]) -> None:
        self._queries = iter(query for query, _ in turns)
        self._drafts = FakeQueuedStructuredLLM([draft for _, draft in turns])

    def bind_tools(self, _tools: list, /, *, tool_choice: str) -> FakeToolCallingLLM:
        assert tool_choice == "auto"
        return FakeToolCallingLLM(next(self._queries))

    def with_structured_output(self, _schema: type, /) -> FakeQueuedStructuredLLM:
        return self._drafts


def _build_test_graph(
    llm: FakeLLM, thread_id: str
) -> tuple[CompiledStateGraph, RunnableConfig]:
    embeddings = OllamaEmbeddings(model="bge-m3")
    retriever = build_vector_store(FIXTURE_PDF, embeddings, FIXTURE_PAGE_RANGE)
    graph = build_graph(cast(BaseChatModel, llm), retriever, InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    return graph, config


def test_history_accumulates_across_multiple_turns_on_the_same_thread() -> None:
    llm = FakeLLM(
        [
            (
                RELEVANT_QUESTION,
                AnswerDraft(text="שיעור המס נע בין 10% ל-50%.", confidence="answered"),
            ),
            (
                RELEVANT_QUESTION,
                AnswerDraft(text="ייתכן ש-10% עד 50%.", confidence="uncertain"),
            ),
        ]
    )
    graph, config = _build_test_graph(llm, thread_id="multi-turn")

    graph.invoke({"question": RELEVANT_QUESTION}, config)
    graph.invoke({"question": RELEVANT_QUESTION}, config)

    final_state = graph.get_state(config).values
    assert len(final_state["history"]) == 2
    assert final_state["history"][0].answer.confidence == "answered"
    assert final_state["history"][1].answer.confidence == "uncertain"
    assert final_state["answer"].confidence == "uncertain"


def test_not_found_confidence_state_is_returned_end_to_end() -> None:
    llm = FakeLLM(
        [(IRRELEVANT_QUESTION, AnswerDraft(text="unused", confidence="answered"))]
    )
    graph, config = _build_test_graph(llm, thread_id="not-found")

    graph.invoke({"question": IRRELEVANT_QUESTION}, config)

    final_state = graph.get_state(config).values
    assert final_state["answer"].confidence == "not_found"
    assert final_state["answer"].citations == []
