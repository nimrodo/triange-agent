from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import OllamaEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from triange_agent.graph import MAX_RETRIES, build_graph
from triange_agent.nodes.answer import AnswerDraft
from triange_agent.nodes.classify import CategoryAssignment
from triange_agent.nodes.review import ReviewResponse
from triange_agent.state import Ticket, TriageState
from triange_agent.tools import build_vector_store

TICKET = Ticket(
    subject="Refund request outside the stated window",
    body="I was charged for a subscription I cancelled two months ago.",
)


class FakeToolCallingLLM:
    def invoke(self, _messages: list) -> AIMessage:
        return AIMessage(content="", tool_calls=[])


class FakeQueuedStructuredLLM:
    def __init__(self, results: list) -> None:
        self._results = iter(results)

    def invoke(self, _input: str):
        return next(self._results)


class FakeLLM:
    """Serves classify's CategoryAssignment calls and answer's AnswerDraft calls
    from pre-scripted queues, since build_graph casts one LLM into both roles."""

    def __init__(
        self, categories: list[CategoryAssignment], drafts: list[AnswerDraft]
    ) -> None:
        self._categories = FakeQueuedStructuredLLM(categories)
        self._drafts = FakeQueuedStructuredLLM(drafts)

    def bind_tools(self, _tools: list, /, *, tool_choice: str) -> FakeToolCallingLLM:
        assert tool_choice == "auto"
        return FakeToolCallingLLM()

    def with_structured_output(self, schema: type, /) -> FakeQueuedStructuredLLM:
        if schema is CategoryAssignment:
            return self._categories
        return self._drafts


def _build_test_graph(
    llm: FakeLLM, thread_id: str
) -> tuple[CompiledStateGraph, RunnableConfig]:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    retriever = build_vector_store(embeddings)
    graph = build_graph(cast(BaseChatModel, llm), retriever, InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    return graph, config


def _drive(
    graph: CompiledStateGraph,
    config: RunnableConfig,
    responses: list[ReviewResponse],
) -> int:
    """Streams the graph to completion, resuming with one scripted review
    response per interrupt. Returns how many interrupts fired."""
    stream_input: TriageState | Command = TriageState(ticket=TICKET)
    responses_iter = iter(responses)
    interrupt_count = 0
    while True:
        interrupted = False
        for chunk in graph.stream(stream_input, config):
            if "__interrupt__" in chunk:
                interrupted = True
                interrupt_count += 1
        if not interrupted:
            return interrupt_count
        stream_input = Command(resume=next(responses_iter))


def test_happy_path_resolves_without_escalation_and_never_interrupts() -> None:
    llm = FakeLLM(
        categories=[CategoryAssignment(category="billing")],
        drafts=[AnswerDraft(answer="You have 30 days.", needs_escalation=False)],
    )
    graph, config = _build_test_graph(llm, thread_id="happy-path")

    interrupt_count = _drive(graph, config, responses=[])

    assert interrupt_count == 0
    final_state = graph.get_state(config).values
    assert final_state["category"] == "billing"
    assert len(final_state["history"]) == 1
    assert final_state["history"][0].answer == "You have 30 days."
    assert final_state.get("review_decision") is None


def test_escalate_then_approve_interrupts_once_and_ends_the_run() -> None:
    llm = FakeLLM(
        categories=[CategoryAssignment(category="billing")],
        drafts=[
            AnswerDraft(
                answer="Let me check with a manager.",
                needs_escalation=True,
                escalation_reason="Refund is outside the policy window.",
            )
        ],
    )
    graph, config = _build_test_graph(llm, thread_id="escalate-approve")

    interrupt_count = _drive(
        graph, config, responses=[ReviewResponse(decision="approved")]
    )

    assert interrupt_count == 1
    final_state = graph.get_state(config).values
    assert final_state["review_decision"] == "approved"
    assert final_state["retry_count"] == 0
    assert len(final_state["history"]) == 1


def test_escalate_then_reject_repeatedly_terminates_at_the_retry_cap() -> None:
    attempts = MAX_RETRIES + 1
    llm = FakeLLM(
        categories=[CategoryAssignment(category="billing")],
        drafts=[
            AnswerDraft(
                answer=f"Draft attempt {n}.",
                needs_escalation=True,
                escalation_reason="Refund is outside the policy window.",
            )
            for n in range(1, attempts + 1)
        ],
    )
    graph, config = _build_test_graph(llm, thread_id="escalate-reject")
    responses = [
        ReviewResponse(decision="rejected", feedback=f"Try again #{n}.")
        for n in range(1, attempts + 1)
    ]

    interrupt_count = _drive(graph, config, responses=responses)

    assert interrupt_count == attempts
    final_state = graph.get_state(config).values
    assert final_state["retry_count"] == attempts
    assert final_state["review_decision"] == "rejected"
    assert len(final_state["history"]) == attempts
    assert [attempt.answer for attempt in final_state["history"]] == [
        f"Draft attempt {n}." for n in range(1, attempts + 1)
    ]
    assert [attempt.reviewer_feedback for attempt in final_state["history"]] == [
        None,
        *(response.feedback for response in responses[:-1]),
    ]
