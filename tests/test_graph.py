from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.vectorstores import InMemoryVectorStore

from triange_agent.graph import (
    MAX_RETRIES,
    _route_after_answer,
    _route_after_review,
    build_graph,
)
from triange_agent.state import AttemptRecord, Ticket, TriageState


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0]


def _build_graph():
    llm = GenericFakeChatModel(messages=iter([]))
    retriever = InMemoryVectorStore(embedding=FakeEmbeddings())
    return build_graph(llm, retriever)


def test_build_graph_wires_expected_nodes() -> None:
    graph = _build_graph()

    assert set(graph.get_graph().nodes) >= {
        "__start__",
        "classify",
        "answer",
        "review",
    }


def test_build_graph_routes_classify_to_answer() -> None:
    graph = _build_graph()

    edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}

    assert ("classify", "answer") in edges


def test_build_graph_routes_answer_conditionally_to_review_or_end() -> None:
    graph = _build_graph()

    edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}

    assert ("answer", "review") in edges
    assert ("answer", "__end__") in edges


def test_build_graph_routes_review_conditionally_back_to_answer_or_end() -> None:
    graph = _build_graph()

    edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}

    assert ("review", "answer") in edges
    assert ("review", "__end__") in edges


def _state(**overrides: object) -> TriageState:
    base = TriageState(
        ticket=Ticket(subject="Refund window", body="How long is the window?"),
        history=[AttemptRecord(answer="30 days.", attempt_number=1)],
    )
    return base.model_copy(update=overrides)


def test_route_after_answer_goes_to_review_when_escalation_needed() -> None:
    assert _route_after_answer(_state(needs_escalation=True)) == "review"


def test_route_after_answer_ends_when_no_escalation_needed() -> None:
    assert _route_after_answer(_state(needs_escalation=False)) == "__end__"


def test_route_after_review_ends_on_approval() -> None:
    state = _state(review_decision="approved", retry_count=0)

    assert _route_after_review(state) == "__end__"


def test_route_after_review_loops_back_for_every_retry_up_to_the_cap() -> None:
    for retry_count in range(1, MAX_RETRIES + 1):
        state = _state(review_decision="rejected", retry_count=retry_count)

        assert _route_after_review(state) == "answer"


def test_route_after_review_ends_once_the_retry_cap_is_exceeded() -> None:
    state = _state(review_decision="rejected", retry_count=MAX_RETRIES + 1)

    assert _route_after_review(state) == "__end__"
