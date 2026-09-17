from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.vectorstores import InMemoryVectorStore

from tax_qa.graph import build_graph


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0]


def _build_graph():
    llm = GenericFakeChatModel(messages=iter([]))
    retriever = InMemoryVectorStore(embedding=FakeEmbeddings())
    return build_graph(llm, retriever)


def test_build_graph_wires_the_single_answer_node() -> None:
    graph = _build_graph()

    assert set(graph.get_graph().nodes) >= {"__start__", "answer"}


def test_build_graph_routes_start_to_answer_to_end() -> None:
    graph = _build_graph()

    edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}

    assert ("__start__", "answer") in edges
    assert ("answer", "__end__") in edges
