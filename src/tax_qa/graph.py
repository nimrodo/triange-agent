from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from tax_qa.nodes.answer import AnswerLLM, answer
from tax_qa.state import TaxQAState
from tax_qa.tools import make_search_tool


def build_graph(
    llm: BaseChatModel,
    retriever: VectorStore,
    checkpointer: BaseCheckpointSaver | None = None,
    search_k: int = 4,
) -> CompiledStateGraph:
    search_tool = make_search_tool(retriever, k=search_k)
    answer_llm = cast(AnswerLLM, llm)

    graph = StateGraph(TaxQAState)
    graph.add_node("answer", lambda state: answer(state, answer_llm, search_tool))
    graph.set_entry_point("answer")
    graph.add_edge("answer", END)

    return graph.compile(checkpointer=checkpointer)
