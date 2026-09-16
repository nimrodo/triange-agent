from typing import Literal, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from triange_agent.nodes.answer import AnswerLLM, answer
from triange_agent.nodes.classify import ClassifierLLM, classify
from triange_agent.nodes.review import review
from triange_agent.state import TriageState
from triange_agent.tools import make_search_tool

MAX_RETRIES = 3


def _route_after_answer(state: TriageState) -> Literal["review", "__end__"]:
    if state.needs_escalation:
        return "review"
    return "__end__"


def _route_after_review(state: TriageState) -> Literal["answer", "__end__"]:
    if state.review_decision == "rejected" and state.retry_count <= MAX_RETRIES:
        return "answer"
    return "__end__"


def build_graph(
    llm: BaseChatModel, retriever: InMemoryVectorStore
) -> CompiledStateGraph:
    search_tool = make_search_tool(retriever)
    classifier_llm = cast(ClassifierLLM, llm)
    answer_llm = cast(AnswerLLM, llm)

    graph = StateGraph(TriageState)
    graph.add_node("classify", lambda state: classify(state, classifier_llm))
    graph.add_node("answer", lambda state: answer(state, answer_llm, search_tool))
    graph.add_node("review", review)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "answer")
    graph.add_conditional_edges(
        "answer", _route_after_answer, {"review": "review", END: END}
    )
    graph.add_conditional_edges(
        "review", _route_after_review, {"answer": "answer", END: END}
    )

    return graph.compile()
