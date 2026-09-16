from langchain_core.messages import AIMessage
from langchain_ollama import OllamaEmbeddings

from triange_agent.nodes.answer import AnswerDraft, answer
from triange_agent.state import Ticket, TriageState
from triange_agent.tools import build_vector_store, make_search_tool


class FakeToolCallingLLM:
    def __init__(self, response: AIMessage) -> None:
        self._response = response

    def invoke(self, _messages: list) -> AIMessage:
        return self._response


class FakeStructuredLLM:
    def __init__(self, result: AnswerDraft) -> None:
        self._result = result

    def invoke(self, _input: str) -> AnswerDraft:
        return self._result


class FakeLLM:
    def __init__(self, tool_response: AIMessage, draft: AnswerDraft) -> None:
        self._tool_response = tool_response
        self._draft = draft

    def bind_tools(self, _tools: list, /, *, tool_choice: str) -> FakeToolCallingLLM:
        assert tool_choice == "auto"
        return FakeToolCallingLLM(self._tool_response)

    def with_structured_output(self, _schema: type[AnswerDraft]) -> FakeStructuredLLM:
        return FakeStructuredLLM(self._draft)


def _search_tool():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = build_vector_store(embeddings)
    return make_search_tool(vector_store)


def test_answer_returns_non_escalation_draft_with_retrieved_snippets() -> None:
    state = TriageState(
        ticket=Ticket(
            subject="Refund window",
            body="How long do I have to request a refund?",
        )
    )
    tool_call_response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_knowledge_base",
                "args": {"query": "refund window"},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )
    draft = AnswerDraft(
        answer="You have 30 days to request a refund.", needs_escalation=False
    )
    fake_llm = FakeLLM(tool_call_response, draft)

    result = answer(state, fake_llm, _search_tool())

    assert result["needs_escalation"] is False
    assert result["escalation_reason"] is None
    assert result["retrieved_snippets"]
    assert any(
        snippet.source == "billing.md" for snippet in result["retrieved_snippets"]
    )
    assert len(result["history"]) == 1
    assert result["history"][0].answer == draft.answer
    assert result["history"][0].attempt_number == 1


def test_answer_returns_escalation_draft_when_llm_flags_it() -> None:
    state = TriageState(
        ticket=Ticket(
            subject="Angry customer", body="This is unacceptable, I demand a manager."
        )
    )
    tool_call_response = AIMessage(content="", tool_calls=[])
    draft = AnswerDraft(
        answer="I understand your frustration.",
        needs_escalation=True,
        escalation_reason="Customer is escalating to a manager.",
    )
    fake_llm = FakeLLM(tool_call_response, draft)

    result = answer(state, fake_llm, _search_tool())

    assert result["needs_escalation"] is True
    assert result["escalation_reason"] == "Customer is escalating to a manager."
    assert result["retrieved_snippets"] == []
