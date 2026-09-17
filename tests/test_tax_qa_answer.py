from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_ollama import OllamaEmbeddings

from tax_qa.nodes.answer import AnswerDraft, answer
from tax_qa.state import TaxQAState
from tax_qa.tools import build_vector_store, make_search_tool

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "ordinance_excerpt.pdf"
FIXTURE_PAGE_RANGE = range(4)
RELEVANT_QUESTION = "מה שיעורי המס החלים על הכנסתו החייבת של יחיד?"
IRRELEVANT_QUESTION = "מהי בירת צרפת?"


class FakeToolCallingLLM:
    def __init__(self, response: AIMessage) -> None:
        self._response = response

    def invoke(self, _messages: list) -> AIMessage:
        return self._response


class FakeStructuredLLM:
    def __init__(self, result: AnswerDraft) -> None:
        self._result = result
        self.invoked = False

    def invoke(self, _input: str) -> AnswerDraft:
        self.invoked = True
        return self._result


class FakeLLM:
    def __init__(self, tool_response: AIMessage, draft: AnswerDraft) -> None:
        self._tool_response = tool_response
        self.structured_llm = FakeStructuredLLM(draft)

    def bind_tools(self, _tools: list, /, *, tool_choice: str) -> FakeToolCallingLLM:
        assert tool_choice == "auto"
        return FakeToolCallingLLM(self._tool_response)

    def with_structured_output(self, _schema: type[AnswerDraft]) -> FakeStructuredLLM:
        return self.structured_llm


def _search_tool():
    embeddings = OllamaEmbeddings(model="bge-m3")
    vector_store = build_vector_store(FIXTURE_PDF, embeddings, FIXTURE_PAGE_RANGE)
    return make_search_tool(vector_store)


def _tool_call_response(query: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_tax_ordinance",
                "args": {"query": query},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )


def test_answer_returns_answered_draft_with_citations_when_floor_is_cleared() -> None:
    state = TaxQAState(question=RELEVANT_QUESTION)
    draft = AnswerDraft(text="שיעור המס נע בין 10% ל-50%.", confidence="answered")
    fake_llm = FakeLLM(_tool_call_response(RELEVANT_QUESTION), draft)

    result = answer(state, fake_llm, _search_tool())

    assert fake_llm.structured_llm.invoked is True
    assert result["answer"].confidence == "answered"
    assert result["answer"].text == draft.text
    assert result["answer"].citations
    assert any(c.source == "סעיף 121" for c in result["answer"].citations)
    assert len(result["history"]) == 1
    assert result["history"][0].question == RELEVANT_QUESTION
    assert result["history"][0].answer is result["answer"]


def test_answer_returns_uncertain_draft_when_llm_self_rates_uncertain() -> None:
    state = TaxQAState(question=RELEVANT_QUESTION)
    draft = AnswerDraft(text="ייתכן ש-10% עד 50%.", confidence="uncertain")
    fake_llm = FakeLLM(_tool_call_response(RELEVANT_QUESTION), draft)

    result = answer(state, fake_llm, _search_tool())

    assert result["answer"].confidence == "uncertain"
    assert result["answer"].citations


def test_answer_forces_not_found_without_calling_llm_for_a_draft() -> None:
    state = TaxQAState(question=IRRELEVANT_QUESTION)
    draft = AnswerDraft(text="should never be used", confidence="answered")
    fake_llm = FakeLLM(_tool_call_response(IRRELEVANT_QUESTION), draft)

    result = answer(state, fake_llm, _search_tool())

    assert fake_llm.structured_llm.invoked is False
    assert result["answer"].confidence == "not_found"
    assert result["answer"].text == ""
    assert result["answer"].citations == []
    assert len(result["history"]) == 1


def test_answer_treats_similarity_floor_as_a_hard_override() -> None:
    state = TaxQAState(question=RELEVANT_QUESTION)
    draft = AnswerDraft(text="should never be used", confidence="answered")
    fake_llm = FakeLLM(_tool_call_response(RELEVANT_QUESTION), draft)

    result = answer(state, fake_llm, _search_tool(), similarity_floor=0.99)

    assert fake_llm.structured_llm.invoked is False
    assert result["answer"].confidence == "not_found"
