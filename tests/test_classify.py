from triange_agent.nodes.classify import CategoryAssignment, classify
from triange_agent.state import Ticket, TriageState


class FakeStructuredLLM:
    def __init__(self, result: CategoryAssignment) -> None:
        self._result = result

    def invoke(self, _input: str) -> CategoryAssignment:
        return self._result


class FakeLLM:
    def __init__(self, result: CategoryAssignment) -> None:
        self._result = result

    def with_structured_output(
        self, _schema: type[CategoryAssignment]
    ) -> FakeStructuredLLM:
        return FakeStructuredLLM(self._result)


def test_classify_assigns_category_from_llm_output() -> None:
    state = TriageState(
        ticket=Ticket(subject="Charged twice", body="I was billed twice this month")
    )
    fake_llm = FakeLLM(CategoryAssignment(category="billing"))

    result = classify(state, fake_llm)

    assert result == {"category": "billing"}
