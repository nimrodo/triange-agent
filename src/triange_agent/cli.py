from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Interrupt

from triange_agent.dependencies import build_llm, build_retriever
from triange_agent.graph import build_graph
from triange_agent.nodes.review import ReviewRequest, ReviewResponse
from triange_agent.state import Ticket, TriageState

THREAD_ID = "cli-session"

SAMPLE_TICKET = Ticket(
    subject="Refund request outside the stated window",
    body=(
        "I was charged for a subscription I cancelled two months ago and "
        "the refund policy page doesn't cover this case. Can you refund me?"
    ),
)


def _prompt_for_review(request: ReviewRequest) -> ReviewResponse:
    print("\n--- Human review requested ---")
    print(f"Category: {request.category}")
    print(f"Attempt #{request.attempt_number}")
    print(f"Escalation reason: {request.escalation_reason}")
    print(f"Draft answer:\n{request.answer}\n")

    decision = ""
    while decision not in {"approve", "reject"}:
        decision = input("Approve or reject? [approve/reject]: ").strip().lower()

    if decision == "approve":
        return ReviewResponse(decision="approved")

    feedback = input("Feedback for the next attempt: ").strip()
    return ReviewResponse(decision="rejected", feedback=feedback or None)


def run(ticket: Ticket) -> None:
    graph = build_graph(build_llm(), build_retriever(), InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": THREAD_ID}}

    stream_input: TriageState | Command = TriageState(ticket=ticket)
    while True:
        interrupt: Interrupt | None = None
        for chunk in graph.stream(stream_input, config):
            if "__interrupt__" in chunk:
                interrupt = chunk["__interrupt__"][0]
            else:
                print(chunk)

        if interrupt is None:
            break

        response = _prompt_for_review(interrupt.value)
        stream_input = Command(resume=response)

    final_state = graph.get_state(config).values
    print("\n--- Triage result ---")
    print(f"Category: {final_state['category']}")
    print(f"Final answer: {final_state['history'][-1].answer}")


def main() -> None:
    run(SAMPLE_TICKET)


if __name__ == "__main__":
    main()
