from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from tax_qa.dependencies import Settings, build_llm, build_retriever
from tax_qa.formatting import percent
from tax_qa.graph import build_graph
from tax_qa.nodes.answer import DEFAULT_SIMILARITY_FLOOR
from tax_qa.state import Answer, Citation, RetrievedClause

THREAD_ID = "cli-session"
SEPARATOR = "-" * 60
EXIT_COMMANDS = {"exit", "quit", "q"}

NOT_FOUND_EXPLANATION = "התשובה אינה מבוססת על מסמך זה"


def _format_citation(index: int, citation: Citation, *, show_score: bool) -> str:
    score_part = ""
    if show_score and citation.score is not None:
        score_part = f"  ({percent(citation.score)})"
    return f'  [{index}] {citation.source}{score_part} — "{citation.excerpt}"'


def _format_considered(clause: RetrievedClause) -> str:
    gist = clause.content.splitlines()[0][:60]
    return f"  - {clause.source}  ({percent(clause.score)}) — {gist}"


def format_answer(answer: Answer) -> str:
    if answer.confidence == "not_found":
        floor_pct = percent(DEFAULT_SIMILARITY_FLOOR)
        lines = [
            f"✗ לא נמצאה התייחסות — {NOT_FOUND_EXPLANATION}",
            "",
            f"הסעיפים הקרובים ביותר שנבדקו, ולא עברו את סף ההתאמה ({floor_pct}):",
        ]
        lines.extend(_format_considered(clause) for clause in answer.considered)
        return "\n".join(lines)

    lines = []
    if answer.confidence == "uncertain":
        scores = [c.score for c in answer.citations if c.score is not None]
        best = max(scores, default=0.0)
        lines.append(f"⚠ תשובה לא ודאית ({percent(best)})")
        lines.append("")

    lines.append(answer.text)
    lines.append("")
    lines.append("מקורות (התאמה):" if answer.confidence == "uncertain" else "מקורות:")
    lines.extend(
        _format_citation(index, citation, show_score=answer.confidence == "uncertain")
        for index, citation in enumerate(answer.citations, start=1)
    )
    return "\n".join(lines)


def run() -> None:
    settings = Settings()
    graph = build_graph(
        build_llm(settings),
        build_retriever(settings),
        InMemorySaver(),
        search_k=settings.retrieval_k,
    )
    config: RunnableConfig = {"configurable": {"thread_id": THREAD_ID}}

    print("סוכן שאלות ותשובות לפקודת מס הכנסה. הקלידו שאלה, או 'exit' ליציאה.")
    while True:
        try:
            question = input("\n> ").strip()
        except EOFError:
            break

        if not question or question.lower() in EXIT_COMMANDS:
            break

        result = graph.invoke({"question": question}, config)
        print(SEPARATOR)
        print(format_answer(result["answer"]))
        print(SEPARATOR)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
