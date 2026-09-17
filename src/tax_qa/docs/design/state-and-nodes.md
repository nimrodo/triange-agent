# State schema and node structure

Resolves wayfinder ticket "Design state schema and node structure for the Income Tax Q&A graph" (issue #35).

## State model

Mirrors `TriageState`'s shape, adapted to this context's vocabulary (see `src/tax_qa/CONTEXT.md`):

- `TaxQAState` carries the current `question: str`, the current turn's `answer: Answer | None`, and `history: Annotated[list[Exchange], operator.add]` — accumulated across turns via LangGraph's reducer pattern, same mechanism as `TriageState.history`.
- `Exchange` holds one past turn: the `question` asked and the `answer` it produced.
- `Answer` is its own model (not flattened onto state): `text: str`, `citations: list[Citation]`, `confidence: Literal["answered", "uncertain", "not_found"]`.
- `Citation` is its own model: a Clause identifier plus a verbatim quoted excerpt (see `src/tax_qa/CONTEXT.md`), plus an optional `score: float | None` carried through from the originating `RetrievedClause` -- not part of the Citation concept itself, but needed at the CLI/UI boundary to annotate each source with its similarity score on `uncertain` Answers (issue #43).
- `Answer.considered: list[RetrievedClause]` holds the near-miss Clauses that were retrieved but didn't clear the similarity floor, populated only on `not_found` -- see step 3 below and the CLI's "considered but didn't qualify" display (issue #43).
- The retrieval tool's output extends the triage agent's `RetrievedSnippet` shape with a similarity score per retrieved Clause (needed for the confidence floor check below) — e.g. `RetrievedClause(content: str, source: str, score: float)`.

## Node structure

One combined node (mirrors `src/triange_agent/nodes/answer.py`'s single-pass shape — no separate `retrieve` node):

1. Bind the search tool with `tool_choice="auto"`, invoke the tool-calling LLM with the Question.
2. Execute any tool calls to get back `RetrievedClause`s (content + source + score).
3. **Similarity floor check, in code, before any structured-output call**: if no retrieved Clause's score clears the floor, short-circuit — return `confidence="not_found"` with no synthesized answer text, just the near-miss Clauses (for the CLI's "considered but didn't qualify" display). The LLM is never invoked to draft an answer in this branch — cheaper, and removes any chance of the LLM overriding the floor.
4. Otherwise, call `llm.with_structured_output(AnswerDraft)` (an `AnswerDraft`-equivalent structured type carrying `text` and a self-rated `confidence: Literal["answered", "uncertain"]`) to draft the answer.
5. Assemble the `Answer`, append an `Exchange` to `history`, return the partial-dict state update.

## Testing

Same conventions as the triage agent: mock only the LLM boundary, real retrieval against real data. For this context, "real data" means a small, real excerpt of the actual Income Tax Ordinance (a handful of genuine Clauses, not synthetic text) checked into `tests/fixtures/` — not the full 312-page document — mirroring the triage agent's toy `KnowledgeBase` but sourced from genuine document text. Exact fixture construction (which Clauses, how they're chunked) is left to implementation time, since it depends on the chunking/embedding strategy (issue #31, in progress).

- Node-isolation tests call the answer node directly with a hand-built `TaxQAState`, mocked LLM, real retriever over the fixture excerpt.
- Graph-integration tests drive the compiled graph via `.stream()`/`.invoke()` with a real checkpointer and a stable thread id, mocked LLM, real retrieval — same shape as the triage agent's graph-level tests, covering multi-turn `history` accumulation across turns.
