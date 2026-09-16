# LangGraph and LangChain concepts in this codebase

This project exists to teach five patterns. Each section below explains the general
LangGraph/LangChain concept first, then points at exactly where this repo demonstrates it. If
you haven't run the agent yet, do that first via
[the getting-started tutorial](../tutorials/getting-started.md) — this doc will make more sense
once you've seen these patterns fire.

For what this repo specifically *decided* about each pattern (chunk sizes, retry limits, field
names), see `docs/specs/triage-agent.md` — this doc explains the *concepts* those decisions sit
on top of.

## 1. Thread-scoped state and memory

**The concept.** A LangGraph graph is built around one shared state object that every node reads
from and writes to, rather than nodes passing data to each other directly as function arguments.
You define that state's shape (here, a Pydantic model), and each node returns a **partial
update** — just the fields it changed — which LangGraph merges into the running state for you.

That state is scoped to a **thread**: an identifier (`thread_id`, passed as invoke-time config,
never as part of the state itself) that a **checkpointer** uses to persist and retrieve state
between separate calls to `.stream()`/`.invoke()`. This is what makes pause-and-resume possible
at all — without a checkpointer, a graph would have no memory of where it left off.

One wrinkle: some fields shouldn't simply be overwritten on each update — they should
*accumulate*. `history` is a list of every drafted attempt so far, and if `answer` just returned
`{"history": [new_attempt]}`, LangGraph needs to know that means "append," not "replace." That's
what `Annotated[list[AttemptRecord], operator.add]` declares: a **reducer** function
(`operator.add`, i.e. list concatenation) that LangGraph applies instead of plain overwrite.

**How this repo demonstrates it.**
- `src/triange_agent/state.py` — `TriageState` is the shared state; `history` is the one
  reducer-backed field, everything else uses plain-overwrite semantics.
- `src/triange_agent/graph.py` — `build_graph(..., checkpointer=None)` accepts any
  `BaseCheckpointSaver`; this project always passes `InMemorySaver`.
- `src/triange_agent/cli.py` — `THREAD_ID = "cli-session"` is the stable identifier passed via
  `{"configurable": {"thread_id": ...}}` on every `.stream()` call, which is what lets the
  second call (after a human responds) resume the *same* run instead of starting a new one.

## 2. Cyclic conditional routing

**The concept.** Most tutorials show a graph as a straight line: node A to node B to END. But a
`StateGraph`'s edges don't have to form a DAG that only moves forward — a **conditional edge**
routes based on a function of the current state, and nothing stops that function from routing
back to a node the graph already visited. That's a **cycle**: the same node runs again, with
different state than last time (because the previous pass updated it). Cycles are how you model
retry loops, refinement loops, or any "try again with feedback" behavior — but an unbounded
cycle is a bug, not a feature, so real cyclic graphs almost always carry an explicit counter and
a cap in the state used to route.

**How this repo demonstrates it.**
- `src/triange_agent/graph.py` — `_route_after_review` is the conditional-edge function: if
  `review_decision == "rejected"` and `retry_count` hasn't exceeded `MAX_RETRIES` (3), route back
  to `answer`; otherwise route to `END`. The `answer` → `review` → `answer` cycle is exactly this
  pattern.
- `src/triange_agent/state.py` — `retry_count: int = 0` is the counter that makes the cap
  possible; it's incremented in `nodes/review.py`'s `_apply_review_response`, never in the
  routing function itself (routing functions in this repo are pure predicates over state, they
  don't mutate it).
- `tests/test_integration.py` — `test_escalate_then_reject_repeatedly_terminates_at_the_retry_cap`
  drives the full cycle for real and asserts it actually stops at the cap rather than looping
  forever.

## 3. Human-in-the-loop pause and resume

**The concept.** LangGraph's `interrupt()` function, called from inside a node, does something
unusual: it raises a special exception that unwinds execution back out of `.stream()`/
`.invoke()`, *without* discarding the state accumulated so far (the checkpointer already
persisted it). The caller sees a distinguishable `__interrupt__` entry in the stream instead of
a normal node output. Execution is now genuinely paused — the Python process can do anything at
this point, including exit — as long as the same checkpointer and `thread_id` are available
later.

To resume, the caller calls `.stream()`/`.invoke()` again, but instead of passing new input, it
passes `Command(resume=<value>)`. LangGraph resumes the interrupted node, and `interrupt()`
*returns* the value from the `Command` — as if the call had simply returned normally all along.
The node's code doesn't need any special "am I resuming?" branch; from the node's point of view,
`interrupt()` just eventually returns something.

A design choice worth noticing: the value crossing that boundary doesn't have to be (and
usually shouldn't be) the raw graph state. A human reviewer doesn't need to see every internal
field, and a stable-looking human-facing contract lets you refactor internal state without
breaking whatever's driving the human side of things.

**How this repo demonstrates it.**
- `src/triange_agent/nodes/review.py` — `ReviewRequest`/`ReviewResponse` are exactly that
  narrow, stable contract. `review()` builds a `ReviewRequest` from state, calls
  `interrupt(request)`, and gets a `ReviewResponse` back on resume — never the full
  `TriageState` in either direction.
- Because `interrupt()` can't meaningfully be called with a fake value in a unit test, the
  node's logic is split into two pure, directly-testable helpers —
  `_build_review_request(state) -> ReviewRequest` and
  `_apply_review_response(state, response) -> dict` — leaving only the pause/resume mechanic
  itself to an integration test. See `tests/test_review.py` for the unit tests and
  `tests/test_integration.py` for the integration coverage.
- `src/triange_agent/cli.py` — the loop that detects `chunk["__interrupt__"]`, prompts a human,
  and calls `graph.stream(Command(resume=response), config)` is the concrete resume mechanic
  described above, end to end.

## 4. RAG tool calling, decided by the LLM

**The concept.** There are two very different ways to combine retrieval with an LLM call. The
simpler one — call your retriever yourself, stuff the results into the prompt, then call the
LLM — teaches you about retrieval, but nothing about *tool calling*. The pattern this repo
actually demonstrates is different: you `bind_tools([...], tool_choice="auto")` to the LLM, hand
it a `HumanMessage`, and let the model itself decide, based on the input, whether calling the
tool would help. If it decides yes, the model's response comes back with `tool_calls` populated
instead of a direct answer; your code is responsible for actually executing the tool call and
feeding the result back in, but the *decision* to retrieve was the model's, not yours.
`tool_choice="auto"` is what leaves that decision open — forcing the call every time would
collapse this back into the simpler pattern.

**How this repo demonstrates it.**
- `src/triange_agent/tools.py` — `make_search_tool(retriever)` wraps a real `InMemoryVectorStore`
  retriever as a `@tool`-decorated function, `search_knowledge_base`, using
  `response_format="content_and_artifact"` so the tool can return both a string for the model to
  read and a structured `list[RetrievedSnippet]` for the calling code to use directly (rather
  than having to re-parse the model-facing string).
- `src/triange_agent/nodes/answer.py` — `answer()` is the explicit two-call round this pattern
  requires: first `llm.bind_tools([search_tool], tool_choice="auto")` and inspect
  `response.tool_calls` for whether/what it decided to search; then a *second*,
  `with_structured_output(AnswerDraft)` call that produces the actual drafted answer and the
  `needs_escalation` judgment, now with any retrieved snippets folded into its prompt.
- Run `uv run pytest tests/test_answer.py -v` and read the two scenarios — one where the fake
  LLM's tool response has `tool_calls=[]` (skips retrieval) and one where it doesn't — to see the
  branch this decision creates play out in isolation.

## 5. Embeddings as a load-bearing decision

**The concept.** In a lot of introductory RAG material, "just embed the documents" is treated as
a solved, uninteresting step. In practice, three choices — how you split documents into chunks,
which embedding model you use, and which similarity metric your vector store uses — jointly
determine what gets retrieved for a given query, and getting any of them wrong silently degrades
answer quality without throwing an error. Chunking that's too large blends unrelated
sections together; chunking that's too small strips away the surrounding context a chunk needs
to be useful; the wrong model or metric ranks distant matches above close ones. This is only a
*genuine* decision when your source documents are large and multi-part enough that where you cut
them actually matters — a two-line FAQ doesn't exercise this at all.

**How this repo demonstrates it.**
- `src/triange_agent/knowledge_base/` — four real, multi-section markdown policy docs
  (`billing.md`, `technical.md`, `account.md`, `general.md`), each covering several
  sub-topics, specifically so that chunk boundaries can land in the middle of a doc and matter.
- `src/triange_agent/tools.py` — `build_vector_store()` is the production pipeline:
  `RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)`, OpenAI's
  `text-embedding-3-small`, and `InMemoryVectorStore`, whose similarity search is fixed to cosine
  similarity (a property of that vector store class, not a parameter you choose separately).
- `scripts/compare_embeddings.py` — a standalone script, deliberately never imported by
  `tools.py` or anything the graph touches, that runs the *same* query against multiple
  chunk-size/overlap/model configurations side by side and prints each result's similarity
  score. Run it yourself (`uv run python scripts/compare_embeddings.py`) to see, concretely, how
  changing chunking or model choice changes which chunk of a real policy doc comes back first.
- `tests/test_tools.py` — `test_retrieve_snippets_returns_relevant_chunks_for_real_query` is the
  one test in this repo that asserts on real embeddings/retrieval quality rather than mocking it
  away, precisely because this is the one place where "does it actually work well" can't be
  faked without losing the point of the test.
