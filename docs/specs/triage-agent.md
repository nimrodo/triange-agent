# Customer Support Triage Agent — Spec

Consolidated spec for map [#1](https://github.com/nimrodo/triange-agent/issues/1). Not yet published to the issue tracker. Tickets [#9](https://github.com/nimrodo/triange-agent/issues/9)–[#16](https://github.com/nimrodo/triange-agent/issues/16) remain open as the sequenced build checklist; this doc is the source of truth for the "why" behind them.

## Problem Statement

There's no single reference implementation that shows, end to end, how a handful of core LangGraph/LangChain patterns fit together in one real agent. Learning them piecemeal — a memory example here, a HITL snippet there — never shows how thread-scoped state, cyclic routing, human-in-the-loop pause/resume, and RAG tool calling interact and constrain each other inside one working graph.

## Solution

Build a small but real Customer Support Triage agent that classifies an incoming Ticket, drafts an answer using a genuine RAG tool call against a toy KnowledgeBase, and routes low-confidence or judged-risky answers to a human Review step that can approve or reject-with-feedback, looping the answer back through revision up to a retry cap. The whole run is driven by a CLI, with no API gateway — but the code is structured so one could be added later without rework.

## User Stories

1. As a learner studying LangGraph, I want a Pydantic `TriageState` model that carries a Ticket through the whole graph, so that I can see how thread-scoped shared state is modeled and passed between nodes.
2. As a learner, I want `TriageState` to accumulate a history of `Attempt`s via LangGraph's reducer pattern, so that I can see how repeated cycles append to state instead of overwriting it.
3. As a learner, I want only `Ticket` required at graph entry and `thread_id` kept out of state entirely, so that I understand the boundary between graph state and invoke-time configuration.
4. As a learner, I want a `classify` node that assigns a Ticket's Category via an LLM call, so that I can see the simplest possible node: one collaborator in, one partial-state update out.
5. As a learner, I want Category constrained to a small fixed `Literal` set, so that I see how a narrow domain-specific model differs from a general classifier.
6. As a learner, I want an `answer` node where the LLM itself decides whether to call a search tool (not a forced call), so that I see genuine LLM-driven tool use rather than a scripted retrieval call.
7. As a learner, I want the Answer tool (`make_search_tool`) to wrap a retriever as a real `@tool` bound with `tool_choice="auto"`, so that I understand how LangChain exposes retrieval as a callable the LLM can choose to invoke.
8. As a learner, I want the `answer` node to capture `RetrievedSnippet`s from the tool call and then make a second `with_structured_output` call to produce the final `AnswerDraft`, so that I see a two-step LLM round: act, then decide.
9. As a learner, I want escalation to be an LLM judgment field on the structured output, not a deterministic rule, so that I see how "should a human get involved" is itself modeled as a model decision.
10. As a learner, I want a toy `KnowledgeBase` made of real multi-section markdown policy docs (one per Category), so that chunking strategy is a genuine, load-bearing decision rather than a toy detail.
11. As a learner, I want the KnowledgeBase chunked with `RecursiveCharacterTextSplitter` (`chunk_size=400`, `chunk_overlap=50`) and embedded with `text-embedding-3-small` into an `InMemoryVectorStore` (cosine similarity), so that I see a concrete, defensible embeddings configuration rather than default values.
12. As a learner, I want a standalone `scripts/compare_embeddings.py` that is never imported by the real pipeline, so that I have a safe hands-on sandbox for comparing chunk size, overlap, and embedding model choices against a real KB doc without touching production code paths.
13. As a learner, I want a `review` node that pauses the graph via `interrupt()` and resumes via `Command(resume=...)`, so that I see LangGraph's native HITL primitives in action rather than a polling or webhook-based workaround.
14. As a learner, I want the interrupt/resume boundary to cross dedicated `ReviewRequest`/`ReviewResponse` Pydantic models (not raw dicts or the full `TriageState`), so that I see how to keep a pause boundary narrow and typed.
15. As a learner, I want Review rejection to loop back to `answer` with feedback folded into state, and approval or a `MAX_RETRIES` cap (3) to exit, so that I see a cyclic graph with a real termination condition, not an infinite loop risk.
16. As a learner, I want the whole graph assembled in a single `graph.py` exporting only `build_graph(llm, retriever) -> CompiledGraph`, so that I see the entire routing topology in one place with no I/O mixed in.
17. As a learner, I want `dependencies.py` to build the real LLM and retriever collaborators separately from graph assembly, so that I see dependency injection cleanly separating "what the graph does" from "what it's wired to."
18. As a learner, I want a `cli.py` that drives the compiled graph via a plain synchronous `.stream()` call, detects `__interrupt__`, prompts for human input, and resumes with a stable `thread_id`, so that I see a complete, runnable HITL loop from the terminal.
19. As a learner, I want `cli.py` to be the only module touching stdin/stdout or `.stream()`/`.invoke()`, so that I see I/O cleanly isolated from graph logic, in a way that a future API handler could later swap in for the CLI without touching the graph.
20. As a maintainer, I want each node to take its collaborators (LLM, retriever) as parameters and return partial-dict updates, so that nodes are callable and testable in isolation without constructing a full graph.
21. As a maintainer, I want node-isolation unit tests that call node functions directly with a hand-built `TriageState`, mocking only the LLM boundary via a fake `Runnable`, so that tests stay fast and focused on node logic rather than full-graph wiring.
22. As a maintainer, I want retrieval to run for real (not mocked) inside node-isolation tests, so that a fake retriever's shape can't silently diverge from the real one.
23. As a maintainer, I want `review`'s `interrupt()` split into pure helpers `_build_review_request` and `_apply_review_response`, unit-tested directly, so that the untestable-in-isolation `interrupt()` call itself is the only thing left needing an integration test.
24. As a maintainer, I want a small set of integration tests driving the compiled graph end to end via `build_graph(llm, retriever)` with a real `InMemorySaver` checkpointer and a stable `thread_id`, so that interrupt/resume and the retry cycle are proven to work together, not just in isolation.
25. As a maintainer, I want three canonical integration scenarios covered — happy path (no escalation), escalate then approve, and escalate then reject repeatedly up to `MAX_RETRIES` — so that both graph exits (approval, retry cap) and the escalation branch are demonstrated.
26. As a maintainer, I want persistence limited to `InMemorySaver` with no cross-process checkpointing, so that the scope stays focused on demonstrating the pause/resume pattern within a single script/test run, not building durable infrastructure.
27. As a maintainer, I want no API gateway built in this effort, but the CLI/dependencies split structured so one could be added later, so that scope stays thin without foreclosing a natural next step.
28. As a maintainer, I want domain vocabulary (Ticket, Category, TriageState, KnowledgeBase, Answer tool, Escalation, Review, Thread, Attempt) used consistently across code and docs per `CONTEXT.md`, so that the codebase reads as a single coherent teaching example rather than a patchwork of synonyms.

## Implementation Decisions

- **State (`state.py`)**: `TriageState` is a Pydantic model with nested `Ticket`, `AttemptRecord`, and `RetrievedSnippet` models. `category` is a `Literal` over the fixed Category set. `history: list[AttemptRecord]` accumulates via LangGraph's reducer pattern (an `Annotated[..., operator.add]`-style accumulator, or the LangGraph-idiomatic equivalent). Only `ticket` is required at graph entry; all other fields have safe defaults. `thread_id` is never part of `TriageState` — it's invoke-time config only (`{"configurable": {"thread_id": ...}}`).
- **KnowledgeBase (`knowledge_base/`)**: one real, multi-section markdown policy doc per Category (3–4 docs total), substantial enough that chunk boundaries meaningfully affect which sections come back for a query — not FAQ-style one-liners.
- **Retrieval (`tools.py`)**: `RecursiveCharacterTextSplitter` with `chunk_size=400`, `chunk_overlap=50`; embeddings via `text-embedding-3-small` (`langchain-openai`); vector store is `InMemoryVectorStore` (`langchain-core`, zero extra deps), which fixes the similarity metric to cosine. `tools.py` builds the vector store from the KnowledgeBase docs and exposes retrieval over it.
- **Comparison script (`scripts/compare_embeddings.py`)**: standalone entry point, never imported by `tools.py`, `nodes/`, or `graph.py`. Exists purely as a hands-on vehicle for comparing chunk size/overlap/embedding-model choices against a real KB doc.
- **Answer tool (`tools.py`)**: `make_search_tool(retriever)` wraps an injected retriever into a real `@tool`. The LLM used in the `answer` node is bound to this tool with `tool_choice="auto"` — the LLM decides at its own discretion whether to retrieve, never a forced call.
- **Classify node (`nodes/classify.py`)**: takes `TriageState` and an LLM collaborator; one LLM call assigns `category`; returns a partial-dict state update.
- **Answer node (`nodes/answer.py`)**: takes `TriageState`, an LLM collaborator, and a retriever; orchestrates the round explicitly — LLM call with the search tool bound (`tool_choice="auto"`), captures any `RetrievedSnippet`s from a tool call directly (not left implicit in tool-call output), then a second `with_structured_output(AnswerDraft)` call producing the drafted answer plus an `escalation` judgment field. Returns a partial-dict state update.
- **Review node (`nodes/review.py`)**: defines `ReviewRequest`/`ReviewResponse` Pydantic models as the sole payload crossing the `interrupt()`/`Command(resume=...)` boundary. The node body is split into pure helpers — `_build_review_request(state) -> ReviewRequest` and `_apply_review_response(state, response: ReviewResponse) -> dict` — with `interrupt()` wired around them to pause and resume.
- **Graph (`graph.py`, no I/O)**: exports only `build_graph(llm, retriever) -> CompiledGraph`. Topology: `classify` → `answer` → conditional on `needs_escalation` routing to `review` or `END`; `review` is conditional on `review_decision`/`retry_count`, looping back to `answer` (re-running it from scratch with feedback folded into state) or ending. `MAX_RETRIES = 3` bounds the loop. No `run_ticket`-style wrapper — `build_graph` is the only export, since a wrapper would be a premature abstraction for a nonexistent second caller.
- **Dependencies (`dependencies.py`)**: builds the real LLM and retriever collaborators (`langchain-openai` chat model, the `tools.py`-backed retriever) that `build_graph` and `cli.py` consume. Shared by any future entry point.
- **CLI (`cli.py`)**: the only module touching stdin/stdout or `.stream()`/`.invoke()`. Drives a full run via a plain synchronous `build_graph(...).stream(...)` call with `InMemorySaver` as the checkpointer and a stable `thread_id`, detects `__interrupt__`, collects human input via an `input()`-loop, and resumes via `Command(resume=ReviewResponse(...))` using the same `thread_id`.
- **Checkpointer**: `InMemorySaver` throughout — HITL resume is demonstrated within a single script/test run; no `SqliteSaver` or cross-process persistence.
- **Layout**: `state.py`, `nodes/classify.py`, `nodes/answer.py`, `nodes/review.py`, `graph.py`, `tools.py`, `dependencies.py`, `cli.py`, `knowledge_base/`, `scripts/compare_embeddings.py`.
- **Provider**: `langchain-openai` (already a dependency) for both chat and embeddings models.

## Testing Decisions

A good test here exercises externally observable node/graph behavior — the state update a node produces, or the routing/pause/resume behavior of the compiled graph — never internal call sequencing or prompt text. Mock only the LLM boundary (the one true external dependency with no cheap deterministic substitute); let retrieval and vector-store lookups run for real against the toy KnowledgeBase.

- **Node-isolation unit tests** (one seam: call node functions directly): `nodes/classify.py`, `nodes/answer.py`, and the `review.py` helpers `_build_review_request`/`_apply_review_response` are each called directly with a hand-built `TriageState` (and, for review, a hand-built `ReviewResponse`), mocking only the LLM via a fake `Runnable`. Retrieval in `answer`'s test runs for real against `InMemoryVectorStore`. `answer` is asserted on both a non-escalation and an escalation path. `classify` is asserted on Category assignment. `state.py` models are round-tripped through direct construction/validation with no graph involved.
- **Graph-level integration tests** (second seam: `build_graph(llm, retriever)` driven via `.stream()`): a real `InMemorySaver` checkpointer and a stable `thread_id`, LLM still mocked, retrieval real. Three canonical scenarios: happy path (no escalation, no interrupt fires), escalate then approve (interrupt fires once, resuming with approval ends the run), escalate then reject repeatedly up to `MAX_RETRIES` (retry loop re-runs `answer` with feedback folded into state, terminates at the cap). A separate structural test compiles the graph and asserts routing/edges match the design without necessarily driving a full run.
- **`retriever`/vector-store test**: a real query against the real KnowledgeBase asserts relevant chunks come back — this is the one place embeddings/chunking correctness gets a behavioral check, distinct from the standalone comparison script.
- **Not tested**: `cli.py` and `dependencies.py` have no dedicated automated test seam — `cli.py`'s acceptance criterion is a manual run against a sample ticket producing a visible triage result with a live HITL pause/resume. `scripts/compare_embeddings.py` is a manual comparison tool, not test-covered.
- **Prior art**: none yet in this repo (greenfield `src/triange_agent/`) — this spec establishes the testing pattern for everything that follows it.

## Out of Scope

- Building an actual API gateway (FastAPI or otherwise). The CLI/`dependencies.py` split is structured to allow one later, but none is built here.
- Persistence across process restarts (e.g. `SqliteSaver` or another durable checkpointer). HITL resume is demonstrated within a single script/test run only.
- Auth, multi-tenancy, or deployment concerns of any kind.
- A Reflex UI or any other friendlier runner — would require the API-gateway boundary this effort deliberately excludes.
- Exact LLM prompt wording/temperature tuning for `classify` and `answer` — left to implementation, not specified here.
- A graph visualization/export step (e.g. Mermaid) — not decided as in-scope.

## Further Notes

- This spec consolidates the design decisions already closed out across #2–#8 (linked from map #1) into one implementation-ready document. The granular task tickets #9–#16 remain open as the sequenced build checklist (`#9` → `#10`/`#11` → `#12` → `#13` → `#14` → `#15`/`#16`); this spec is the single source of truth for the "why" and cross-cutting decisions behind them, so implementers should read this spec first and use #9–#16 as the ordered work breakdown.
- Domain vocabulary is fixed by `CONTEXT.md` (Ticket, Category, TriageState, KnowledgeBase, Answer tool, Escalation, Review, Thread, Attempt) — use these terms, not synonyms, in code, tests, and docs.
- Final KnowledgeBase doc topics/section counts are not yet chosen — left to whoever builds #9, as long as docs stay real multi-section policy text sized so chunking is a genuine decision.
- Per #1's execution override: after each of the underlying tickets resolves, the running "Triage Graph" HTML Artifact (design explanation + interactive simulation) should be republished to its existing URL — this is a standing instruction on the map, not part of this spec's scope.
