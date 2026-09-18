<!-- reflex managed begin (do not edit inside this block; add custom content outside the markers) -->
# Agents

This project uses the **Reflex agent skills** — up-to-date knowledge for building full-stack Python web apps with [Reflex](https://reflex.dev). Install them first, then use them as described below.

## Installing the Skills

Before working on this project, make sure the Reflex skills are available to you.

### Claude Code

If the skills are not already available, install them yourself by running these shell commands:

```bash
claude plugin marketplace add reflex-dev/agent-skills
claude plugin install reflex@reflex-agent-skills
```

Newly installed plugins load on the next session. After installing, tell the user to restart the session so the skills auto-load; until then, read the skill files directly from `~/.claude/plugins/` when you need them.

### Other agents (Cursor, OpenCode, Codex, Pi)

```
npx skills add reflex-dev/agent-skills
```

Or clone https://github.com/reflex-dev/agent-skills and copy the `skills/` folders into your agent's skill directory (see the repo README for paths).

### Verifying

Before writing or editing any Reflex code, confirm these three skills are available: `reflex-docs`, `setup-python-env`, and `reflex-process-management`. If they are not, STOP and run the install step above — do not proceed without them.

## Using the Skills

### Reflex documentation

For anything about Reflex APIs — components, state management, events, styling, database, routing, authentication — use the **reflex-docs** skill rather than relying on memory. It carries current, version-accurate docs.

### Initializing a new Reflex project

When starting a new Reflex project or setting up a development environment, you **must** follow the **setup-python-env** skill before doing anything else.

Do not skip any steps. Do not assume a virtual environment or Reflex is already available — always verify first by following the skill's instructions in order.

After the environment is ready and Reflex is installed, run:

```bash
reflex init
```

Then proceed with the user's request.

### Managing a Reflex process

When you need to compile, run, reload, or debug a Reflex application, follow the **reflex-process-management** skill for the correct sequence and error investigation steps.
<!-- reflex managed end -->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A learning-focused LangGraph/LangChain agent that classifies and routes incoming customer support tickets. It exercises thread-scoped state, cyclic routing, human-in-the-loop pause/resume, and RAG tool calling in one working graph. See `CONTEXT.md` for the domain vocabulary (Ticket, Category, TriageState, KnowledgeBase, Answer tool, Escalation, Review, Thread, Attempt) — use these terms, not synonyms, in code and docs.

The full design/architecture spec lives at `docs/specs/triage-agent.md`. It is the source of truth for "why" behind the code; read it before implementing anything non-trivial. The work is tracked as sequenced GitHub issues (`#9`–`#16`) off map issue `#1`.

## Commands

- Sync env: `uv sync`
- Add dependency: `uv add <package>` / dev dependency: `uv add --dev <package>`
- Run all tests: `uv run pytest`
- Run one test file: `uv run pytest tests/test_state.py`
- Run one test: `uv run pytest tests/test_state.py::test_name`
- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`
- Type check: `uv run ty check`

Never use `pip` or a bare `python` — always `uv run`/`uv add`.

## Architecture

Target layout (per `docs/specs/triage-agent.md`; not all modules exist yet — check what's actually present before assuming a file exists):

- `state.py` — `TriageState` (Pydantic) plus nested `Ticket`, `AttemptRecord`, `RetrievedSnippet`. Only `ticket` is required; `thread_id` is never part of state (invoke-time config only). `history` accumulates via LangGraph's reducer pattern (`Annotated[..., operator.add]`).
- `knowledge_base/` — one real, multi-section markdown policy doc per Category, sized so chunking is a genuine decision.
- `tools.py` — builds the `InMemoryVectorStore` retriever over the KnowledgeBase (`RecursiveCharacterTextSplitter`, chunk_size=400/chunk_overlap=50, `text-embedding-3-small`, cosine similarity) and wraps it as a `make_search_tool(retriever)` `@tool` for LLM-driven retrieval (`tool_choice="auto"`, never forced).
- `scripts/compare_embeddings.py` — standalone sandbox for comparing chunk size/overlap/embedding-model choices against a real KB doc. Never imported by `tools.py`, `nodes/`, or `graph.py`.
- `nodes/classify.py`, `nodes/answer.py`, `nodes/review.py` — each node takes `TriageState` plus its collaborators (LLM, retriever) and returns a partial-dict state update, so nodes are callable/testable in isolation. `review.py` splits `interrupt()` handling into pure, directly-testable helpers `_build_review_request`/`_apply_review_response`, with dedicated `ReviewRequest`/`ReviewResponse` models as the sole payload crossing the pause/resume boundary.
- `graph.py` — no I/O; exports only `build_graph(llm, retriever) -> CompiledGraph`. Topology: `classify` → `answer` → conditional on `needs_escalation` (→ `review` or `END`); `review` loops back to `answer` with feedback folded into state, or ends, bounded by `MAX_RETRIES = 3`.
- `dependencies.py` — builds the real LLM/retriever collaborators consumed by `build_graph` and `cli.py`.
- `cli.py` — the only module touching stdin/stdout or `.stream()`/`.invoke()`. Drives a full run with `InMemorySaver` and a stable `thread_id`, detects `__interrupt__`, and resumes via `Command(resume=...)`.

Escalation is an LLM judgment field on structured output (`AnswerDraft`), not a deterministic rule. Persistence is `InMemorySaver` only — no cross-process checkpointing, no API gateway (though the CLI/`dependencies.py` split leaves room for one later).

## Testing conventions

- Mock only the LLM boundary (a fake `Runnable`); let retrieval and vector-store lookups run for real against the toy KnowledgeBase — a fake retriever's shape could otherwise silently diverge from the real one.
- Node-isolation tests call node functions directly with a hand-built `TriageState`; no graph construction needed.
- Graph-level integration tests drive the compiled `build_graph(llm, retriever)` via `.stream()` with a real `InMemorySaver` and a stable `thread_id`; LLM still mocked, retrieval real.
- The retriever gets one behavioral test that issues a real query against the real KnowledgeBase and asserts relevant chunks come back — the one place embeddings/chunking correctness is checked directly (distinct from the manual `compare_embeddings.py` sandbox).
- `cli.py` and `dependencies.py` have no automated test seam (manual run only); `scripts/compare_embeddings.py` is a manual tool, not test-covered.

## Working practices

- Follow TDD: write/update a failing test first, then implement.
- For bug fixes, add a regression test before changing the code.
- Type hints on public functions; keep functions small and single-purpose.
- Issues and specs live in GitHub Issues (`gh` CLI) — see `docs/agents/issue-tracker.md` for conventions (creating/reading/labeling issues, the wayfinder map/child-ticket/blocking model). Triage labels are mapped in `docs/agents/triage-labels.md`.
- Never push directly to `main` — always work on a branch and open a PR (`gh pr create`).
