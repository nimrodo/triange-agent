# Getting started: run the Triage agent

This tutorial walks you through running the Customer Support Triage agent end to end. By the
end, you will have watched all four patterns this project exists to teach happen in front of
you: thread-scoped state accumulating across a run, a cyclic retry loop, a human-in-the-loop
pause and resume, and an LLM deciding for itself whether to search a knowledge base.

You don't need to read any code first. Everything you need to notice is called out as you go.

## Prerequisites

- Python 3.13 and [`uv`](https://docs.astral.sh/uv/) installed.
- An OpenAI API key (the agent's chat and embedding models are both OpenAI-hosted).
- This repo cloned locally, with a terminal open at its root.

Install dependencies:

```bash
uv sync
```

Set your API key so `dependencies.py`'s `Settings` model can pick it up:

```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

## Step 1: Run it

```bash
uv run triange-agent
```

This runs `src/triange_agent/cli.py`, which drives a single hard-coded sample Ticket through
the graph:

> **Subject**: Refund request outside the stated window
> **Body**: I was charged for a subscription I cancelled two months ago and the refund policy
> page doesn't cover this case. Can you refund me?

You'll see log lines print as the graph streams through its nodes. The first two are
`classify` (assigns a Category — `billing`, `technical`, `account`, or `general`) and `answer`
(drafts a response, deciding along the way whether it needs a human).

## Step 2: Watch the state accumulate

Every node returns a partial update to a single shared `TriageState` object, keyed by a
`thread_id`. As `answer` runs, it appends an `AttemptRecord` — the drafted answer plus which
attempt number this is — to `state.history`. This is what "thread-scoped memory" means in
practice: nothing is passed between nodes as function arguments; it all lives in state that
LangGraph's checkpointer persists per thread.

Nothing to do here except notice it — the printed chunks are the raw partial-state dicts each
node returned.

## Step 3: Hit the human-in-the-loop pause

Because this sample ticket describes a refund outside the normal policy window, `answer` should
judge `needs_escalation = True`. When that happens, the graph routes to the `review` node, which
calls LangGraph's `interrupt()`. The CLI detects this and prints:

```
--- Human review requested ---
Category: billing
Attempt #1
Escalation reason: ...
Draft answer:
...

Approve or reject? [approve/reject]:
```

At this exact moment, graph execution is frozen mid-run. The `InMemorySaver` checkpointer has
the entire `TriageState` saved against this run's `thread_id` (`"cli-session"`, a constant in
`cli.py`). Nothing is lost — you could restart the process and resume later, as long as it's
still the same Python process with the same in-memory saver (there is no cross-process
persistence in this project; see [the concepts doc](../explanations/langgraph-concepts.md) for
why that's a deliberate scope decision).

## Step 4: Resume it

Type `reject` and give any feedback string when prompted, e.g. `Check the cancellation date
policy more carefully.`

The CLI sends your answer back via `Command(resume=ReviewResponse(...))`, using the *same*
`thread_id`. Watch what happens: the graph doesn't restart from `classify` — it resumes exactly
where it left off, routes back to `answer`, and `answer` runs again, this time with your
feedback folded into its prompt. This is the cyclic retry loop: `review` → `answer` → `review`
→ ..., bounded by `MAX_RETRIES = 3` so it can't loop forever.

Reject a couple more times if you want to see `retry_count` climb, or type `approve` to end the
run immediately. Either way, the CLI prints the final result once the graph reaches `END`:

```
--- Triage result ---
Category: billing
Final answer: ...
```

## Step 5: See the RAG tool decide for itself

Re-run `uv run triange-agent` and pay closer attention to `answer` this time. Before drafting,
`answer` binds a `search_knowledge_base` tool to the LLM with `tool_choice="auto"` — meaning the
model itself decides whether retrieving from the knowledge base would help, rather than being
forced to call it every time. The knowledge base is four real, multi-section markdown policy
docs under `src/triange_agent/knowledge_base/` (`billing.md`, `technical.md`, `account.md`,
`general.md`) — open one to see what the model has available to search.

If you want to see this pattern more directly than the CLI's log output shows, the isolated
node test makes it explicit:

```bash
uv run pytest tests/test_answer.py -v
```

## Where to go next

- **[LangGraph/LangChain concepts used here](../explanations/langgraph-concepts.md)** — for
  each pattern you just watched happen, why it works the way it does and where to read the
  implementation.
- `scripts/compare_embeddings.py` is a standalone sandbox for experimenting with chunk size,
  chunk overlap, and embedding model choice against a real knowledge base doc — run it directly
  with `uv run python scripts/compare_embeddings.py` to see how those choices change what gets
  retrieved.
- `docs/specs/triage-agent.md` is the full design spec, if you want the "why" behind every
  implementation decision in one place.
