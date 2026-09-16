# CLI output prototype: cited, confidence-flagged answers

Resolves wayfinder ticket "Prototype CLI output format for cited, confidence-flagged answers" (issue #34).

Prototype script: `scripts/prototype_cli_output.py` (this branch, `prototype/cli-output`), run with `uv run scripts/prototype_cli_output.py`.

## Decision

For each confidence state:

- **answered**: prose answer, then a `מקורות:` (sources) list — each Citation numbered `[1]`, `[2]`, ... with its Clause identifier (e.g. `סעיף 121(ב)`) and a verbatim quoted excerpt.
- **uncertain**: prefixed with `⚠ תשובה לא ודאית (<similarity>)`, otherwise the same shape as `answered` — prose answer plus numbered, cited sources, each also annotated with its similarity score.
- **not_found**: prefixed with `✗ לא נמצאה התייחסות`, a one-line explanation, no synthesized answer or citation list. Instead, lists the top few clauses that were considered but fell below the similarity floor, each with its (low) score and a one-line gist, so the accountant can see the agent didn't just give up blind.

Similarity scores are shown throughout — not just the answered/uncertain/not_found label — because "show more than just the label" was explicit user feedback: the accountant should be able to see *how* confident the agent was, and what near-misses looked like when nothing qualified.

See `scripts/prototype_cli_output.py` for the full mocked exchanges (one example per state).
