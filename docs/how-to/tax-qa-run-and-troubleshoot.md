# How to run and troubleshoot the Tax Q&A chatbot

Recipes for getting the Income Tax Q&A chatbot running and for making sense of what it tells
you. If you just want to ask it a question and someone else has already started it for you, see
the [tutorial](../tutorials/tax-qa-first-question.md) instead.

## How to run it locally

1. Make sure [`uv`](https://docs.astral.sh/uv/) is installed, then install dependencies:

   ```bash
   uv sync
   ```

2. Copy `.env.example` to `.env` and fill in at least one provider's settings (see
   [How to switch providers](#how-to-switch-providers) below for what each variable means).

   ```bash
   cp .env.example .env
   ```

3. Start the web app:

   ```bash
   uv run reflex run
   ```

4. Open `http://localhost:3000/` in your browser.

The first run indexes the Income Tax Ordinance into a local vector store — this can take a
while, especially on a rate-limited or free-tier embedding provider. Subsequent runs reuse the
already-indexed store and start faster.

## How to run it with Docker

```bash
docker build -t tax-qa .
docker run -p 3000:3000 -p 8000:8000 --env-file .env tax-qa
```

Then open `http://localhost:3000/`.

Notes:

- The image defaults to `CHAT_PROVIDER=openai` / `EMBEDDING_PROVIDER=openai` — pass
  `OPENAI_API_KEY` via `--env-file .env` or `-e`. To use Ollama instead, override both provider
  variables and point `OLLAMA_BASE_URL` at a reachable Ollama instance (e.g.
  `http://host.docker.internal:11434` for one running on the host machine).
- The build requires `.research-data/ordinance.pdf` to exist locally (it's gitignored) — the
  image bakes it in, so building currently only works on a machine that already has that file.
- Startup runs the frontend build and imports the backend's ML/LangChain stack concurrently,
  which is real CPU and memory work — give Docker at least a few CPUs and several GB of RAM, or
  the backend can take a long time (or appear to hang) to come up.
- Chat history and the retrieval vector store are both in-memory, so they reset on restart and
  aren't shared across replicas if you run more than one container.

## How to switch providers

The chatbot's chat model and embedding model are each configured independently via `.env`, and
can be different providers. Set these two variables to `openai`, `gemini`, or `ollama`:

```bash
CHAT_PROVIDER=openai
EMBEDDING_PROVIDER=openai
```

Then fill in the matching section of `.env.example` for whichever provider(s) you chose:

- **OpenAI** — needs `OPENAI_API_KEY`.
- **Gemini** — needs `GEMINI_API_KEY`.
- **Ollama** — needs a reachable `OLLAMA_BASE_URL` (defaults to `http://localhost:11434`), and
  no API key. The embedding model must be multilingual — the Ordinance is Hebrew, and an
  English-only embedding model (e.g. `nomic-embed-text`) will silently produce near-random
  retrieval instead of an error. The default, `bge-m3`, is multilingual.

If you switch `EMBEDDING_PROVIDER`, the chatbot builds a separate, fresh index for the new
provider rather than reusing the old one — so the first question after switching will trigger
re-indexing.

If you're on a rate-limited free-tier embedding API (e.g. Gemini's free tier), lower
`EMBEDDING_BATCH_SIZE` and raise `EMBEDDING_REQUEST_DELAY_SECONDS` in `.env` to stay under the
provider's limits during indexing.

## How to interpret an uncertain or not-found answer

- **⚠ ודאות לא נמוכה (uncertain)** — the agent found a section of the Ordinance related to your
  question, but isn't confident it fully answers it. Read the cited excerpt yourself before
  relying on the answer; consider rephrasing the question to be more specific.
- **✗ לא נמצא (not found)** — nothing in the indexed Ordinance cleared the similarity threshold
  for your question. No answer is synthesized. The panel below still lists the closest sections
  the agent considered, in case one of them is relevant despite scoring below the threshold —
  worth a manual look, especially if your question used different wording than the Ordinance
  does.

If you're consistently getting not-found answers for questions you'd expect the Ordinance to
cover, check that indexing completed successfully (see the startup logs) and that
`EMBEDDING_PROVIDER` matches a model you can actually reach.
