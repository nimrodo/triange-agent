# triange-agent

## Running the Income Tax Q&A chatbot with Docker

Builds and serves the Reflex web UI (`src/tax_qa/web.py`) in a single container.

```sh
docker build -t tax-qa .
docker run -p 3000:3000 -p 8000:8000 --env-file .env tax-qa
```

Then open http://localhost:3000/.

Notes:

- The image defaults to `CHAT_PROVIDER=openai` / `EMBEDDING_PROVIDER=openai`;
  pass `OPENAI_API_KEY` via `--env-file .env` or `-e`. To use Ollama instead,
  override both provider vars and point `OLLAMA_BASE_URL` at a reachable
  Ollama instance (e.g. `http://host.docker.internal:11434` for one running
  on the host machine).
- The build requires `.research-data/ordinance.pdf` to exist locally (it's
  gitignored) — the image bakes it in, so building currently only works on a
  machine that already has that file.
- Startup runs the Reflex frontend build (Next.js/Bun) and imports the
  Python backend's ML/LangChain stack concurrently, which is real CPU and
  memory work — give Docker at least a few CPUs and several GB of RAM, or
  the backend can take a long time (or appear to hang) to come up.
- Single-replica assumption: chat history and the retrieval vector store are
  both in-memory (`InMemorySaver`, `InMemoryVectorStore`), so they reset on
  restart and aren't shared across replicas — same as running the app
  outside Docker today.
- This covers local `docker run` only. Deploying behind a reverse proxy or
  on a single-port host (Fly.io, Render, etc.) needs additional config
  (Reflex bakes the backend URL into the frontend build) and isn't covered
  here.