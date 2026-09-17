# Embedding model comparison: bge-m3 vs. nomic-embed-text for the Income Tax Ordinance

Research spike for issue #39 (follow-up to #31 on `research/chunking-embedding`).
Question: `nomic-embed-text` scored 0/6 on both chunking strategies in #31's
retrieval eval, attributed to weak Hebrew/multilingual support. Does a
free/local embedding model with genuine, documented multilingual (including
Hebrew) support do meaningfully better on the same eval? Paid
`text-embedding-3-small` remains out of scope (issue #37, declined for cost).

Scratch script used to produce these findings: `scripts/scratch_chunk_embed_compare.py`
(this branch only) — the same script #31 used, with
`EMBEDDING_MODEL` swapped from `nomic-embed-text` to `bge-m3`. Run with:

```
uv run --with pymupdf --with langchain-text-splitters --with langchain-ollama \
    --with langchain-core python scripts/scratch_chunk_embed_compare.py
```

All numbers below are the actual output of that run against the real PDF at
`.research-data/ordinance.pdf`, not estimates.

## Model chosen: bge-m3 (BAAI), served via Ollama

Candidates considered, per the issue's shortlist:

- **Ollama-servable**: `mxbai-embed-large`, `bge-m3`, `snowflake-arctic-embed`
- **Local sentence-transformers**: `intfloat/multilingual-e5-large(-base)`,
  `sentence-transformers/LaBSE`, `paraphrase-multilingual-mpnet-base-v2`
- **Gemini embeddings** (`text-embedding-004` / `gemini-embedding-001`) —
  checked but not chosen: as of this writing Google's Gemini API pricing page
  lists a free tier for `gemini-embedding-001` only under the rate-limited
  "free tier" available to unpaid API keys, functionally equivalent to the
  OpenAI situation already ruled out in #37 for "requires signing up for a
  hosted API, not truly local/offline" reasons the issue's Ollama/local
  candidates avoid. Kept as a fallback, not the first choice, since the repo
  already runs everything else (chat, prior embeddings) locally via Ollama.

**Picked bge-m3** (`BAAI/bge-m3`, served locally via `ollama pull bge-m3`):

- The official model card and Ollama library page both advertise "support
  [for] more than 100 working languages" (BAAI/bge-m3 model card,
  https://huggingface.co/BAAI/bge-m3; Ollama library,
  https://ollama.com/library/bge-m3).
- Unlike a bare "100+ languages" marketing claim, the BGE-M3 paper's own
  cross-lingual evaluation table (Table 3, MKQA cross-lingual retrieval)
  explicitly reports per-language results for Hebrew (abbreviated `he`)
  alongside dozens of other languages — i.e. Hebrew retrieval quality is
  something the model's authors themselves measured and published, not an
  inference from a language-count claim. (Chen et al., "BGE M3-Embedding:
  Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings
  Through Self-Knowledge Distillation," https://arxiv.org/abs/2402.03216,
  Table 3.) This is a stronger signal than `mxbai-embed-large` or
  `snowflake-arctic-embed`, whose model cards make general multilingual/100+
  language claims but do not call out Hebrew or a Hebrew-inclusive
  cross-lingual benchmark specifically.
- It is directly `ollama pull`-able (`ollama pull bge-m3`, 1.2 GB, 8K context
  window per the Ollama library page), so it drops into this repo's existing
  `EMBEDDING_PROVIDER=ollama` setup with no new dependency or provider code —
  only `OLLAMA_EMBEDDING_MODEL` needs to change once this is adopted outside
  the spike script.
- Note for completeness: MIRACL, BGE-M3's other headline benchmark, covers 18
  languages and does *not* include Hebrew — the Hebrew-specific evidence comes
  from the MKQA table, not MIRACL, which is why this doc cites Table 3
  specifically rather than the MIRACL results.

`intfloat/multilingual-e5-large` was a close second (mE5 is trained on
mC4/CC100 data, whose language lists include Hebrew) but would have required
adding `sentence-transformers` as a new dependency and a new local-inference
code path outside this repo's existing Ollama-embeddings integration; `bge-m3`
was preferred as the smaller change for a spike whose job is to test the
embedding-model hypothesis, not to add new infrastructure.

## What was compared

Same method as #31, same six questions, same two chunking strategies, only
the embedding model changed:

1. **Fixed-size** — `RecursiveCharacterTextSplitter(chunk_size=400,
   chunk_overlap=50)` over the flattened body text (2091 chunks).
2. **Structure-aware** — one chunk per סעיף (section), TOC-title-matched
   boundaries, 328 chunks, each carrying a `סעיף <number>: <title>` citation.

## Results

### Embedding-based retrieval (bge-m3, top-4)

| Strategy | Hit rate | #31's nomic-embed-text hit rate |
|---|---|---|
| Structure-aware (n=328) | **6/6** | 0/6 |
| Fixed-size 400/50 (n=2091) | **5/6** | 0/6 |

Every one of the 6 questions retrieved its correct section in the top-4 under
structure-aware chunking; fixed-size chunking missed only one ("מהו המס הנוסף
על הכנסות גבוהות" / the surtax on high incomes — the correct clause, סעיף
ב121, wasn't among the fixed-size top-4 for that question, though it did
appear as a correct hit on the *other* five and on the structure-aware run).

### Lexical word-overlap baseline (top-4, embedding-independent, unchanged from #31)

| Strategy | Hit rate |
|---|---|
| Structure-aware (n=328) | 2/6 |
| Fixed-size 400/50 (n=2091) | 1/6 |

These numbers are reproduced unchanged from #31 (the lexical baseline doesn't
depend on the embedding model) and are included only to reconfirm the
run — they weren't re-derived as new evidence here.

### Chunk volume/shape

Unchanged from #31 (chunking logic wasn't touched): 328 structure-aware
chunks, 2091 fixed-size chunks, 328/426 (77%) TOC-title boundary-location
recall. See `chunking-embedding-findings.md` for the caveats on that
recall figure — they still apply and are orthogonal to embedding choice.

## Interpretation

This is a clean confirmation of #31's diagnosis: the embedding model, not the
chunking strategy, was the dominant failure mode. Swapping only the embedding
model — same chunks, same script, same questions — took both strategies from
0/6 to 5/6 or 6/6. That the *ranking between chunking strategies* (structure-
aware ≥ fixed-size) held up under bge-m3 (6/6 vs 5/6) and matched the
direction of #31's embedding-independent lexical baseline (2/6 vs 1/6) is a
second, independent confirmation that structure-aware chunking is the better
choice, now demonstrated with a working embedding model rather than only via
the lexical proxy.

bge-m3's near-perfect scores here should be read as "good enough to stop
blocking on embedding-model choice," not as "flawless" — 6 questions is a
small eval set, and the one fixed-size miss shows there's still room for a
larger eval before treating this as fully validated.

## Recommendation

**Adopt `bge-m3` (via Ollama) as the project's embedding model**, replacing
`nomic-embed-text`. Concretely: set `OLLAMA_EMBEDDING_MODEL=bge-m3` in `.env`
and run `ollama pull bge-m3` wherever this project runs — no code changes are
needed beyond that, since the existing Ollama embedding-provider path already
takes the model name from configuration (issue #28).

**Chunking strategy: reconfirm structure-aware (one chunk per סעיף)** as
recommended in #31 — now with a working embedding model backing that
recommendation (6/6 vs 5/6) in addition to the lexical baseline (2/6 vs 1/6).
The open item from #31 about hardening section-boundary detection (replacing
title-text matching with a position/bounding-box–based anchor) still stands
and is unrelated to this spike.

**Not further blocked on embedding model.** Unlike #31, which left the
embedding-model question explicitly unresolved pending either
`text-embedding-3-small` or an alternative, this spike gives a usable
answer without a paid API: `bge-m3` clears the bar this project needs. If a
future need arises to squeeze out the one remaining fixed-size miss, the
next things worth trying (in order of cost) are: (a) increasing top-k, (b) a
larger/less-noisy eval set to check whether 5/6 and 6/6 are robust or an
artifact of only 6 questions, then (c) `text-embedding-3-small` if the
budget situation in #37 changes.
