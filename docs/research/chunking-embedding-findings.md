# Chunking and embedding findings: Income Tax Ordinance

Research spike for issue #31 (child of map #29), building on the extraction
findings from #30 (`research/pdf-extraction` branch). Question: given the
real extracted text of the Income Tax Ordinance, does the triage agent's
existing KnowledgeBase tuning (`chunk_size=400/chunk_overlap=50`,
`RecursiveCharacterTextSplitter`, `text-embedding-3-small`) retrieve
relevant clauses well for this document, or does it need different chunk
sizing, an alternative embedding model, or structure-aware splitting?

Scratch script used to produce these findings: `scripts/scratch_chunk_embed_compare.py`
(this branch only). Run with:

```
uv run --with pymupdf --with langchain-text-splitters --with langchain-ollama \
    --with langchain-core python scripts/scratch_chunk_embed_compare.py
```

All numbers below are the actual output of that run against the real PDF at
`.research-data/ordinance.pdf`, not estimates.

## What was compared

Two chunking strategies over the same extracted body text (pages 11-277,
per #30's page-range findings):

1. **Fixed-size** — `RecursiveCharacterTextSplitter(chunk_size=400,
   chunk_overlap=50)`, i.e. the triage agent's existing `tools.py` tuning,
   applied as-is to the flattened body text with no structural awareness.
2. **Structure-aware** — one chunk per סעיף (section), with boundaries
   located by matching each TOC entry's *title text* against the body (not
   a digit-based section-number regex — see "Boundary-location recall"
   below for why), each chunk prefixed with a `סעיף <number>: <title>`
   citation line.

Both chunk sets were embedded and queried against 6 representative Hebrew
payroll-accountant questions (e.g. "what tax rates apply to an individual's
taxable income", "by when must an annual report be filed"), each paired
with a distinctive phrase from the section that actually answers it
(verified by manually reading the extracted body text).

**Only `nomic-embed-text` (via Ollama) could be tested live.** This
environment has no `OPENAI_API_KEY` set — `.env` is configured for
`EMBEDDING_PROVIDER=ollama` only — so `text-embedding-3-small`, the model
the existing KnowledgeBase config uses, could not be compared directly.
`nomic-embed-text` is a 137M-parameter, English-centric embedding model; it
was the only one available in this environment.

To get a signal on chunking quality independent of embedding-model
quality, the script also ran a lexical (Hebrew word-tokenized Jaccard
overlap) ranking baseline over the same two chunk sets — not a proposed
production retrieval method, just a sanity check that doesn't depend on
`nomic-embed-text` at all.

## Results

### Chunk volume/shape

- TOC entries parsed: 426
- Section boundaries located in body text: 328/426 (77% recall — see
  caveats below)
- Structure-aware chunks produced: 328
- Fixed-size chunks produced (400/50): 2091
- Structure-aware chunk length: min=53, max=4047, avg=1505 characters (the
  400-char cap doesn't apply to structure-aware chunks; a real section can
  run long, and a spike-level 4000-char safety cap was needed for outliers
  from boundary-location errors — see below)

### Embedding-based retrieval (nomic-embed-text, top-4)

| Strategy | Hit rate |
|---|---|
| Structure-aware (n=328) | **0/6** |
| Fixed-size 400/50 (n=2091) | **0/6** |

Neither strategy retrieved the correct section in its top-4 results for
*any* of the 6 questions, using `nomic-embed-text`.

### Lexical word-overlap baseline (top-4, embedding-independent)

| Strategy | Hit rate |
|---|---|
| Structure-aware (n=328) | **2/6** |
| Fixed-size 400/50 (n=2091) | **1/6** |

The lexical baseline — which has zero embedding-model involvement — did
strictly better than the embedding-based search on both chunk sets, and
structure-aware chunking outperformed fixed-size chunking here (2/6 vs.
1/6). This is the cleanest signal the spike produced: **`nomic-embed-text`
is not doing better than dumb word-overlap on this corpus, and in the one
apples-to-apples comparison available (lexical ranking), structure-aware
chunks retrieve better than fixed-size chunks.**

### Interpreting the 0/6 on embedding-based search

A 0/6 hit rate for *both* strategies under `nomic-embed-text` is a finding
about the **embedding model**, not primarily about chunking: if chunking
strategy mattered but the embedding model were adequate, we'd expect at
least a gap between the two strategies' embedding-based scores, similar to
the gap the lexical baseline shows. Instead both are equally at floor. The
most plausible explanation is that `nomic-embed-text` — small and trained
predominantly on English text — doesn't produce embeddings that
meaningfully discriminate dense Hebrew legal prose: similarity rankings
looked close to noise relative to the actual query. This can't be
distinguished from "these are genuinely hard questions" with only one
embedding model tested, which is exactly why testing
`text-embedding-3-small` is called out as open below.

## Boundary-location recall and its caveats

328 of 426 TOC-title lookups (77%) found a matching occurrence in the body
text. This is lower than #30's clean 449-section regex count for a specific
reason documented in the script and worth restating: TOC-title matching is
a **fragile placeholder technique**, not a real anchor.

- 426 (TOC entries the parser extracted) vs. #30's 449 (regex hits for
  `^[א-ת]?\d+\s*סעיף` over body text) don't match because they're counting
  different things — the TOC-entry regex used here captures rows of a
  specific `<num> סעיף <title>Go<page>` shape, and a number of TOC rows
  didn't parse cleanly into that shape (or captured a chapter/part header
  as if it were a section row) and were silently dropped before boundary
  location was even attempted.
- Of the 426 parsed TOC entries, 98 (23%) had no matching title occurrence
  found forward-searching through the body — the script explicitly skips
  these, per its own comment, because "this can happen for very short/
  generic titles or TOC-only rows."
- Short, generic titles are a real failure mode, not a hypothetical one:
  the embedding-based results above show several structure-aware chunks
  retrieved under clearly wrong citations (e.g. "סעיף ז75: הוראות כלליות" —
  "General Provisions" — a title generic enough to plausibly false-match
  elsewhere, or to be a genuinely poor anchor for retrieval regardless).
  The max chunk length outlier (4047 chars, capped from what would
  otherwise be longer) is exactly the failure mode the script's comments
  predict: a short/generic title matches too early and swallows everything
  up to the next *correctly*-matched title.
- This confirms #30's own recommendation: title-text matching was used
  here only because digit-based section-number regex is unreliable in body
  text due to the BiDi digit-run quirk #30 documented. A real
  implementation needs a sturdier anchor than "does this title string
  literally recur" — e.g. word-level bounding-box position data (which
  PyMuPDF exposes via `get_text("dict")`, per #30) to locate section-number
  tokens by position rather than by digit-regex-over-flattened-text or
  title-string-matching.

## Recommendation

**Chunking strategy: structure-aware (one chunk per סעיף), not fixed-size
400/50.** Even with a fragile, spike-level boundary-location technique
(77% recall, known failure modes above), structure-aware chunking beat
fixed-size chunking on the one evaluation method available for genuine
apples-to-apples comparison (lexical baseline: 2/6 vs. 1/6), produces
chunks with real section citations, and matches #30's own recommendation
that section/chapter boundaries are the right primary structuring unit for
this document. The existing triage agent's 400/50 fixed-size tuning was
built for short markdown FAQ docs and has no evidence of working for this
corpus — it produced 2091 undifferentiated, citation-less chunks and its
overlap window is far too small relative to average section length (1505
chars) to keep clause context together. A real implementation should
replace title-text matching with a position-based (bounding-box) section
boundary detector rather than carrying the current approach's known
fragility forward.

**Embedding model: unresolved, needs a resolution before shipping.**
`nomic-embed-text` scored 0/6 for both chunking strategies on live
embedding-based retrieval — at or below the level of a naive lexical
word-overlap baseline running on the same chunks. That is not a usable
result for a legal Q&A agent regardless of chunking strategy. This spike
cannot distinguish "the embedding model is inadequate for Hebrew legal
text" from "these six questions are unusually hard" with only one model
tested. **Recommend testing `text-embedding-3-small` (the triage agent's
existing embedding choice) once an `OPENAI_API_KEY` is available in this
environment**, using this same script's structure-aware chunk set and
question list, before deciding on a final embedding model. If
`text-embedding-3-small` also underperforms on Hebrew, a
multilingual-legal-domain embedding model should be evaluated next rather
than defaulting to the triage agent's toy-doc choice.
