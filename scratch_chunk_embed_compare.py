"""Research spike for issue #31: compare chunking/embedding strategies for
retrieval quality over the real Income Tax Ordinance text.

Builds on the extraction findings from #30 (research/pdf-extraction branch):
PyMuPDF gives correctly-ordered logical Hebrew for prose and for the table of
contents, but section-number tokens in the *body* text can end up in a
surprising position within a flattened line (a BiDi digit-run quirk related
to, but distinct from, the interleaved-citation issue #30 already flagged).
That makes naive "regex for a leading section number" unreliable as a body
chunk-boundary detector. Instead, this script locates section boundaries by
matching each TOC entry's *title text* (pure Hebrew prose, orders correctly)
against the body -- the title string appears verbatim right before each
section's numbered clause.

Compares two chunking strategies:
  1. "fixed" -- RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50),
     the triage agent's existing KnowledgeBase tuning (src/triange_agent/tools.py),
     applied to the flattened body text.
  2. "structure" -- one chunk per סעיף (section), using TOC-title-matched
     boundaries, each chunk prefixed with its recovered chapter + section
     citation so retrieval results carry a meaningful citation.

Embeds both chunk sets with Ollama's nomic-embed-text (the only embedding
model available in this environment -- no OPENAI_API_KEY is set, see
docs/research/chunking-embedding-findings.md for why text-embedding-3-small
could not be compared live) and runs a handful of realistic payroll-
accountant questions against both, reporting which strategy's top-k results
actually contain the section that answers the question.

Usage:
    uv run --with pymupdf --with langchain-text-splitters --with langchain-ollama \
        --with langchain-core python scratch_chunk_embed_compare.py

Requires a local Ollama server with nomic-embed-text pulled (already the
case in this environment per .env: EMBEDDING_PROVIDER=ollama).
"""

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = Path(__file__).parent / ".research-data" / "ordinance.pdf"
# Swapped for issue #39: nomic-embed-text (issue #31) scored 0/6 on both
# chunking strategies, attributed to weak Hebrew/multilingual support.
# bge-m3 (BAAI, via Ollama) explicitly reports Hebrew ("he") results in its
# MKQA cross-lingual retrieval table and claims 100+ working languages --
# see docs/research/embedding-model-comparison-findings.md for citations.
EMBEDDING_MODEL = "bge-m3"

# Body runs ~pages 11-277 per #30's findings; TOC is 0-10, appendices after.
TOC_PAGE_RANGE = range(11)
BODY_PAGE_RANGE = range(11, 278)

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

TOC_ENTRY_RE = re.compile(
    r"(?P<num>[א-ת]?\d+)\s*סעיף\s*\n(?P<title>.+?)Go\s*\n(?P<page>\d+)",
    re.DOTALL,
)
CHAPTER_RE = re.compile(r"פרק\s+\S+[^\n]*")


@dataclass
class SectionBoundary:
    number: str
    title: str
    body_offset: int  # character offset into the concatenated body text


def load_pages(doc: pymupdf.Document, page_range: range) -> list[str]:
    return [doc[i].get_text() for i in page_range]


def parse_toc_entries(toc_pages: list[str]) -> list[tuple[str, str]]:
    """Return ordered (section_number, title) pairs from the TOC."""
    toc_text = "\n".join(toc_pages)
    entries = []
    for m in TOC_ENTRY_RE.finditer(toc_text):
        title = " ".join(m.group("title").split())
        # Titles occasionally swallow a following chapter/part header line;
        # keep only the first line-ish chunk before any obvious next marker.
        if title:
            entries.append((m.group("num"), title))
    return entries


def locate_section_boundaries(
    body_text: str, toc_entries: list[tuple[str, str]]
) -> list[SectionBoundary]:
    """Find each TOC title's first occurrence in the body, in TOC order,
    searching forward from the previous match so repeated title text
    (rare but possible) resolves to sequential occurrences."""
    boundaries: list[SectionBoundary] = []
    search_from = 0
    for num, title in toc_entries:
        idx = body_text.find(title, search_from)
        if idx == -1:
            # Title not found forward of the last match -- skip; this can
            # happen for very short/generic titles or TOC-only rows (e.g.
            # a chapter title captured as a "section" by the regex).
            continue
        boundaries.append(SectionBoundary(number=num, title=title, body_offset=idx))
        search_from = idx + len(title)
    return boundaries


def build_structure_aware_chunks(
    body_text: str, boundaries: list[SectionBoundary]
) -> list[Document]:
    docs = []
    for i, b in enumerate(boundaries):
        end = (
            boundaries[i + 1].body_offset if i + 1 < len(boundaries) else len(body_text)
        )
        content = body_text[b.body_offset : end].strip()
        if not content or "(בוטל)" in content[:40]:
            # Skip empty/repealed-only sections -- no substantive content
            # to retrieve, per #30's repealed-clause finding.
            continue
        citation = f"סעיף {b.number}: {b.title}"
        # A handful of boundaries are mis-located (a short/generic title
        # text matches earlier than its real section, swallowing everything
        # up to the next correctly-matched title) -- cap chunk length as a
        # spike-level safety net so those outliers don't blow past the
        # embedding model's context window. See findings doc for why a real
        # implementation needs a sturdier anchor than title-text matching.
        max_content_chars = 4000
        content = content[:max_content_chars]
        docs.append(
            Document(
                page_content=f"{citation}\n{content}",
                metadata={"source": citation, "section": b.number},
            )
        )
    return docs


def build_fixed_size_chunks(body_text: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.create_documents(
        [body_text], metadatas=[{"source": "ordinance.pdf"}]
    )


# Representative payroll-accountant questions, each paired with a
# distinctive phrase from the section that actually answers it (verified by
# manually reading the extracted body text -- see
# docs/research/chunking-embedding-findings.md). Matching on a content
# phrase, not a section number, lets the same ground truth check both
# structure-aware chunks (which carry an explicit citation) and fixed-size
# chunks (which don't have section metadata at all, only raw text).
QUESTIONS = [
    ("מה שיעורי המס החלים על הכנסתו החייבת של יחיד?", ["שיעור המס ליחיד"]),
    ("עד איזה תאריך יש להגיש דוח שנתי לפקיד השומה?", ["באפריל של כל שנה"]),
    ("האם ניתן לדחות את מועד הגשת הדוח השנתי?", ["דחיית המועד", "לדחות את"]),
    ("מהו המס הנוסף על הכנסות גבוהות?", ["מס על הכנסות גבוהות"]),
    ("מי חייב בהגשת הודעה על פתיחת עסק חדש?", ["הודעה על התחלת התעסקות"]),
    ("כיצד ממוסה הכנסה מהשכרת דירת מגורים?", ["השכרת דירת מגורים"]),
]


def build_store_in_batches(
    docs: list[Document], embeddings: OllamaEmbeddings, batch_size: int = 16
) -> InMemoryVectorStore:
    """Ollama's local embedding server chokes on very large single batches
    (observed connection resets around ~2000 texts in one call) -- batch
    the embedding calls to keep each request small."""
    store = InMemoryVectorStore(embeddings)
    for start in range(0, len(docs), batch_size):
        store.add_documents(docs[start : start + batch_size])
    return store


def evaluate(name: str, vector_store: InMemoryVectorStore, k: int = 4) -> None:
    print(f"\n=== {name} (embedding: {EMBEDDING_MODEL}) ===")
    hits = 0
    for question, expected_phrases in QUESTIONS:
        results = vector_store.similarity_search(question, k=k)
        found = any(
            any(phrase in r.page_content for phrase in expected_phrases)
            for r in results
        )
        hits += found
        marker = "HIT " if found else "MISS"
        print(f"[{marker}] {question}")
        print(f"       expected phrase(s): {expected_phrases}")
        for r in results:
            src = r.metadata.get(
                "source", "ordinance.pdf (fixed-size, no section metadata)"
            )
            print(f"       -> {src}")
    print(
        f"\n{name}: {hits}/{len(QUESTIONS)} questions had the right section in top-{k}"
    )


# --- Lexical (word-overlap) baseline -------------------------------------
# nomic-embed-text (137M params, English-centric training) is the only
# embedding model available in this environment -- no OPENAI_API_KEY, see
# findings doc. It turned out to barely discriminate Hebrew legal text at
# all (see findings doc: the true section ranked ~median, not top-k). That
# tells us about the embedding model, not about chunking. To still get a
# signal on chunk *boundaries* independent of embedding quality, this is a
# crude Hebrew-tokenized word-overlap (Jaccard) ranker -- not a proposed
# production retrieval method, just a sanity check.
_WORD_RE = re.compile(r"[א-ת]+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text))


def lexical_rank_evaluate(name: str, docs: list[Document], k: int = 4) -> None:
    print(f"\n=== {name} (lexical word-overlap baseline, embedding-independent) ===")
    hits = 0
    doc_tokens = [(_tokenize(d.page_content), d) for d in docs]
    for question, expected_phrases in QUESTIONS:
        q_tokens = _tokenize(question)
        scored = sorted(
            doc_tokens,
            key=lambda dt: len(dt[0] & q_tokens) / (len(dt[0] | q_tokens) or 1),
            reverse=True,
        )
        top = [d for _, d in scored[:k]]
        found = any(
            any(phrase in d.page_content for phrase in expected_phrases) for d in top
        )
        hits += found
        marker = "HIT " if found else "MISS"
        sources = [d.metadata.get("source", "ordinance.pdf (fixed-size)") for d in top]
        print(f"[{marker}] {question} -> {sources}")
    print(
        f"\n{name}: {hits}/{len(QUESTIONS)} questions had the right section in top-{k}"
    )


def main() -> None:
    doc = pymupdf.open(PDF_PATH)
    toc_pages = load_pages(doc, TOC_PAGE_RANGE)
    body_pages = load_pages(doc, BODY_PAGE_RANGE)
    body_text = "\n".join(body_pages)

    toc_entries = parse_toc_entries(toc_pages)
    print(f"Parsed {len(toc_entries)} TOC entries")

    boundaries = locate_section_boundaries(body_text, toc_entries)
    print(
        f"Located {len(boundaries)}/{len(toc_entries)} section boundaries in body text "
        f"({len(boundaries) / len(toc_entries):.0%} recall via title-text matching)"
    )

    structure_chunks = build_structure_aware_chunks(body_text, boundaries)
    fixed_chunks = build_fixed_size_chunks(body_text)
    print(f"Structure-aware chunks: {len(structure_chunks)}")
    print(f"Fixed-size chunks ({CHUNK_SIZE}/{CHUNK_OVERLAP}): {len(fixed_chunks)}")

    lengths = [len(c.page_content) for c in structure_chunks]
    print(
        f"Structure-aware chunk length: min={min(lengths)} max={max(lengths)} "
        f"avg={sum(lengths) / len(lengths):.0f}"
    )

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    structure_store = build_store_in_batches(structure_chunks, embeddings)
    fixed_store = build_store_in_batches(fixed_chunks, embeddings)

    evaluate(
        f"structure-aware (one chunk/section, n={len(structure_chunks)})",
        structure_store,
    )
    evaluate(
        f"fixed-size ({CHUNK_SIZE}/{CHUNK_OVERLAP}, n={len(fixed_chunks)})", fixed_store
    )

    lexical_rank_evaluate(
        f"structure-aware (one chunk/section, n={len(structure_chunks)})",
        structure_chunks,
    )
    lexical_rank_evaluate(
        f"fixed-size ({CHUNK_SIZE}/{CHUNK_OVERLAP}, n={len(fixed_chunks)})",
        fixed_chunks,
    )


if __name__ == "__main__":
    main()
