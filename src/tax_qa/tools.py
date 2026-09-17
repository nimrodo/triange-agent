import re
import time
from pathlib import Path

import pymupdf
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool, tool
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tax_qa.state import RetrievedClause

# The body of the Income Tax Ordinance (TOC and legislative-history appendix
# excluded) -- see docs/research/pdf-extraction-findings.md.
BODY_PAGE_RANGE = range(11, 278)

# The ordinance typesets every clause's opening section number in this
# distinct font/size, separate from body text (FrankRuehl) and amendment
# citations (TimesNewRomanPS-BoldMT). Anchoring on it sidesteps the BiDi
# digit-run repositioning that makes a plain "leading digits" regex over
# flattened text unreliable -- see docs/research/chunking-embedding-findings.md.
CLAUSE_NUMBER_FONT = "Miriam"

# Part (חלק) and chapter (פרק) headings are the one other structural marker
# with a distinct, reliable font signature (FrankRuehl at ~12pt, vs. 13pt
# body text) -- subdivision (סימן) headings are typeset identically to body
# text and aren't reliably detectable this way, so aren't tracked.
STRUCTURE_HEADING_FONT = "FrankRuehl"
STRUCTURE_HEADING_SIZE_RANGE = (11.5, 12.5)

REPEALED_MARKER = "(בוטל)"

_LONE_LETTER_RE = re.compile(r"^[א-ת]$")
_LEADING_LETTER_RE = re.compile(r"^([א-ת])\.\s")


def _is_structure_heading(spans: list[dict]) -> bool:
    return bool(spans) and all(
        span["font"] == STRUCTURE_HEADING_FONT
        and STRUCTURE_HEADING_SIZE_RANGE[0]
        < span["size"]
        < STRUCTURE_HEADING_SIZE_RANGE[1]
        for span in spans
    )


def _clause_suffix(spans: list[dict]) -> str:
    """Best-effort recovery of a Hebrew sub-letter suffix (e.g. the "א" in
    "121א") from the two patterns observed in the real document: a standalone
    single-letter span next to the clause-number glyph, or the clause's own
    body text opening with "<letter>. ". Not exhaustive -- a handful of
    sub-lettered clauses embed the letter elsewhere and fall back to a bare
    number, disambiguated by _disambiguate below rather than silently
    colliding with another clause's citation."""
    for span in spans:
        if span["font"] == CLAUSE_NUMBER_FONT:
            continue
        text = span["text"].strip()
        if _LONE_LETTER_RE.match(text):
            return text
    body_text = "".join(
        span["text"] for span in spans if span["font"] != CLAUSE_NUMBER_FONT
    )
    match = _LEADING_LETTER_RE.match(body_text)
    return match.group(1) if match else ""


def _locate_clause_boundaries(
    pdf_doc: pymupdf.Document, body_page_range: range
) -> tuple[list[str], list[tuple[int, str, str | None, str | None]]]:
    """Split the page range into lines and record, for each line, whether it
    opens a new Clause. Returns (lines, boundaries) where boundaries are
    (index into lines, clause number, current part, current chapter) tuples
    in document order -- part/chapter are whichever heading was last seen,
    carrying the containing structure forward across clauses."""
    lines: list[str] = []
    boundaries: list[tuple[int, str, str | None, str | None]] = []
    current_part: str | None = None
    current_chapter: str | None = None
    for page_index in body_page_range:
        if page_index >= pdf_doc.page_count:
            break
        page_dict = pdf_doc[page_index].get_text("dict")
        for block in page_dict["blocks"]:
            for line in block.get("lines", []):
                spans = line["spans"]
                line_text = "".join(span["text"] for span in spans)
                if _is_structure_heading(spans):
                    if "חלק" in line_text:
                        current_part = line_text.strip()
                        current_chapter = None
                    elif "פרק" in line_text:
                        current_chapter = line_text.strip()
                clause_spans = [s for s in spans if s["font"] == CLAUSE_NUMBER_FONT]
                if clause_spans:
                    number = clause_spans[0]["text"].strip() + _clause_suffix(spans)
                    boundaries.append(
                        (len(lines), number, current_part, current_chapter)
                    )
                lines.append(line_text)
    return lines, boundaries


def _disambiguate(numbers: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result = []
    for number in numbers:
        seen[number] = seen.get(number, 0) + 1
        result.append(number if seen[number] == 1 else f"{number}-{seen[number]}")
    return result


def extract_clauses(
    pdf_path: Path, body_page_range: range = BODY_PAGE_RANGE
) -> list[Document]:
    pdf_doc = pymupdf.open(pdf_path)
    lines, boundaries = _locate_clause_boundaries(pdf_doc, body_page_range)
    numbers = _disambiguate([number for _, number, _, _ in boundaries])

    clauses = []
    for i, (start, _, part, chapter) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(lines)
        content = "\n".join(lines[start:end]).strip()
        if REPEALED_MARKER in content[:40]:
            continue
        metadata = {"source": f"סעיף {numbers[i]}", "part": part, "chapter": chapter}
        clauses.append(Document(page_content=content, metadata=metadata))
    return clauses


# The full ordinance is ~570 Clauses; embedding them in one request risks
# exceeding a local embedding server's batch/context limits (observed with
# Ollama's bge-m3). Batching keeps each request small regardless of provider.
DEFAULT_EMBEDDING_BATCH_SIZE = 100

# One chunk per Clause is the primary retrieval unit -- see
# docs/research/chunking-embedding-findings.md, which found this
# structure-aware approach outperforms indiscriminate fixed-size splitting
# on this document. But a handful of outlier Clauses (e.g. section 9's
# exemptions list) run to tens of thousands of characters: embedded whole,
# they represent too many unrelated topics at once to be a useful match for
# any single query, and flood the LLM's context if retrieved. Only those
# outliers get split, each sub-chunk keeping the parent Clause's citation.
MAX_CLAUSE_CHARS = 2000
CLAUSE_CHUNK_OVERLAP = 200


def split_oversized_clauses(clauses: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CLAUSE_CHARS, chunk_overlap=CLAUSE_CHUNK_OVERLAP
    )
    result = []
    for clause in clauses:
        if len(clause.page_content) > MAX_CLAUSE_CHARS:
            result.extend(splitter.split_documents([clause]))
        else:
            result.append(clause)
    return result


def build_vector_store(
    pdf_path: Path,
    embeddings: Embeddings,
    body_page_range: range = BODY_PAGE_RANGE,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    persist_directory: Path | None = None,
    collection_name: str = "tax-ordinance",
    request_delay_seconds: float = 0.0,
) -> Chroma:
    chunks = split_oversized_clauses(extract_clauses(pdf_path, body_page_range))
    chunk_ids = [f"clause-{i}" for i in range(len(chunks))]

    vector_store = Chroma(
        embedding_function=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_directory) if persist_directory else None,
        # Matches the cosine similarity semantics the rest of the pipeline
        # (the similarity floor in nodes/answer.py) is written against.
        collection_metadata={"hnsw:space": "cosine"},
    )

    # ids are stable across runs (deterministic extraction order), so a build
    # interrupted mid-way (e.g. by a provider rate limit) resumes from
    # whichever chunks are still missing instead of restarting from scratch.
    existing_ids = set(vector_store.get(include=[])["ids"])
    pending = [
        (chunk_id, chunk)
        for chunk_id, chunk in zip(chunk_ids, chunks)
        if chunk_id not in existing_ids
    ]
    for start in range(0, len(pending), batch_size):
        if start > 0 and request_delay_seconds:
            time.sleep(request_delay_seconds)
        batch = pending[start : start + batch_size]
        vector_store.add_documents(
            [chunk for _, chunk in batch], ids=[chunk_id for chunk_id, _ in batch]
        )
    return vector_store


def retrieve_clauses(
    vector_store: VectorStore, query: str, k: int = 4
) -> list[RetrievedClause]:
    # Normalized (0=dissimilar, 1=identical) rather than raw distance, so the
    # similarity floor in nodes/answer.py means the same thing across backends.
    results = vector_store.similarity_search_with_relevance_scores(query, k=k)
    return [
        RetrievedClause(
            content=doc.page_content, source=doc.metadata["source"], score=score
        )
        for doc, score in results
    ]


def format_clauses(clauses: list[RetrievedClause]) -> str:
    return "\n\n".join(f"[{c.source}] {c.content}" for c in clauses)


def make_search_tool(vector_store: VectorStore, k: int = 4) -> BaseTool:
    @tool(response_format="content_and_artifact")
    def search_tax_ordinance(query: str) -> tuple[str, list[RetrievedClause]]:
        """Search the Income Tax Ordinance for Clauses relevant to the query."""
        clauses = retrieve_clauses(vector_store, query, k=k)
        return format_clauses(clauses), clauses

    return search_tax_ordinance
