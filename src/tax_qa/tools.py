import re
from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool, tool
from langchain_core.vectorstores import InMemoryVectorStore

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

REPEALED_MARKER = "(בוטל)"

_LONE_LETTER_RE = re.compile(r"^[א-ת]$")
_LEADING_LETTER_RE = re.compile(r"^([א-ת])\.\s")


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
    body_text = "".join(span["text"] for span in spans if span["font"] != CLAUSE_NUMBER_FONT)
    match = _LEADING_LETTER_RE.match(body_text)
    return match.group(1) if match else ""


def _locate_clause_boundaries(
    pdf_doc: pymupdf.Document, body_page_range: range
) -> tuple[list[str], list[tuple[int, str]]]:
    """Split the page range into lines and record, for each line, whether it
    opens a new Clause. Returns (lines, boundaries) where boundaries are
    (index into lines, clause number) pairs in document order."""
    lines: list[str] = []
    boundaries: list[tuple[int, str]] = []
    for page_index in body_page_range:
        if page_index >= pdf_doc.page_count:
            break
        page_dict = pdf_doc[page_index].get_text("dict")
        for block in page_dict["blocks"]:
            for line in block.get("lines", []):
                spans = line["spans"]
                clause_spans = [s for s in spans if s["font"] == CLAUSE_NUMBER_FONT]
                if clause_spans:
                    number = clause_spans[0]["text"].strip() + _clause_suffix(spans)
                    boundaries.append((len(lines), number))
                lines.append("".join(span["text"] for span in spans))
    return lines, boundaries


def _disambiguate(numbers: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result = []
    for number in numbers:
        seen[number] = seen.get(number, 0) + 1
        result.append(number if seen[number] == 1 else f"{number}-{seen[number]}")
    return result


def extract_clauses(pdf_path: Path, body_page_range: range = BODY_PAGE_RANGE) -> list[Document]:
    pdf_doc = pymupdf.open(pdf_path)
    lines, boundaries = _locate_clause_boundaries(pdf_doc, body_page_range)
    numbers = _disambiguate([number for _, number in boundaries])

    clauses = []
    for i, (start, _) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(lines)
        content = "\n".join(lines[start:end]).strip()
        if REPEALED_MARKER in content[:40]:
            continue
        clauses.append(Document(page_content=content, metadata={"source": f"סעיף {numbers[i]}"}))
    return clauses


def build_vector_store(
    pdf_path: Path,
    embeddings: Embeddings,
    body_page_range: range = BODY_PAGE_RANGE,
) -> InMemoryVectorStore:
    clauses = extract_clauses(pdf_path, body_page_range)
    return InMemoryVectorStore.from_documents(clauses, embeddings)


def retrieve_clauses(
    vector_store: InMemoryVectorStore, query: str, k: int = 4
) -> list[RetrievedClause]:
    results = vector_store.similarity_search_with_score(query, k=k)
    return [
        RetrievedClause(content=doc.page_content, source=doc.metadata["source"], score=score)
        for doc, score in results
    ]


def format_clauses(clauses: list[RetrievedClause]) -> str:
    return "\n\n".join(f"[{c.source}] {c.content}" for c in clauses)


def make_search_tool(vector_store: InMemoryVectorStore, k: int = 4) -> BaseTool:
    @tool(response_format="content_and_artifact")
    def search_tax_ordinance(query: str) -> tuple[str, list[RetrievedClause]]:
        """Search the Income Tax Ordinance for Clauses relevant to the query."""
        clauses = retrieve_clauses(vector_store, query, k=k)
        return format_clauses(clauses), clauses

    return search_tax_ordinance
