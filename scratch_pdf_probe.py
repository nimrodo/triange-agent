"""Research spike for issue #30: probe extraction of the Income Tax Ordinance PDF.

Throwaway script (not production code, no tests) to compare pdfplumber vs
PyMuPDF on the real gov.il PDF for reading order, RTL correctness, and
structural markers (chapter/section headings, amendment notes).

Usage:
    uv run --with pymupdf --with pdfplumber python scratch_pdf_probe.py

Downloads the PDF into .research-data/ (gitignored) on first run.
"""

import re
import urllib.request
from pathlib import Path

PDF_URL = (
    "https://www.gov.il/BlobFolder/legalinfo/law_pkudat_mas_hachnasa/he/"
    "LegalInformation_kesher_%D7%A4%D7%A7%D7%95%D7%93%D7%AA%20%D7%9E%D7%A1%20"
    "%D7%94%D7%9B%D7%A0%D7%A1%D7%94%20%5B%D7%A0%D7%95%D7%A1%D7%97%20%D7%97%D7%93"
    "%D7%A9%5D%20-%20%D7%9C%D7%90%20%D7%9E%D7%A8%D7%95%D7%91%D7%93.pdf"
)
DATA_DIR = Path(__file__).parent / ".research-data"
PDF_PATH = DATA_DIR / "ordinance.pdf"

SAMPLE_PAGES = [0, 39, 100, 200, 300]  # 0-indexed: TOC, body, body, body, amendment appendix

SECTION_HEADING_RE = re.compile(r"^([א-ת]?\d+)\s*סעיף")
CHAPTER_RE = re.compile(r"פרק\s+\S+")
AMENDMENT_RE = re.compile(r"תיקון מס")


def ensure_pdf() -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    if not PDF_PATH.exists():
        req = urllib.request.Request(PDF_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            PDF_PATH.write_bytes(resp.read())
    return PDF_PATH


def probe_pymupdf(path: Path) -> None:
    import pymupdf

    doc = pymupdf.open(path)
    print(f"\n=== PyMuPDF: {doc.page_count} pages ===")
    for i in SAMPLE_PAGES:
        text = doc[i].get_text()
        print(f"--- page {i} (first 200 chars) ---")
        print(text[:200].replace("\n", " | "))
    headings = 0
    chapters = 0
    amendments = 0
    for page in doc:
        for line in page.get_text().splitlines():
            if SECTION_HEADING_RE.match(line):
                headings += 1
            if CHAPTER_RE.search(line):
                chapters += 1
            if AMENDMENT_RE.search(line):
                amendments += 1
    print(f"regex-detected section headings: {headings}, chapter headings: {chapters}, "
          f"amendment notes: {amendments}")


def probe_pdfplumber(path: Path) -> None:
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        print(f"\n=== pdfplumber: {len(pdf.pages)} pages ===")
        for i in SAMPLE_PAGES:
            text = pdf.pages[i].extract_text() or ""
            print(f"--- page {i} (first 200 chars, raw) ---")
            print(text[:200].replace("\n", " | "))
        # Demonstrate the RTL problem: naive extraction is glyph-order, not
        # logical order -- reversing the whole line gets close but mangles
        # bracket/punctuation direction, showing real BiDi handling is needed.
        first_line = (pdf.pages[0].extract_text() or "").splitlines()[0]
        print(f"\nraw first line:      {first_line!r}")
        print(f"naive reversed line: {first_line[::-1]!r}")


if __name__ == "__main__":
    pdf_path = ensure_pdf()
    probe_pymupdf(pdf_path)
    probe_pdfplumber(pdf_path)
