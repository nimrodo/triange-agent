# PDF extraction findings: Income Tax Ordinance [New Version]

Research spike for issue #30 (child of map #29). Source document: gov.il's
consolidated (unofficial, "לא מרובד"/not layered — i.e. not an official
government publication) Hebrew text of the Income Tax Ordinance, fetched
from the URL in the issue. All findings below are from directly inspecting
the real PDF, not from documentation about the tools.

Scratch script used to reproduce these findings: `scratch_pdf_probe.py`
(repo root, this branch only — downloads the PDF into `.research-data/`,
gitignored). Run with:

```
uv run --with pymupdf --with pdfplumber python scratch_pdf_probe.py
```

## Document facts

- 312 pages, PDF 1.4, produced by Chrome's print-to-PDF (`Producer: Skia/PDF
  m108`), no PDF outline/bookmarks (`get_toc()` returns empty), no document
  title/author metadata.
- Single column throughout (body text block roughly x=90–480 on a ~595pt-wide
  page); no multi-column layout anywhere I sampled.
- Structure, front to back:
  - Pages 0–10: an 11-page table of contents. Every row is `<num> סעיף
    <title>Go<page>`, one row per section, with GoTo link annotations
    (`nameddest` values like `Seif1`, `med0`) pointing at the body page.
  - Pages ~11–277: the operative law body — parts (חלק), chapters (פרק,
    named with Hebrew ordinals: "פרק ראשון", "פרק שני", not "פרק 1"),
    sub-divisions (סימן), and sections (סעיף N).
  - Pages ~278–289 (approx.): a second appendix, "לוח ההשוואה לסעיפי הנוסח
    הישן" (comparison table to the old-version section numbers) — a
    section-renumbering crosswalk, likely tabular.
  - Pages ~290–311: a dense legislative-history appendix, not per-page
    footnotes — every amendment across the whole document is re-listed here
    with its official citation (ס"ח = Sefer HaChukim/Official Gazette
    number+page+date, ה"ח = Knesset bill number, effective-date and
    transitional-provision notes). This is a distinct corpus from the
    operative text and should probably be excluded from (or indexed
    separately from) the primary Q&A corpus — it's citation metadata, not
    legal rules a payroll accountant would ask about directly.
- No true footnotes/endnotes markers (no superscript numbers in the body);
  amendment history is instead inlined directly under each amended
  paragraph as its own line(s), e.g. `(תיקון מס' 232 תשע"ז-2016)`, sometimes
  several stacked for one clause. Regex over PyMuPDF's `get_text()` finds
  2302 such amendment-note lines.
- Repealed clauses are kept as placeholders, e.g. `(ג)   (בוטל)` ("(c)
  (repealed)"), rather than being removed — numbering stays stable across
  amendments, which is good for citation stability but means "clause exists"
  and "clause has content" are different things.
- Sub-lettered sections are common: `א2`, `ב8`, `א244`, etc. (Hebrew-letter
  suffix = "2a", "8b", "244a"). Because of RTL+embedded-LTR-digit
  bidirectional layout, these can render as `2 סעיף א` visually reordered
  in some contexts — see BiDi gotchas below.
- A recurring garbage line (`ל`, `ל לל`, `ללל`) appears at the very bottom of
  some (not all) sampled pages, most likely a broken/partial glyph from a
  footer watermark or logo whose font has no proper ToUnicode CMap. It's
  cheap to strip (isolated single-Hebrew-letter runs near the page-bottom
  margin) but chunking code should not assume every page's last line is
  real content.

## Extraction approach comparison

Tried on the real file: **PyMuPDF (`pymupdf`/`fitz`)** vs **pdfplumber**.

### PyMuPDF — reading order is correct out of the box

`page.get_text()` returns properly reordered, logical-order Hebrew, e.g.
page 0 starts with:

```
פקודת מס הכנסה [נוסח חדש]
```

— readable Hebrew, right-to-left, punctuation/brackets in the correct
logical direction. This is because PyMuPDF applies Unicode BiDi reordering
internally before returning text. `get_text("blocks")`/`get_text("dict")`
also expose per-block bounding boxes, which is what you'd want for
reconstructing structure by position rather than trusting one giant string.

Simple regex over `get_text()` per page correctly identifies:
- 449 lines matching a section-heading pattern (`^[א-ת]?\d+\s*סעיף`)
- 193 lines matching a chapter/part-heading pattern (`פרק\s+\S+`)
- 2302 amendment-note lines (`תיקון מס`)

**Caveat found while testing:** correctness holds for lines that are pure
Hebrew prose or a single embedded LTR run (a date, a currency amount). But
lines packing *multiple* LTR "islands" next to each other — e.g. two
back-to-back amendment citations like `(תיקון מס' 22 תשל"ה-1975) (תיקון מס'
29 תשל"ח-1978)` — sometimes come out of `get_text()` with the two citations'
digit-runs interleaved rather than in true left-to-right-then-right-to-left
reading order, even though the same content renders correctly when viewed
in a PDF viewer. This is a genuine BiDi-with-multiple-runs limitation, not a
bug in the tool — it means: rely on `get_text()` for prose and section
boundaries (works well), but for parsing amendment-citation metadata
specifically, don't trust naive regex over the flattened string; either
handle the amendment-history appendix (which lists everything in one clean,
consistent format) as the canonical source for citation data, or work from
word-level bounding boxes for citation extraction rather than raw text.

### pdfplumber — RTL is broken out of the box

`page.extract_text()` returns text in raw glyph-placement order with no
BiDi reordering applied. The same page-0 title extracts as:

```
[שדח חסונ] הסנכה סמ תדוקפ
```

which is close to (but not exactly) the character-reversal of the correct
string — reversing the whole line gives `פקודת מס הכנסה ]נוסח חדש[`, which
gets the Hebrew right but flips the bracket direction wrong (brackets need
BiDi-aware mirroring, not naive reversal). Getting correct text out of
pdfplumber for this document would require a real BiDi pass (e.g.
`python-bidi`) per extracted line, plus care around punctuation/brackets —
extra work that PyMuPDF already does for free. I did not pursue a
pdfplumber+python-bidi pipeline further since PyMuPDF already solves the
problem natively.

`unstructured` was not tried — given PyMuPDF already produces clean logical
text and per-block coordinates, and the document is single-column plain
prose (no tables/images needing unstructured's specialized partitioning),
it's unlikely to add value here and wasn't worth the extra dependency for
this spike.

## Recommendation

**Use PyMuPDF (`pymupdf`) for extraction.** It handles Hebrew RTL BiDi
reordering correctly natively — pdfplumber does not, and doesn't have this
project need a table/image-aware tool like `unstructured`. Structure-aware
parsing (splitting into chapter/section/clause units for citation) is
feasible and worth doing as a follow-up:

- Section/chapter boundaries are cleanly recoverable via line-anchored
  regex over `get_text()` output (`^[א-ת]?\d+\s*סעיף` for sections, `פרק
  \S+` for chapters) — this held up well in testing (449 / 193 matches with
  no obvious false positives spot-checked).
- Don't split purely on generic chunk_size — a structure-first pass
  (segment by סעיף/פרק boundaries, keep amendment-note lines attached to
  their clause) will give citations that actually mean something to a
  payroll accountant ("סעיף 9(5)" not "chunk 47"), which is exactly the
  next-ticket concern (#31, chunking/embedding strategy).
- Treat the front-matter TOC (pages 0–10), the renumbering crosswalk
  appendix, and the legislative-history appendix (pages ~290–311)
  as distinct zones: the TOC is redundant with the body and can likely be
  dropped or used only to build a section→page index; the history appendix
  is citation metadata that probably shouldn't be embedded as if it were
  operative legal text.
- Repealed clauses (`(בוטל)`) should be kept in the section index (for
  citation completeness / "why can't I find this section" answers) but
  excluded from retrieval since they have no substantive content.

## Scratch code

`scratch_pdf_probe.py` at the repo root (this branch only) reproduces the
above: downloads the PDF, prints sample-page extracts from both libraries
side by side, and runs the section/chapter/amendment regex counts.
