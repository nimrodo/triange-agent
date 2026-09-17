# Israeli Income Tax Q&A Agent

A LangGraph/LangChain agent that lets a payroll accountant ask natural-language questions in Hebrew about the content of the Israeli Income Tax Ordinance, answering with citations to the source text.

## Language

**TaxOrdinance**:
The single document this agent answers questions about: the Israeli Income Tax Ordinance (gov.il unofficial consolidated text). Named specifically, not generically — this effort is scoped to this one document; multi-document generalization is explicitly out of scope (see the map) and would be a fresh effort, not a rename of this term.
_Avoid_: LegalDocument, Document, Statute

**Clause**:
One addressable, citable, retrievable unit of text within the TaxOrdinance — what gets embedded, retrieved, and cited. Deliberately granularity-agnostic at the term level even though implementation has since settled on one Clause per סעיף (section) — see the retrieval-pipeline ticket.
_Avoid_: Section (collides with generic prose usage), Provision, Chunk (an implementation/vector-store term, not a domain term)

**Question**:
The payroll accountant's natural-language input to the agent.
_Avoid_: Query (collides with vector-store/DB query terminology used in the retrieval machinery)

**Citation**:
A Clause identifier plus a verbatim quoted excerpt of that Clause's source text, attached to an Answer. Deliberately excludes page number — an artifact of PDF pagination, not something an accountant would use to look up the actual law.
_Avoid_: Reference, Source

**Answer**:
The agent's response to a Question: text grounded in retrieved Clauses, one or more Citations, and a Confidence signal.
_Avoid_: Response, Result

**Confidence**:
A three-state signal on an Answer: **answered** (a retrieved Clause clears the similarity floor and the model is confident it fully addresses the Question), **uncertain** (a Clause clears the floor but the model isn't confident it fully answers the Question — the Answer is still shown, visibly flagged), or **not_found** (no retrieved Clause clears the similarity floor — no synthesized Answer is given, only a plain statement that the TaxOrdinance doesn't appear to address the Question). The similarity floor is a hard guard: it overrides the model's own rating, so the model can't talk itself into "answered" on an irrelevant Clause.
_Avoid_: Score (implies a single numeric mechanism; this is a categorical state derived from two signals, not one number)
