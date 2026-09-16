# Legal Document Q&A Agent

A LangGraph/LangChain agent that lets a payroll accountant ask natural-language questions in Hebrew about the content of a single legal document, answering with citations to the source text.

## Language

**LegalDocument**:
The single document this agent answers questions about (currently the Income Tax Ordinance, gov.il unofficial consolidated text). Kept generic on purpose: a future multi-document generalization is out of scope for now but shouldn't force a rename.
_Avoid_: Ordinance, Statute, Document

**Clause**:
One addressable, citable, retrievable unit of text within a LegalDocument — what gets embedded, retrieved, and cited. Deliberately granularity-agnostic: the document's actual chapter/section numbering structure is still under investigation, so a Clause may end up mapping 1:1 to a legal section, or to some other extraction-determined unit.
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
The signal on an Answer indicating whether it's well-grounded in retrieved Clauses. Its mechanics (levels, thresholds, how it's computed) are not yet decided — see the open decision on designing the confidence signal.
_Avoid_: Score (implies a specific numeric mechanism not yet decided)
