# Context Map

This repo hosts two independent bounded contexts, each with its own glossary.

- **Customer Support Triage Agent** — classifies and routes incoming support tickets. Glossary: [CONTEXT.md](CONTEXT.md). Code: `src/triange_agent/`.
- **Israeli Income Tax Q&A Agent** — answers questions about the content of the Israeli Income Tax Ordinance, with citations. Glossary: [src/tax_qa/CONTEXT.md](src/tax_qa/CONTEXT.md). Code: `src/tax_qa/`.

No vocabulary is shared between the two contexts.
