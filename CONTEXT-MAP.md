# Context Map

This repo hosts two independent bounded contexts, each with its own glossary.

- **Customer Support Triage Agent** — classifies and routes incoming support tickets. Glossary: [CONTEXT.md](CONTEXT.md). Code: `src/triange_agent/`.
- **Legal Document Q&A Agent** — answers questions about the content of a legal document, with citations. Glossary: [src/legal_qa/CONTEXT.md](src/legal_qa/CONTEXT.md). Code: `src/legal_qa/`.

No vocabulary is shared between the two contexts.
