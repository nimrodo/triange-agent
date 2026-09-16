# Customer Support Triage Agent

A learning-focused LangGraph/LangChain agent that classifies and routes incoming support tickets, built to exercise thread-scoped state, cyclic routing, human-in-the-loop pause/resume, and RAG tool calling.

## Language

**Ticket**:
An incoming customer support request (subject + body text) that enters the graph.
_Avoid_: Request, message

**Category**:
One of a fixed small set of triage classifications a Ticket is routed to (e.g. billing, technical, account, general). Assigned by a classification node.
_Avoid_: Label, class, type

**TriageState**:
The shared Pydantic model graph state — carries the Ticket, its assigned Category, conversation history, retrieved KnowledgeBase snippets, and the escalation flag/human decision.
_Avoid_: Context, session

**KnowledgeBase**:
The toy corpus of FAQ/markdown docs embedded into the vector store, queried by the Answer tool.
_Avoid_: Docs, corpus

**Answer tool**:
The RAG tool call a node makes against the vector store to retrieve KnowledgeBase snippets and draft a response.
_Avoid_: RAG lookup, retriever

**Escalation**:
The routing decision that a Ticket can't be resolved by the Answer tool and needs a human — triggers the HITL pause.
_Avoid_: Handoff, fallback

**Review**:
The HITL step: the graph interrupts, a human inspects the proposed Category/answer, and either approves or redirects it, then the graph resumes.
_Avoid_: Approval, checkpoint (checkpoint is reserved for LangGraph's own checkpointer concept)

**Thread**:
A single Ticket's run through the graph, identified by a thread_id, scoping the checkpointed memory.
_Avoid_: Session, run

**Attempt**:
One classify/answer/Review cycle for a Ticket: a drafted answer plus the Review feedback that followed it, if any. A Ticket's history is a sequence of Attempts, accumulated each time the Review loop cycles back.
_Avoid_: Turn, iteration, message
