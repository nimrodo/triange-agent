import functools
import uuid

import reflex as rx
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from tax_qa.dependencies import Settings, build_llm, build_retriever
from tax_qa.formatting import percent
from tax_qa.graph import build_graph
from tax_qa.state import Answer, Exchange

INK = "#1c2b33"
INK_MUTED = "#5b6b73"
PAPER = "#efeade"
PAPER_RAISED = "#ffffff"
RULE = "#c9c0aa"
GOLD = "#96721f"
GOLD_BG = "#f4ead2"
AMBER = "#a6720a"
AMBER_BG = "#f6ecd8"
RUST = "#8c3b2e"
RUST_BG = "#f2e2de"

SERIF_FONT = "'Frank Ruhl Libre', 'Times New Roman', serif"
SANS_FONT = "'Heebo', 'Segoe UI', sans-serif"

CONFIDENCE_STYLE = {
    "answered": {"border": GOLD, "bg": GOLD_BG, "fg": GOLD, "icon": "✔"},
    "uncertain": {"border": AMBER, "bg": AMBER_BG, "fg": AMBER, "icon": "⚠"},
    "not_found": {"border": RUST, "bg": RUST_BG, "fg": RUST, "icon": "✗"},
}

NOT_FOUND_NOTE = (
    "הפקודה, כפי שנסרקה, אינה מתייחסת במפורש לנושא זה. "
    "להלן הסעיפים הקרובים ביותר שנמצאו (ציון דמיון נמוך):"
)


@functools.lru_cache(maxsize=1)
def _get_graph():
    settings = Settings()
    return build_graph(
        build_llm(settings),
        build_retriever(settings),
        InMemorySaver(),
        search_k=settings.retrieval_k,
    )


def _badge_text(answer: Answer) -> str:
    icon = CONFIDENCE_STYLE[answer.confidence]["icon"]
    if answer.confidence == "answered":
        return f"{icon} נענתה"
    if answer.confidence == "not_found":
        return f"{icon} לא נמצא"
    scores = [c.score for c in answer.citations if c.score is not None]
    return f"{icon} ודאות נמוכה — דמיון: {percent(max(scores, default=0.0))}"


def _sources_label(answer: Answer) -> str:
    if answer.confidence == "not_found":
        return f"סעיפים קרובים ({len(answer.considered)})"
    return f"מקורות ({len(answer.citations)})"


class SourceView(BaseModel):
    source: str
    excerpt: str
    meta_label: str
    demoted: bool = False


class MessageView(BaseModel):
    number: int
    question: str
    confidence: str
    badge: str
    text: str
    not_found_note: str
    sources_label: str
    sources: list[SourceView]


def _exchange_to_view(exchange: Exchange, number: int) -> MessageView:
    answer = exchange.answer
    if answer.confidence == "not_found":
        sources = [
            SourceView(
                source=c.source,
                excerpt=c.content,
                meta_label=f"דמיון: {percent(c.score)}",
                demoted=True,
            )
            for c in answer.considered
        ]
    else:
        sources = [
            SourceView(
                source=c.source,
                excerpt=c.excerpt,
                meta_label=percent(c.score) if c.score is not None else "",
            )
            for c in answer.citations
        ]
    return MessageView(
        number=number,
        question=exchange.question,
        confidence=answer.confidence,
        badge=_badge_text(answer),
        text=answer.text,
        not_found_note=NOT_FOUND_NOTE if answer.confidence == "not_found" else "",
        sources_label=_sources_label(answer),
        sources=sources,
    )


class ChatState(rx.State):
    question: str = ""
    # Reflex's State metaclass gives each instance its own list; this isn't
    # a shared mutable default the way it would be on a plain class/dataclass.
    messages: list[MessageView] = []  # noqa: RUF012
    is_answering: bool = False
    thread_id: str = ""

    @rx.event
    def set_question(self, value: str) -> None:
        self.question = value

    @rx.event
    def new_question(self) -> None:
        if self.is_answering:
            return
        self.messages = []
        self.question = ""
        self.thread_id = uuid.uuid4().hex

    @rx.event(background=True)
    async def ask(self):
        async with self:
            question = self.question.strip()
            if not question or self.is_answering:
                return
            self.question = ""
            self.is_answering = True
            if not self.thread_id:
                self.thread_id = uuid.uuid4().hex
            thread_id = self.thread_id

        graph = _get_graph()
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        result = await graph.ainvoke({"question": question}, config)

        async with self:
            exchange = Exchange(question=question, answer=result["answer"])
            view = _exchange_to_view(exchange, number=len(self.messages) + 1)
            self.messages.append(view)
            self.is_answering = False


def _source_row(source: SourceView) -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.text(
                source.source,
                font_family=SANS_FONT,
                font_weight="600",
                font_size="13px",
                color=INK,
            ),
            rx.text(
                source.meta_label,
                font_family=SANS_FONT,
                font_size="12px",
                color=INK_MUTED,
            ),
            justify_content="space-between",
            width="100%",
        ),
        rx.text(
            source.excerpt,
            font_family=SERIF_FONT,
            font_size="14px",
            line_height="1.6",
            color=INK,
            margin_top="4px",
        ),
        opacity=rx.cond(source.demoted, "0.6", "1"),
        padding="10px 0",
        border_bottom=f"1px solid {RULE}",
    )


def _answer_entry(msg: MessageView) -> rx.Component:
    badge_bg = rx.match(
        msg.confidence,
        ("answered", CONFIDENCE_STYLE["answered"]["bg"]),
        ("uncertain", CONFIDENCE_STYLE["uncertain"]["bg"]),
        CONFIDENCE_STYLE["not_found"]["bg"],
    )
    badge_fg = rx.match(
        msg.confidence,
        ("answered", CONFIDENCE_STYLE["answered"]["fg"]),
        ("uncertain", CONFIDENCE_STYLE["uncertain"]["fg"]),
        CONFIDENCE_STYLE["not_found"]["fg"],
    )
    return rx.box(
        rx.box(
            msg.badge,
            display="inline-flex",
            font_family=SANS_FONT,
            font_size="12px",
            font_weight="600",
            padding="2px 10px",
            border_radius="3px",
            margin_bottom="10px",
            background=badge_bg,
            color=badge_fg,
        ),
        rx.cond(
            msg.confidence == "not_found",
            rx.text(
                msg.not_found_note,
                font_family=SERIF_FONT,
                font_size="15px",
                line_height="1.6",
                color=INK_MUTED,
            ),
            rx.markdown(
                msg.text,
                font_family=SERIF_FONT,
                font_size="16px",
                line_height="1.7",
                color=INK,
            ),
        ),
        rx.vstack(
            rx.text(
                msg.sources_label,
                font_family=SANS_FONT,
                font_size="12px",
                color=INK_MUTED,
                margin_bottom="2px",
            ),
            rx.foreach(msg.sources, _source_row),
            align_items="stretch",
            spacing="0",
            margin_top="14px",
        ),
        align_self="stretch",
        padding_top="10px",
    )


def _turn(msg: MessageView) -> rx.Component:
    return rx.vstack(
        rx.text(
            f"שאלה {msg.number}",
            font_family=SANS_FONT,
            font_size="12px",
            color=INK_MUTED,
        ),
        rx.text(
            msg.question,
            font_family=SANS_FONT,
            font_size="17px",
            font_weight="500",
            color=INK,
        ),
        rx.box(height="1px", background=RULE, margin="12px 0"),
        _answer_entry(msg),
        align_items="stretch",
        spacing="1",
        width="100%",
        padding="0 0 24px",
        border_bottom=f"1px solid {RULE}",
    )


def _thinking_indicator() -> rx.Component:
    return rx.hstack(
        rx.spinner(size="1"),
        rx.text("חושב...", font_family=SANS_FONT, font_size="14px", color=INK_MUTED),
        spacing="2",
        align_items="center",
        padding_top="4px",
    )


def _composer() -> rx.Component:
    return rx.form(
        rx.hstack(
            rx.text_area(
                value=ChatState.question,
                on_change=ChatState.set_question,
                placeholder="הקלד שאלה בעברית...",
                disabled=ChatState.is_answering,
                direction="rtl",
                enter_key_submit=True,
                auto_height=True,
                rows="1",
                flex="1",
                font_family=SANS_FONT,
                background=PAPER_RAISED,
                border=f"1px solid {RULE}",
                color=INK,
            ),
            rx.button(
                "שלח",
                type="button",
                on_click=ChatState.ask,
                disabled=ChatState.is_answering,
                font_family=SANS_FONT,
                background=INK,
                color=PAPER_RAISED,
                border="none",
            ),
            rx.button(
                "שאלה חדשה",
                type="button",
                on_click=ChatState.new_question,
                disabled=ChatState.is_answering,
                font_family=SANS_FONT,
                background="transparent",
                color=INK,
                border=f"1px solid {RULE}",
            ),
            align_items="flex-end",
        ),
        on_submit=ChatState.ask,
        border_top=f"1px solid {RULE}",
        background=PAPER_RAISED,
        padding="12px 16px",
        width="100%",
    )


def index() -> rx.Component:
    return rx.box(
        rx.box(
            rx.heading(
                "עוזר שאלות על פקודת מס הכנסה",
                size="5",
                font_family=SERIF_FONT,
                color=INK,
            ),
            rx.text(
                "שאלות בעברית בלבד • התשובות מבוססות על ציטוט מהפקודה בלבד",
                font_family=SANS_FONT,
                font_size="12px",
                color=INK_MUTED,
                margin_top="4px",
            ),
            padding="16px 20px",
            border_bottom=f"1px solid {RULE}",
            background=PAPER_RAISED,
        ),
        rx.vstack(
            rx.foreach(ChatState.messages, _turn),
            rx.cond(ChatState.is_answering, _thinking_indicator()),
            flex="1",
            overflow_y="auto",
            padding="20px",
            align_items="stretch",
            spacing="6",
            width="100%",
        ),
        _composer(),
        direction="rtl",
        width="100%",
        max_width="760px",
        height="100vh",
        margin="0 auto",
        display="flex",
        flex_direction="column",
        background=PAPER,
    )


app = rx.App(
    html_lang="he",
    html_custom_attrs={"dir": "rtl"},
    style={"direction": "rtl", "font_family": SANS_FONT},
    stylesheets=[
        (
            "https://fonts.googleapis.com/css2"
            "?family=Frank+Ruhl+Libre:wght@400;500;700&family=Heebo:wght@400;500;700"
            "&display=swap"
        )
    ],
)
app.add_page(index, title="עוזר שאלות על פקודת מס הכנסה")
