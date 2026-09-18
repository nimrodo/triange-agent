import functools

import reflex as rx
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from tax_qa.dependencies import Settings, build_llm, build_retriever
from tax_qa.formatting import percent
from tax_qa.graph import build_graph
from tax_qa.state import Answer, Exchange

CONFIDENCE_STYLE = {
    "answered": {"border": "#1e8e3e", "bg": "#e6f4ea", "fg": "#1e8e3e", "icon": "✔"},
    "uncertain": {"border": "#b98900", "bg": "#fff6e0", "fg": "#b98900", "icon": "⚠"},
    "not_found": {"border": "#8a2c2c", "bg": "#f6e9e9", "fg": "#8a2c2c", "icon": "✗"},
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
    percent: str


class MessageView(BaseModel):
    question: str
    confidence: str
    badge: str
    text: str
    not_found_note: str
    sources_label: str
    sources: list[SourceView]


def _exchange_to_view(exchange: Exchange) -> MessageView:
    answer = exchange.answer
    if answer.confidence == "not_found":
        sources = [
            SourceView(source=c.source, excerpt=c.content, percent=percent(c.score))
            for c in answer.considered
        ]
    else:
        sources = [
            SourceView(
                source=c.source,
                excerpt=c.excerpt,
                percent=percent(c.score) if c.score is not None else "",
            )
            for c in answer.citations
        ]
    return MessageView(
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

    @rx.event
    def set_question(self, value: str) -> None:
        self.question = value

    @rx.event(background=True)
    async def ask(self):
        async with self:
            question = self.question.strip()
            if not question or self.is_answering:
                return
            self.question = ""
            self.is_answering = True

        graph = _get_graph()
        config: RunnableConfig = {
            "configurable": {"thread_id": self.router.session.client_token}
        }
        result = await graph.ainvoke({"question": question}, config)

        async with self:
            exchange = Exchange(question=question, answer=result["answer"])
            self.messages.append(_exchange_to_view(exchange))
            self.is_answering = False


def _source_card(source: SourceView) -> rx.Component:
    return rx.el.details(
        rx.el.summary(
            rx.text(source.source, font_weight="600", color="#333"),
            rx.text(f"דמיון: {source.percent}", color="#888", font_size="12px"),
            display="flex",
            justify_content="space-between",
            align_items="center",
            padding="7px 10px",
            cursor="pointer",
        ),
        rx.box(
            source.excerpt,
            padding="8px 10px",
            color="#444",
            line_height="1.5",
            background="#fff",
            border_top="1px solid #ddd6cc",
        ),
        border="1px solid #ddd6cc",
        border_radius="8px",
        background="#fafaf8",
        font_size="13px",
        overflow="hidden",
    )


def _answer_bubble(msg: MessageView) -> rx.Component:
    border_color = rx.match(
        msg.confidence,
        ("answered", CONFIDENCE_STYLE["answered"]["border"]),
        ("uncertain", CONFIDENCE_STYLE["uncertain"]["border"]),
        CONFIDENCE_STYLE["not_found"]["border"],
    )
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
            font_size="11px",
            font_weight="600",
            padding="2px 8px",
            border_radius="999px",
            margin_bottom="8px",
            background=badge_bg,
            color=badge_fg,
        ),
        rx.cond(
            msg.confidence == "not_found",
            rx.text(
                msg.not_found_note,
                font_size="14px",
                color="#555",
                font_style="italic",
            ),
            rx.text(msg.text, font_size="15px", line_height="1.5", color="#222"),
        ),
        rx.vstack(
            rx.text(msg.sources_label, font_size="11px", color="#888"),
            rx.foreach(msg.sources, _source_card),
            align_items="stretch",
            spacing="1",
            margin_top="10px",
        ),
        align_self="flex-start",
        max_width="85%",
        background="#fff",
        border="1px solid #ddd6cc",
        border_left=f"5px solid {border_color}",
        border_radius="14px 14px 14px 4px",
        padding="12px 14px",
    )


def _turn(msg: MessageView) -> rx.Component:
    return rx.vstack(
        rx.box(
            msg.question,
            align_self="flex-end",
            background="#2f6f4f",
            color="#fff",
            padding="10px 14px",
            border_radius="14px 14px 4px 14px",
            max_width="80%",
            font_size="15px",
        ),
        _answer_bubble(msg),
        align_items="stretch",
        spacing="2",
        width="100%",
    )


def _thinking_indicator() -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.spinner(size="1"),
            rx.text("חושב...", font_size="14px", color="#888"),
            spacing="2",
            align_items="center",
        ),
        align_self="flex-start",
        background="#fff",
        border="1px solid #ddd6cc",
        border_radius="14px 14px 14px 4px",
        padding="10px 14px",
    )


def _composer() -> rx.Component:
    return rx.hstack(
        rx.input(
            value=ChatState.question,
            on_change=ChatState.set_question,
            placeholder="הקלד שאלה בעברית...",
            disabled=ChatState.is_answering,
            direction="rtl",
            flex="1",
        ),
        rx.button(
            "שלח",
            on_click=ChatState.ask,
            disabled=ChatState.is_answering,
            background="#2f6f4f",
            color="#fff",
        ),
        border_top="1px solid #ddd6cc",
        background="#fff",
        padding="12px 16px",
        width="100%",
    )


def index() -> rx.Component:
    return rx.box(
        rx.box(
            rx.heading("עוזר שאלות על פקודת מס הכנסה", size="5"),
            rx.text(
                "הדגמה • שאלות בעברית בלבד • התשובות מבוססות על ציטוט מהפקודה בלבד",
                font_size="12px",
                color="#777",
                margin_top="4px",
            ),
            padding="16px 20px",
            border_bottom="1px solid #ddd6cc",
            background="#fff",
        ),
        rx.vstack(
            rx.foreach(ChatState.messages, _turn),
            rx.cond(ChatState.is_answering, _thinking_indicator()),
            flex="1",
            overflow_y="auto",
            padding="20px",
            align_items="stretch",
            spacing="4",
            width="100%",
        ),
        _composer(),
        direction="rtl",
        width="100%",
        max_width="720px",
        height="100vh",
        margin="0 auto",
        display="flex",
        flex_direction="column",
        background="#f4f2ee",
    )


app = rx.App(
    html_lang="he",
    html_custom_attrs={"dir": "rtl"},
    style={"direction": "rtl"},
)
app.add_page(index, title="עוזר שאלות על פקודת מס הכנסה")
