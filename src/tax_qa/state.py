from pydantic import BaseModel


class RetrievedClause(BaseModel):
    content: str
    source: str
    score: float
