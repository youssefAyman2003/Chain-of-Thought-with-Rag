from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to answer using RAG + chain-of-thought")


class AskResponse(BaseModel):
    question: str
    sub_steps: list[str]
    answer: str
    sources: list[str]
