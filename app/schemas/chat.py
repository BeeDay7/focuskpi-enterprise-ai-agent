from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):

    message: str = Field(
        min_length=2,
        max_length=2000,
    )


class ChatResponse(BaseModel):

    answer: str

    intent: str

    tool_calls: list[str]

    data: dict[str, Any] = {}

    mode: str = "demo"