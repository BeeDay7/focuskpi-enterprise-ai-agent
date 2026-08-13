from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):

    message: str = Field(
        min_length=2,
        max_length=2000,
    )

    user_id: str = Field(
        default="anonymous",
        min_length=1,
        max_length=100,
    )


class ChatResponse(BaseModel):

    answer: str

    intent: str

    tool_calls: list[str]

    data: dict[str, Any] = Field(
        default_factory=dict
    )

    mode: str = "demo"