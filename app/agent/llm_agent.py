import json

from sqlalchemy.orm import Session

from app.agent.tools import TOOLS, execute_tool
from app.llm.client import LLMClient, LLMError


SYSTEM_PROMPT = """
You are an enterprise data intelligence assistant.

Rules:

1. Use approved tools whenever enterprise data is required.
2. Never invent database values.
3. Never invent ML predictions.
4. Never generate or execute arbitrary SQL.
5. Clearly distinguish factual database information from predictions.
6. Keep answers concise and business-oriented.
7. If available tools cannot answer a request, explain the limitation.

Available capabilities:
- sales analytics
- customer churn prediction
- customer risk ranking
"""


class LLMToolAgent:

    def __init__(self) -> None:
        self.client = LLMClient()

    async def run(
        self,
        db: Session,
        user_message: str,
    ) -> dict:

        if not self.client.enabled:
            raise LLMError(
                "LLM provider is not configured."
            )

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ]

        first_response = await self.client.chat(
            messages=messages,
            tools=TOOLS,
        )

        choice = (
            first_response
            .get("choices", [{}])[0]
            .get("message", {})
        )

        tool_calls = choice.get("tool_calls") or []

        if not tool_calls:
            return {
                "answer": (
                    choice.get("content")
                    or "I could not produce an answer."
                ),
                "intent": "llm_response",
                "tool_calls": [],
                "data": {},
                "mode": "llm",
            }

        messages.append(choice)

        tool_results = []
        tool_names = []

        for tool_call in tool_calls:

            function = tool_call.get(
                "function",
                {},
            )

            tool_name = function.get("name")

            arguments = json.loads(
                function.get("arguments") or "{}"
            )

            result = execute_tool(
                db=db,
                tool_name=tool_name,
                arguments=arguments,
            )

            tool_names.append(tool_name)
            tool_results.append(result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get(
                        "id",
                        "",
                    ),
                    "name": tool_name,
                    "content": json.dumps(result),
                }
            )

        final_response = await self.client.chat(
            messages=messages
        )

        final_message = (
            final_response
            .get("choices", [{}])[0]
            .get("message", {})
        )

        return {
            "answer": (
                final_message.get("content")
                or "The requested tools completed successfully."
            ),
            "intent": "tool_augmented_llm",
            "tool_calls": tool_names,
            "data": {
                "tool_results": tool_results,
            },
            "mode": "llm",
        }