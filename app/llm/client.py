from typing import Any

import httpx

from app.core.config import settings


class LLMError(RuntimeError):
    """Raised when the configured LLM provider cannot be reached."""


class LLMClient:
    """
    OpenAI-compatible LLM client.

    The application can operate in deterministic demo mode when
    no LLM provider is configured.
    """

    def __init__(self) -> None:
        self.provider = settings.llm_provider
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model
        self.base_url = settings.llm_base_url.rstrip("/")

    @property
    def enabled(self) -> bool:
        return all(
            [
                self.provider == "openai_compatible",
                bool(self.api_key),
                bool(self.model),
                bool(self.base_url),
            ]
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:

        if not self.enabled:
            raise LLMError(
                "LLM is not configured."
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        endpoint = f"{self.base_url}/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                )
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Unable to communicate with LLM provider: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise LLMError(
                f"LLM provider returned HTTP "
                f"{response.status_code}: "
                f"{response.text[:500]}"
            )

        return response.json()