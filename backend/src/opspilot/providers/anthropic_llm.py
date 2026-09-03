"""Anthropic (Claude) LLM provider.

Thin wrapper over the official SDK. Structured-output orchestration is built on
top of this in the ``generation`` package (Phase 3).
"""

from __future__ import annotations

from typing import Any

from opspilot.providers.base import LLMResult


class AnthropicLLMProvider:
    name = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-sonnet-5") -> None:
        # Imported here so the dependency is only required when this provider
        # is actually selected.
        from anthropic import Anthropic

        self.model = model
        self._client = Anthropic(api_key=api_key)

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> LLMResult:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": user}],
        }
        message = self._client.messages.create(**kwargs)
        text = "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        )
        return LLMResult(
            text=text,
            model=self.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            stop_reason=message.stop_reason,
        )
