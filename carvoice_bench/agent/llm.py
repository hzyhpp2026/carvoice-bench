"""Small structured-output LLM adapter layer for candidate generation."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMClient(Protocol):
    def complete_json(self, prompt: str) -> Dict[str, Any]:
        """Return a JSON object; callers validate it before it becomes a test asset."""


class OpenAICompatibleClient:
    def __init__(self, endpoint: str, api_key: str, model: str, timeout_seconds: int = 45):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def complete_json(self, prompt: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return only a JSON object that satisfies the requested schema."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
        request = Request(
            self.endpoint + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError("LLM request failed: " + str(exc)) from exc
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("LLM response did not contain chat content") from exc
        return _parse_json_object(content)


class StaticLLMClient:
    """Deterministic client used by tests and offline demonstrations."""

    def __init__(self, response: Dict[str, Any]):
        self.response = response

    def complete_json(self, prompt: str) -> Dict[str, Any]:
        return self.response


def build_llm_client(provider: str, model: str, endpoint: str = "") -> Optional[LLMClient]:
    provider = provider.lower()
    if provider in {"heuristic", "none"}:
        return None
    if provider == "auto":
        provider = "dashscope" if os.getenv("DASHSCOPE_API_KEY") else "heuristic"
        if provider == "heuristic":
            return None
    if provider == "dashscope":
        key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIYUN_API_KEY")
        if not key:
            return None
        return OpenAICompatibleClient(
            endpoint or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            key,
            model or "qwen-plus",
        )
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required for the openai provider")
        return OpenAICompatibleClient(endpoint or "https://api.openai.com/v1", key, model)
    if provider == "compatible":
        key = os.getenv("AGENT_LLM_API_KEY")
        if not key or not endpoint:
            raise RuntimeError("AGENT_LLM_API_KEY and --endpoint are required for a compatible provider")
        return OpenAICompatibleClient(endpoint, key, model)
    raise ValueError("unsupported LLM provider: " + provider)


def _parse_json_object(content: Any) -> Dict[str, Any]:
    if isinstance(content, dict):
        return content
    text = str(content).strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1].strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("LLM did not return valid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError("LLM JSON response must be an object")
    return value
