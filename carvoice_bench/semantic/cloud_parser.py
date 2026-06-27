"""Placeholder for future cloud LLM semantic parsing."""

from __future__ import annotations

from carvoice_bench.config import Config


class CloudSemanticParser:
    """Future adapter for Alibaba Cloud LLM intent/slot extraction."""

    def __init__(self, config: Config):
        self.config = config

    def parse(self, text: str, case: dict | None = None) -> dict:
        raise NotImplementedError(
            "Cloud semantic parsing is reserved for the next adapter. "
            "Use semantic_parser='rule' for offline rule parsing."
        )
