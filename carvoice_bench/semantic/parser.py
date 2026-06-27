"""Semantic parser interface."""

from __future__ import annotations

from typing import Protocol


class SemanticParser(Protocol):
    """Convert recognized text into intent/slots."""

    def parse(self, text: str, case: dict | None = None) -> dict:
        """Return ``{"intent": str, "slots": dict, ...}``."""
        ...
