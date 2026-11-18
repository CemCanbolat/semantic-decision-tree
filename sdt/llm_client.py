"""Frontier LLM wrapper, used only for offline induction and escalations.

Phase 1 baselines (single-prompt classifier) also route through here so every
external call is logged in one place. No network in this sandbox: the mock
echoes scripted responses; the HTTP variant fails loudly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from sdt.schemas import SdtError


class FrontierClient(ABC):
    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 1024,
                 temperature: float = 0.7) -> str:
        ...


class MockFrontierClient(FrontierClient):
    """Returns scripted responses from a FIFO list; records every prompt."""

    def __init__(self, scripted: Optional[List[str]] = None) -> None:
        self.scripted = list(scripted or [])
        self.calls: List[Dict[str, Any]] = []

    def complete(self, system: str, user: str, max_tokens: int = 1024,
                 temperature: float = 0.7) -> str:
        self.calls.append({"system": system, "user": user})
        if not self.scripted:
            raise SdtError("MockFrontierClient ran out of scripted responses")
        return self.scripted.pop(0)


class HttpFrontierClient(FrontierClient):
    def __init__(self, model: str = "frontier") -> None:
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        raise SdtError(
            "HttpFrontierClient requires network access; unavailable in this sandbox."
        )
