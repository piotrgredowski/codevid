"""Base LLM provider interface."""

from abc import ABC, abstractmethod
from typing import Any

from codevid.models import ParsedTest, TestStep, VideoScript


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate_script(
        self,
        test: ParsedTest,
        context: dict[str, Any] | None = None,
    ) -> VideoScript:
        """Generate a narration script from a parsed test.

        Args:
            test: The parsed test to generate a script for.
            context: Additional context like app name, purpose, etc.

        Returns:
            VideoScript with narration segments.
        """
        pass

    @abstractmethod
    async def enhance_description(
        self,
        step: TestStep,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Generate a human-friendly description of a test step.

        Args:
            step: The test step to describe.
            context: Additional context for better descriptions.

        Returns:
            Natural language description of the step.
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this LLM provider."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model being used."""
        pass


class LLMError(Exception):
    """Raised when LLM operations fail."""

    def __init__(self, message: str, provider: str, model: str | None = None):
        self.provider = provider
        self.model = model
        super().__init__(f"[{provider}] {message}")
