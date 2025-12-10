"""LLM integration for script generation."""

from codevid.llm.base import LLMError, LLMProvider
from codevid.llm.factory import create_llm_provider, get_provider_for_name

__all__ = [
    "LLMProvider",
    "LLMError",
    "create_llm_provider",
    "get_provider_for_name",
]
