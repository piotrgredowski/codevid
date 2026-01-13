"""Ollama local LLM provider."""

import json
import re
from typing import Any

import httpx

from codevid.llm.base import LLMError, LLMProvider
from codevid.llm.prompts import SCRIPT_GENERATION_PROMPT, STEP_ENHANCEMENT_PROMPT, format_steps_for_prompt
from codevid.models import NarrationSegment, ParsedTest, TestStep, VideoScript


class OllamaProvider(LLMProvider):
    """LLM provider using local Ollama server."""

    DEFAULT_MODEL = "llama3.2"
    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ):
        """Initialize Ollama provider.

        Args:
            model: Model to use. Defaults to llama3.2.
            base_url: Ollama server URL. Defaults to http://localhost:11434.
            timeout: Request timeout in seconds.
        """
        self._model = model or self.DEFAULT_MODEL
        self._base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self._timeout = timeout

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate_script(
        self,
        test: ParsedTest,
        context: dict[str, Any] | None = None,
    ) -> VideoScript:
        """Generate a narration script from a parsed test."""
        context = context or {}

        prompt = SCRIPT_GENERATION_PROMPT.format(
            test_name=test.name,
            app_name=context.get("app_name", "the application"),
            test_purpose=context.get("purpose", test.metadata.get("docstring", "demonstrate functionality")),
            formatted_steps=format_steps_for_prompt(test.steps),
        )

        response_text = await self._generate(prompt)
        return self._parse_script_response(response_text, test)

    async def enhance_description(
        self,
        step: TestStep,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Generate a human-friendly description of a test step."""
        context = context or {}

        prompt = STEP_ENHANCEMENT_PROMPT.format(
            action=step.action.value,
            target=step.target,
            value=step.value or "N/A",
            context=context.get("previous_steps", "This is the first step"),
        )

        response_text = await self._generate(prompt)
        return response_text.strip()

    async def _generate(self, prompt: str) -> str:
        """Make a generate request to Ollama."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(
                    f"{self._base_url}/api/generate",
                    json={
                        "model": self._model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                        },
                    },
                )
                response.raise_for_status()
            except httpx.ConnectError:
                raise LLMError(
                    f"Could not connect to Ollama at {self._base_url}. "
                    "Make sure Ollama is running.",
                    provider="ollama",
                    model=self._model,
                )
            except httpx.HTTPStatusError as e:
                raise LLMError(
                    f"Ollama request failed: {e.response.status_code}",
                    provider="ollama",
                    model=self._model,
                )
            except Exception as e:
                raise LLMError(f"Ollama request failed: {e}", provider="ollama", model=self._model)

        data = response.json()
        return data.get("response", "")

    async def list_models(self) -> list[str]:
        """List available models on the Ollama server."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(f"{self._base_url}/api/tags")
                response.raise_for_status()
            except Exception as e:
                raise LLMError(f"Failed to list models: {e}", provider="ollama")

        data = response.json()
        return [model["name"] for model in data.get("models", [])]

    def _parse_script_response(self, response_text: str, test: ParsedTest) -> VideoScript:
        """Parse the JSON response into a VideoScript."""
        # Extract JSON from response
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try to find JSON object in text
            json_match = re.search(r"\{[\s\S]*\}", response_text)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Fall back to defaults if no JSON found
                return self._create_default_script(test, response_text)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # If JSON parsing fails, create default script
            return self._create_default_script(test, response_text)

        segments = []
        for seg_data in data.get("segments", []):
            segments.append(
                NarrationSegment(
                    text=seg_data.get("text", ""),
                    step_index=seg_data.get("step_index", 0),
                    timing_hint=seg_data.get("timing_hint", 3.0),
                    emphasis_words=seg_data.get("emphasis_words", []),
                )
            )

        if not segments:
            segments = self._create_default_segments(test)

        return VideoScript(
            title=data.get("title", test.name),
            introduction=data.get("introduction", f"In this tutorial, we'll walk through {test.name}."),
            segments=segments,
            conclusion=data.get("conclusion", "That's all for this tutorial!"),
        )

    def _create_default_script(self, test: ParsedTest, raw_response: str) -> VideoScript:
        """Create a default script when JSON parsing fails."""
        return VideoScript(
            title=test.name,
            introduction=f"In this tutorial, we'll walk through {test.name}.",
            segments=self._create_default_segments(test),
            conclusion="That's all for this tutorial!",
        )

    def _create_default_segments(self, test: ParsedTest) -> list[NarrationSegment]:
        """Create default narration segments from test steps."""
        segments = []
        for i, step in enumerate(test.steps):
            segments.append(
                NarrationSegment(
                    text=step.description or f"Perform {step.action.value} action",
                    step_index=i,
                    timing_hint=3.0,
                )
            )
        return segments
