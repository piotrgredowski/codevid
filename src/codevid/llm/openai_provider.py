"""OpenAI LLM provider."""

import json
import re
from typing import Any

from codevid.llm.base import LLMError, LLMProvider
from codevid.llm.prompts import SCRIPT_GENERATION_PROMPT, STEP_ENHANCEMENT_PROMPT, format_steps_for_prompt
from codevid.models import NarrationSegment, ParsedTest, TestStep, VideoScript


class OpenAIProvider(LLMProvider):
    """LLM provider using OpenAI's API."""

    DEFAULT_MODEL = "gpt-5.1-2025-11-13"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: Model to use. Defaults to gpt-4o.
            base_url: Optional custom base URL for API-compatible services.
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise LLMError(
                "openai package is required. Install with: pip install openai",
                provider="openai",
            )

        self._model = model or self.DEFAULT_MODEL
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    @property
    def provider_name(self) -> str:
        return "openai"

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

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_completion_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            raise LLMError(f"API call failed: {e}", provider="openai", model=self._model)

        response_text = response.choices[0].message.content or ""
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

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            raise LLMError(f"API call failed: {e}", provider="openai", model=self._model)

        return (response.choices[0].message.content or "").strip()

    def _parse_script_response(self, response_text: str, test: ParsedTest) -> VideoScript:
        """Parse the JSON response into a VideoScript."""
        # Extract JSON from response (handle markdown code blocks if present)
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_str = response_text.strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise LLMError(
                f"Failed to parse LLM response as JSON: {e}",
                provider="openai",
                model=self._model,
            )

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
