"""Lightweight, local-only LLM provider for demos/tests."""

from codevid.llm.base import LLMProvider
from codevid.models import NarrationSegment, ParsedTest, TestStep, VideoScript


class SimpleLLM(LLMProvider):
    """Generate a deterministic script from parsed steps without external APIs."""

    @property
    def provider_name(self) -> str:
        return "simple"

    @property
    def model_name(self) -> str:
        return "rule-based"

    async def generate_script(
        self,
        test: ParsedTest,
        context: dict[str, str] | None = None,
    ) -> VideoScript:
        app_name = (context or {}).get("app_name") or "the application"

        human_name = test.name.replace("_", " ")
        purpose = f"This tutorial shows the purpose behind the {human_name} flow in {app_name} and what a successful run should look like."
        intro = (
            f"Welcome to the {human_name} tutorial. {purpose} "
            "We'll start with the goal, then walk through each action and the expected result."
        )

        # Build a quick roadmap of steps for the overview segment
        step_summaries = [self._step_to_text(step) for step in test.steps]
        roadmap = " ".join(f"Step {i + 1}: {text}" for i, text in enumerate(step_summaries))

        segments = []
        segments.append(
            NarrationSegment(
                step_index=-1,
                text=f"How to do it: we'll follow this plan. {roadmap}",
                timing_hint=4.0,
            )
        )
        for idx, step in enumerate(test.steps):
            text = f"Step {idx + 1}: {self._step_to_text(step)}"
            segments.append(
                NarrationSegment(
                    step_index=idx,
                    text=text,
                    timing_hint=2.5,
                )
            )

        outcome = (
            f"Outcome: you should now have a completed {human_name} run in {app_name}, "
            "with the final screen reflecting the expected result of the test."
        )
        conclusion = (
            f"{outcome} If you saw different results, rerun the test and adjust inputs or timings as needed. "
            "Otherwise, you can reuse this flow as a template for related tutorials."
        )

        return VideoScript(
            title=f"{app_name} tutorial: {test.name.replace('_', ' ')}",
            introduction=intro,
            segments=segments,
            conclusion=conclusion,
        )

    async def enhance_description(
        self,
        step: TestStep,
        context: dict[str, str] | None = None,
    ) -> str:
        return self._step_to_text(step)

    def _step_to_text(self, step: TestStep) -> str:
        """Convert a step into a short narration line."""
        base = step.description or f"{step.action.value.title()} {step.target}"
        if step.value:
            return f"{base} using value '{step.value}'."
        return f"{base}."
