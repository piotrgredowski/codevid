"""Data models for test steps and parsed tests."""

from dataclasses import dataclass, field
from enum import Enum


class ActionType(Enum):
    """Types of actions that can be performed in a test."""

    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    WAIT = "wait"
    ASSERT = "assert"
    HOVER = "hover"
    SELECT = "select"
    PRESS = "press"
    SCREENSHOT = "screenshot"
    CUSTOM = "custom"


@dataclass
class TestStep:
    """Represents a single step in a test."""

    action: ActionType
    target: str  # Selector, URL, or key
    value: str | None = None  # Input value for type actions
    description: str = ""  # Human-readable description
    line_number: int = 0
    source_code: str = ""
    skip_recording: bool = False  # If True, execute but don't record in video

    def __str__(self) -> str:
        if self.value:
            return f"{self.action.value}({self.target!r}, {self.value!r})"
        return f"{self.action.value}({self.target!r})"


@dataclass
class ParsedTest:
    """Represents a fully parsed test file."""

    name: str
    file_path: str
    steps: list[TestStep]
    setup_code: str = ""
    teardown_code: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def get_actions_summary(self) -> dict[ActionType, int]:
        """Count occurrences of each action type."""
        summary: dict[ActionType, int] = {}
        for step in self.steps:
            summary[step.action] = summary.get(step.action, 0) + 1
        return summary

    def has_skip_markers(self) -> bool:
        """Check if any steps are marked to skip recording."""
        return any(step.skip_recording for step in self.steps)

    def get_setup_steps(self) -> list[TestStep]:
        """Get contiguous skipped steps at the beginning (pre-recording setup).

        Returns steps that are marked skip_recording=True from the start,
        stopping at the first non-skipped step.
        """
        setup: list[TestStep] = []
        for step in self.steps:
            if step.skip_recording:
                setup.append(step)
            else:
                break
        return setup

    def get_recorded_steps(self) -> list[TestStep]:
        """Get steps that should be recorded (between first and last non-skipped).

        Returns the middle section of steps, from the first non-skipped step
        to the last non-skipped step (inclusive). May include skipped steps
        in the middle that will execute with minimal delay but no narration.
        """
        if not self.steps:
            return []

        # Find first non-skipped step
        first_recorded = 0
        for i, step in enumerate(self.steps):
            if not step.skip_recording:
                first_recorded = i
                break
        else:
            # All steps are skipped
            return []

        # Find last non-skipped step
        last_recorded = len(self.steps) - 1
        for i in range(len(self.steps) - 1, -1, -1):
            if not self.steps[i].skip_recording:
                last_recorded = i
                break

        return self.steps[first_recorded : last_recorded + 1]

    def get_teardown_steps(self) -> list[TestStep]:
        """Get contiguous skipped steps at the end (post-recording teardown).

        Returns steps that are marked skip_recording=True from the end,
        stopping at the last non-skipped step.
        """
        teardown: list[TestStep] = []
        for step in reversed(self.steps):
            if step.skip_recording:
                teardown.insert(0, step)
            else:
                break
        return teardown

    def get_step_original_index(self, step: TestStep) -> int:
        """Get the original index of a step in the full steps list.

        Useful for mapping recorded steps back to their original indices
        for EventMarker synchronization.
        """
        return self.steps.index(step)
