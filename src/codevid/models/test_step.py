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
    metadata: dict = field(default_factory=dict)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def get_actions_summary(self) -> dict[ActionType, int]:
        """Count occurrences of each action type."""
        summary: dict[ActionType, int] = {}
        for step in self.steps:
            summary[step.action] = summary.get(step.action, 0) + 1
        return summary
