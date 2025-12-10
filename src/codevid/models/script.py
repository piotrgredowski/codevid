"""Data models for video scripts and narration."""

from dataclasses import dataclass, field


@dataclass
class NarrationSegment:
    """A segment of narration corresponding to a test step."""

    text: str
    step_index: int
    timing_hint: float  # Suggested duration in seconds
    emphasis_words: list[str] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def estimated_duration(self, words_per_minute: int = 150) -> float:
        """Estimate speaking duration based on word count."""
        return (self.word_count / words_per_minute) * 60


@dataclass
class VideoScript:
    """Complete script for a video tutorial."""

    title: str
    introduction: str
    segments: list[NarrationSegment]
    conclusion: str
    total_estimated_duration: float = 0.0

    def __post_init__(self) -> None:
        if self.total_estimated_duration == 0.0:
            self.total_estimated_duration = self.calculate_duration()

    def calculate_duration(self, words_per_minute: int = 150) -> float:
        """Calculate total estimated duration in seconds."""
        intro_words = len(self.introduction.split())
        conclusion_words = len(self.conclusion.split())
        segment_duration = sum(seg.timing_hint for seg in self.segments)

        spoken_duration = ((intro_words + conclusion_words) / words_per_minute) * 60
        return spoken_duration + segment_duration

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    def get_full_text(self) -> str:
        """Get the complete narration text."""
        parts = [self.introduction]
        parts.extend(seg.text for seg in self.segments)
        parts.append(self.conclusion)
        return "\n\n".join(parts)
