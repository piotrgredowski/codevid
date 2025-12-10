"""Caption and subtitle generation for videos."""

from dataclasses import dataclass
from pathlib import Path

from codevid.models import VideoScript
from codevid.recorder.screen import EventMarker


@dataclass
class Caption:
    """A single caption/subtitle entry."""

    text: str
    start_time: float  # Seconds from video start
    end_time: float  # Seconds from video start

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_srt_entry(self, index: int) -> str:
        """Convert to SRT subtitle format entry."""
        start = self._format_timestamp(self.start_time)
        end = self._format_timestamp(self.end_time)
        return f"{index}\n{start} --> {end}\n{self.text}\n"

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class CaptionGenerator:
    """Generate captions from script and markers."""

    def __init__(
        self,
        max_chars_per_line: int = 42,
        max_lines: int = 2,
    ):
        self.max_chars_per_line = max_chars_per_line
        self.max_lines = max_lines

    def generate_from_script(
        self,
        script: VideoScript,
        markers: list[EventMarker],
        audio_durations: list[float] | None = None,
    ) -> list[Caption]:
        """Generate captions from a video script.

        Args:
            script: The video script with narration.
            markers: Event markers from recording for timing.
            audio_durations: Actual durations of audio segments if available.

        Returns:
            List of Caption objects with timing.
        """
        captions = []
        current_time = 0.0

        # Introduction caption
        if script.introduction:
            intro_duration = audio_durations[0] if audio_durations else 5.0
            intro_captions = self._split_text(script.introduction, current_time, intro_duration)
            captions.extend(intro_captions)
            current_time += intro_duration + 0.5  # Small gap

        # Segment captions
        for i, segment in enumerate(script.segments):
            # Use audio duration if available, otherwise use timing hint
            duration = segment.timing_hint
            if audio_durations and i + 1 < len(audio_durations):
                duration = audio_durations[i + 1]

            segment_captions = self._split_text(segment.text, current_time, duration)
            captions.extend(segment_captions)
            current_time += duration + 0.3

        # Conclusion caption
        if script.conclusion:
            conclusion_duration = audio_durations[-1] if audio_durations else 4.0
            conclusion_captions = self._split_text(script.conclusion, current_time, conclusion_duration)
            captions.extend(conclusion_captions)

        return captions

    def _split_text(self, text: str, start_time: float, total_duration: float) -> list[Caption]:
        """Split long text into multiple captions."""
        words = text.split()
        if not words:
            return []

        # Calculate how many captions we need
        max_chars = self.max_chars_per_line * self.max_lines
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > max_chars and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += word_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Distribute time across chunks
        if not chunks:
            return []

        time_per_chunk = total_duration / len(chunks)
        captions = []

        for i, chunk in enumerate(chunks):
            caption_start = start_time + (i * time_per_chunk)
            caption_end = caption_start + time_per_chunk - 0.1  # Small gap
            captions.append(Caption(
                text=self._wrap_text(chunk),
                start_time=caption_start,
                end_time=caption_end,
            ))

        return captions

    def _wrap_text(self, text: str) -> str:
        """Wrap text to fit within max_chars_per_line."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 > self.max_chars_per_line and current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
                if len(lines) >= self.max_lines:
                    break
            else:
                current_line.append(word)
                current_length += len(word) + 1

        if current_line and len(lines) < self.max_lines:
            lines.append(" ".join(current_line))

        return "\n".join(lines)

    def export_srt(self, captions: list[Caption], output_path: Path) -> Path:
        """Export captions to SRT subtitle file."""
        srt_content = "\n".join(
            caption.to_srt_entry(i + 1)
            for i, caption in enumerate(captions)
        )
        output_path.write_text(srt_content)
        return output_path
