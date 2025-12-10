"""Screen recording functionality."""

from dataclasses import dataclass, field
from pathlib import Path
import platform
import subprocess
import time
from typing import Any


@dataclass
class RecordingConfig:
    """Configuration for screen recording."""

    output_path: Path
    fps: int = 30
    resolution: tuple[int, int] | None = None  # None = auto-detect
    capture_mouse: bool = True
    highlight_clicks: bool = True


@dataclass
class EventMarker:
    """A timestamped event during recording."""

    timestamp: float  # Seconds from start
    event_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ScreenRecorder:
    """FFmpeg-based screen recorder with event markers."""

    def __init__(self, config: RecordingConfig):
        self.config = config
        self._process: subprocess.Popen | None = None
        self._markers: list[EventMarker] = []
        self._start_time: float | None = None
        self._is_recording = False

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def markers(self) -> list[EventMarker]:
        return self._markers.copy()

    @property
    def elapsed_time(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def start(self) -> None:
        """Start screen recording."""
        if self._is_recording:
            raise RecordingError("Recording already in progress")

        cmd = self._build_ffmpeg_command()
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._start_time = time.time()
        self._is_recording = True
        self._markers = []

    def mark_event(self, event_type: str, metadata: dict[str, Any] | None = None) -> EventMarker:
        """Mark a timestamped event during recording.

        Args:
            event_type: Type of event (e.g., "click", "navigate", "step_start").
            metadata: Additional data about the event.

        Returns:
            The created EventMarker.
        """
        if not self._is_recording:
            raise RecordingError("Not currently recording")

        marker = EventMarker(
            timestamp=self.elapsed_time,
            event_type=event_type,
            metadata=metadata or {},
        )
        self._markers.append(marker)
        return marker

    def stop(self) -> Path:
        """Stop recording and return output path.

        Returns:
            Path to the recorded video file.
        """
        if not self._is_recording or self._process is None:
            raise RecordingError("Not currently recording")

        # Send 'q' to gracefully stop FFmpeg
        try:
            self._process.stdin.write(b"q")
            self._process.stdin.flush()
            self._process.wait(timeout=10)
        except Exception:
            self._process.terminate()
            self._process.wait(timeout=5)

        self._is_recording = False
        return self.config.output_path

    def _build_ffmpeg_command(self) -> list[str]:
        """Build FFmpeg command for current platform."""
        system = platform.system()

        if system == "Darwin":
            # macOS: Use AVFoundation
            input_args = ["-f", "avfoundation", "-i", "1:none"]
        elif system == "Linux":
            # Linux: Use X11grab
            display = ":0.0"
            input_args = ["-f", "x11grab", "-i", display]
        elif system == "Windows":
            # Windows: Use GDI grab
            input_args = ["-f", "gdigrab", "-i", "desktop"]
        else:
            raise RecordingError(f"Unsupported platform: {system}")

        # Build resolution args if specified
        resolution_args = []
        if self.config.resolution:
            w, h = self.config.resolution
            resolution_args = ["-video_size", f"{w}x{h}"]

        # Mouse capture (platform-specific)
        mouse_args = []
        if self.config.capture_mouse and system == "Darwin":
            mouse_args = ["-capture_cursor", "1"]

        return [
            "Ye",
            "-y",  # Overwrite output
            *resolution_args,
            "-framerate",
            str(self.config.fps),
            *mouse_args,
            *input_args,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            str(self.config.output_path),
        ]


class RecordingError(Exception):
    """Raised when recording operations fail."""

    pass
