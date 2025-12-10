"""Text-to-speech provider interface and implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AudioSegment:
    """Represents a generated audio segment."""

    path: Path
    duration: float  # Duration in seconds
    text: str


class TTSProvider(ABC):
    """Abstract base class for TTS providers."""

    @abstractmethod
    async def synthesize(self, text: str, output_path: Path) -> AudioSegment:
        """Synthesize speech from text.

        Args:
            text: The text to convert to speech.
            output_path: Path to save the audio file.

        Returns:
            AudioSegment with path and duration.
        """
        pass

    @abstractmethod
    async def list_voices(self) -> list[str]:
        """List available voices for this provider.

        Returns:
            List of voice identifiers.
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this TTS provider."""
        pass

    @property
    @abstractmethod
    def current_voice(self) -> str:
        """Return the currently configured voice."""
        pass


class TTSError(Exception):
    """Raised when TTS operations fail."""

    def __init__(self, message: str, provider: str, voice: str | None = None):
        self.provider = provider
        self.voice = voice
        super().__init__(f"[{provider}] {message}")
