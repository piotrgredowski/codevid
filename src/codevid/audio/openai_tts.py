"""OpenAI TTS provider - high-quality text-to-speech."""

from pathlib import Path

from codevid.audio.tts import AudioSegment, TTSError, TTSProvider


class OpenAITTSProvider(TTSProvider):
    """TTS provider using OpenAI's text-to-speech API."""

    DEFAULT_VOICE = "alloy"
    DEFAULT_MODEL = "tts-1"

    # Available voices
    VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    def __init__(
        self,
        api_key: str | None = None,
        voice: str | None = None,
        model: str | None = None,
        speed: float = 1.0,
    ):
        """Initialize OpenAI TTS provider.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            voice: Voice to use. Options: alloy, echo, fable, onyx, nova, shimmer.
            model: Model to use. Options: tts-1 (faster), tts-1-hd (higher quality).
            speed: Speech speed multiplier (0.25 to 4.0).
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise TTSError(
                "openai package is required. Install with: pip install openai",
                provider="openai",
            )

        self._client = OpenAI(api_key=api_key)
        self._voice = voice or self.DEFAULT_VOICE
        self._model = model or self.DEFAULT_MODEL
        self._speed = max(0.25, min(4.0, speed))

        if self._voice not in self.VOICES:
            raise TTSError(
                f"Unknown voice: {self._voice}. Available: {', '.join(self.VOICES)}",
                provider="openai",
                voice=self._voice,
            )

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def current_voice(self) -> str:
        return self._voice

    async def synthesize(self, text: str, output_path: Path) -> AudioSegment:
        """Synthesize speech from text using OpenAI TTS."""
        try:
            response = self._client.audio.speech.create(
                model=self._model,
                voice=self._voice,
                input=text,
                speed=self._speed,
                response_format="mp3",
            )
            response.stream_to_file(str(output_path))
        except Exception as e:
            raise TTSError(
                f"Speech synthesis failed: {e}",
                provider="openai",
                voice=self._voice,
            )

        # Get actual duration from the generated audio file
        duration = await self._get_audio_duration(output_path)

        return AudioSegment(
            path=output_path,
            duration=duration,
            text=text,
        )

    async def list_voices(self) -> list[str]:
        """List available OpenAI TTS voices."""
        return self.VOICES.copy()

    async def _get_audio_duration(self, audio_path: Path) -> float:
        """Get the actual duration of an audio file in seconds."""
        try:
            from mutagen.mp3 import MP3

            audio = MP3(str(audio_path))
            return audio.info.length
        except ImportError:
            # Fall back to estimating from file size (~128kbps)
            file_size = audio_path.stat().st_size
            return file_size / (128 * 1024 / 8)
        except Exception:
            # Ultimate fallback: estimate based on text length
            return self._estimate_duration_fallback()

    def _estimate_duration_fallback(self) -> float:
        """Fallback duration estimation (only used if file reading fails)."""
        return 3.0  # Default fallback
