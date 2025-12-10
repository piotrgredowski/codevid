"""Edge TTS provider - free text-to-speech using Microsoft Edge."""

import asyncio
from pathlib import Path

from codevid.audio.tts import AudioSegment, TTSError, TTSProvider


class EdgeTTSProvider(TTSProvider):
    """TTS provider using Microsoft Edge's free TTS service."""

    DEFAULT_VOICE = "en-US-AriaNeural"

    # Common voices for quick reference
    VOICES = {
        # English (US)
        "aria": "en-US-AriaNeural",
        "guy": "en-US-GuyNeural",
        "jenny": "en-US-JennyNeural",
        # English (UK)
        "sonia": "en-GB-SoniaNeural",
        "ryan": "en-GB-RyanNeural",
        # English (AU)
        "natasha": "en-AU-NatashaNeural",
        "william": "en-AU-WilliamNeural",
    }

    def __init__(
        self,
        voice: str | None = None,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
    ):
        """Initialize Edge TTS provider.

        Args:
            voice: Voice name or shorthand (e.g., "en-US-AriaNeural" or "aria").
            rate: Speech rate adjustment (e.g., "+10%", "-20%").
            volume: Volume adjustment (e.g., "+50%", "-10%").
            pitch: Pitch adjustment (e.g., "+10Hz", "-5Hz").
        """
        # Resolve voice shorthand
        voice = voice or self.DEFAULT_VOICE
        self._voice = self.VOICES.get(voice.lower(), voice)
        self._rate = rate
        self._volume = volume
        self._pitch = pitch

    @property
    def provider_name(self) -> str:
        return "edge"

    @property
    def current_voice(self) -> str:
        return self._voice

    async def synthesize(self, text: str, output_path: Path) -> AudioSegment:
        """Synthesize speech from text using Edge TTS."""
        try:
            import edge_tts
        except ImportError:
            raise TTSError(
                "edge-tts package is required. Install with: pip install edge-tts",
                provider="edge",
            )

        try:
            communicate = edge_tts.Communicate(
                text,
                self._voice,
                rate=self._rate,
                volume=self._volume,
                pitch=self._pitch,
            )
            await communicate.save(str(output_path))
        except Exception as e:
            raise TTSError(f"Speech synthesis failed: {e}", provider="edge", voice=self._voice)

        # Calculate duration from the audio file
        duration = await self._get_audio_duration(output_path)

        return AudioSegment(
            path=output_path,
            duration=duration,
            text=text,
        )

    async def list_voices(self) -> list[str]:
        """List all available Edge TTS voices."""
        try:
            import edge_tts
        except ImportError:
            raise TTSError(
                "edge-tts package is required. Install with: pip install edge-tts",
                provider="edge",
            )

        try:
            voices = await edge_tts.list_voices()
            return [v["ShortName"] for v in voices]
        except Exception as e:
            raise TTSError(f"Failed to list voices: {e}", provider="edge")

    async def _get_audio_duration(self, audio_path: Path) -> float:
        """Get the duration of an audio file in seconds."""
        try:
            # Try using mutagen for accurate duration
            from mutagen.mp3 import MP3

            audio = MP3(str(audio_path))
            return audio.info.length
        except ImportError:
            # Fall back to estimating from file size
            # Edge TTS outputs at ~128kbps
            file_size = audio_path.stat().st_size
            return file_size / (128 * 1024 / 8)
        except Exception:
            # Estimate based on text length (~150 words per minute)
            return 3.0  # Default fallback
