"""Kokoro TTS provider."""

import logging
from pathlib import Path

from codevid.audio.tts import AudioSegment, TTSError, TTSProvider

logger = logging.getLogger(__name__)


class KokoroTTSProvider(TTSProvider):
    """Local TTS provider using Kokoro via the 'kokoro' package."""

    def __init__(self, voice: str | None = None, speed: float = 1.0) -> None:
        """Initialize Kokoro TTS provider.

        Args:
            voice: Voice ID (default: 'af_bella').
            speed: Speed multiplier (default: 1.0).
        """
        import warnings

        # Suppress specific Kokoro/Torch warnings
        warnings.filterwarnings(
            "ignore", message=".*dropout option adds dropout after all but last recurrent layer.*"
        )
        warnings.filterwarnings("ignore", message=".*weight_norm.*is deprecated.*")

        try:
            from kokoro import KPipeline  # type: ignore[import-not-found]
        except ImportError as e:
            raise TTSError(
                "Kokoro dependencies not installed. Install with: uv sync --group local-kokoro",
                provider="local_kokoro",
            ) from e

        # Optional hardening to avoid runtime downloads when optional deps are installed.
        try:
            import espeakng_loader  # type: ignore[import-not-found]
            from phonemizer.backend.espeak.wrapper import (  # type: ignore[import-not-found]
                EspeakWrapper,
            )

            EspeakWrapper.set_library(espeakng_loader.get_library_path())
            EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
        except ImportError:
            pass

        try:
            import spacy.util  # type: ignore[import-not-found]

            if not spacy.util.is_package("en_core_web_sm"):
                logger.warning(
                    "spaCy model 'en_core_web_sm' not found. "
                    "If Kokoro attempts to download it via pip and fails, run: "
                    "uv sync --group local-kokoro"
                )
        except ImportError:
            pass

        # Set defaults
        self._voice = voice or "af_bella"  # Common default female voice
        self._speed = speed

        # Initialize pipeline only once to save resources
        # 'a' = American English
        try:
            # We must pass repo_id to avoid the warning, and verify it doesn't trigger pip.
            self._pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
        except Exception as e:
            # If it fails due to pip, we might need to explain it better or fix the env.
            raise TTSError(
                f"Failed to initialize Kokoro pipeline: {e}",
                provider="local_kokoro"
            ) from e

    @property
    def provider_name(self) -> str:
        return "local_kokoro"

    @property
    def current_voice(self) -> str:
        return self._voice

    async def synthesize(self, text: str, output_path: Path) -> AudioSegment:
        """Synthesize speech from text using Kokoro."""
        try:
            import soundfile as sf  # type: ignore[import-not-found]
        except ImportError as e:
            raise TTSError(
                "Kokoro dependencies not installed. Install with: uv sync --group local-kokoro",
                provider="local_kokoro",
            ) from e

        try:
            # Generate audio
            # Kokoro returns a generator of (graphemes, phonemes, audio)
            # We'll concatenate all audio segments

            # Simple wrapper to just get audio for the full text
            # We treat the whole text as one block for simplicity in this integration
            # though Kokoro handles splitting internally.

            generator = self._pipeline(
                text,
                voice=self._voice,
                speed=self._speed,
                split_pattern=r"\n+",
            )

            import numpy as np
            all_audio = []

            for _, _, audio in generator:
                if audio is not None:
                    all_audio.append(audio)

            if not all_audio:
                raise TTSError("No audio generated", provider="local_kokoro")

            # Concatenate all numpy arrays
            final_audio = np.concatenate(all_audio)

            # Save to file
            # Kokoro usually outputs at 24000Hz
            sample_rate = 24000
            sf.write(str(output_path), final_audio, sample_rate)

            # Calculate duration
            duration = len(final_audio) / sample_rate

            return AudioSegment(
                path=output_path,
                duration=duration,
                text=text,
            )

        except Exception as e:
            raise TTSError(
                f"Kokoro synthesis failed: {e}",
                provider="local_kokoro",
            ) from e

    async def list_voices(self) -> list[str]:
        """List available voices."""
        # Hardcoded list of common voices available in Kokoro v0.19+
        # This list might need updating as the model evolves
        return [
            "af_bella", "af_nicole", "af_sarah", "af_sky",
            "am_adam", "am_michael",
            "bf_emma", "bf_isabella",
            "bm_george", "bm_lewis",
        ]
