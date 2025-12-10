"""Audio generation and text-to-speech."""

from codevid.audio.tts import AudioSegment, TTSError, TTSProvider
from codevid.audio.factory import create_tts_provider, get_provider_for_name

__all__ = [
    "AudioSegment",
    "TTSProvider",
    "TTSError",
    "create_tts_provider",
    "get_provider_for_name",
]
