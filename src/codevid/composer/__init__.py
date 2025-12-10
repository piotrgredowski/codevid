"""Video composition and editing."""

from codevid.composer.captions import Caption, CaptionGenerator
from codevid.composer.editor import CompositionConfig, CompositionError, CompositionResult, VideoComposer
from codevid.composer.overlays import OverlayConfig, OverlayGenerator
from codevid.composer.templates import VideoTheme, get_theme, list_themes

__all__ = [
    "VideoComposer",
    "CompositionConfig",
    "CompositionResult",
    "CompositionError",
    "Caption",
    "CaptionGenerator",
    "OverlayConfig",
    "OverlayGenerator",
    "VideoTheme",
    "get_theme",
    "list_themes",
]
