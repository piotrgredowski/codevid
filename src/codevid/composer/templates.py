"""Video templates and themes for composition."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CaptionStyle:
    """Styling for video captions."""

    font: str = "Arial"
    font_size: int = 28
    color: str = "white"
    bg_color: str = "rgba(0,0,0,0.7)"
    position: str = "bottom"  # top, center, bottom
    margin_bottom: int = 50
    stroke_color: str | None = "black"
    stroke_width: int = 1


@dataclass
class TransitionStyle:
    """Styling for video transitions."""

    type: str = "crossfade"  # crossfade, fade, cut
    duration: float = 0.5


@dataclass
class VideoTheme:
    """Complete theme configuration for video output."""

    name: str
    caption_style: CaptionStyle = field(default_factory=CaptionStyle)
    transition_style: TransitionStyle = field(default_factory=TransitionStyle)

    # Click highlight
    click_highlight_color: tuple[int, int, int] = (255, 100, 100)
    click_highlight_radius: int = 30

    # Step indicator
    step_indicator_enabled: bool = True
    step_indicator_bg: str = "rgba(0,0,0,0.7)"
    step_indicator_color: str = "white"

    # General
    background_color: tuple[int, int, int] = (30, 30, 30)


# Predefined themes
THEMES: dict[str, VideoTheme] = {
    "default": VideoTheme(
        name="default",
        caption_style=CaptionStyle(
            font_size=28,
            color="white",
            bg_color="rgba(0,0,0,0.7)",
        ),
    ),
    "modern": VideoTheme(
        name="modern",
        caption_style=CaptionStyle(
            font="Helvetica",
            font_size=32,
            color="white",
            bg_color="rgba(33,33,33,0.85)",
            stroke_color=None,
        ),
        click_highlight_color=(100, 180, 255),
        transition_style=TransitionStyle(type="crossfade", duration=0.3),
    ),
    "minimal": VideoTheme(
        name="minimal",
        caption_style=CaptionStyle(
            font_size=24,
            color="#f0f0f0",
            bg_color="transparent",
            stroke_color="black",
            stroke_width=2,
        ),
        step_indicator_enabled=False,
        transition_style=TransitionStyle(type="cut", duration=0),
    ),
    "professional": VideoTheme(
        name="professional",
        caption_style=CaptionStyle(
            font="Georgia",
            font_size=26,
            color="white",
            bg_color="rgba(0,51,102,0.8)",
            position="bottom",
            margin_bottom=60,
        ),
        click_highlight_color=(255, 200, 50),
        click_highlight_radius=25,
    ),
    "dark": VideoTheme(
        name="dark",
        caption_style=CaptionStyle(
            font_size=28,
            color="#e0e0e0",
            bg_color="rgba(18,18,18,0.9)",
        ),
        click_highlight_color=(180, 100, 255),
        background_color=(18, 18, 18),
    ),
}


def get_theme(name: str) -> VideoTheme:
    """Get a theme by name.

    Args:
        name: Theme name (default, modern, minimal, professional, dark).

    Returns:
        VideoTheme configuration.
    """
    return THEMES.get(name, THEMES["default"])


def list_themes() -> list[str]:
    """List available theme names."""
    return list(THEMES.keys())


def apply_caption_style(
    text_clip: Any,
    style: CaptionStyle,
    video_height: int,
) -> Any:
    """Apply caption style to a MoviePy TextClip.

    Args:
        text_clip: MoviePy TextClip object.
        style: Caption style configuration.
        video_height: Height of the video for positioning.

    Returns:
        Styled TextClip.
    """
    # Position based on style
    if style.position == "top":
        y_pos = 50
    elif style.position == "center":
        y_pos = video_height // 2
    else:  # bottom
        y_pos = video_height - style.margin_bottom

    text_clip = text_clip.with_position(("center", y_pos))

    return text_clip
