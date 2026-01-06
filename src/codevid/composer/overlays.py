"""Visual overlays for video composition."""

from dataclasses import dataclass
from typing import Any

from codevid.recorder.screen import EventMarker


@dataclass
class OverlayConfig:
    """Configuration for video overlays."""

    # Click highlight settings
    click_highlight_enabled: bool = True
    click_highlight_color: tuple[int, int, int] = (255, 100, 100)  # RGB
    click_highlight_radius: int = 30
    click_highlight_duration: float = 0.5  # seconds

    # Mouse cursor spotlight (yellow circle that follows clicks)
    cursor_spotlight_enabled: bool = False
    cursor_spotlight_color: tuple[int, int, int] = (255, 220, 50)  # Yellow
    cursor_spotlight_radius: int = 25
    cursor_spotlight_opacity: float = 0.7
    cursor_spotlight_pulse_scale: float = 1.5  # How much to expand on click

    # Step indicator
    step_indicator_enabled: bool = True
    step_indicator_position: str = "top-left"  # top-left, top-right, bottom-left, bottom-right


class OverlayGenerator:
    """Generate visual overlays for video composition."""

    def __init__(self, config: OverlayConfig | None = None):
        self.config = config or OverlayConfig()

    def create_click_highlights(
        self,
        markers: list[EventMarker],
        video_size: tuple[int, int],
    ) -> list[dict[str, Any]]:
        """Create click highlight overlay definitions.

        Returns a list of overlay specifications that can be applied
        to the video during composition.
        """
        if not self.config.click_highlight_enabled:
            return []

        highlights = []
        for marker in markers:
            if marker.event_type == "click" and "x" in marker.metadata and "y" in marker.metadata:
                highlights.append({
                    "type": "click_ripple",
                    "timestamp": marker.timestamp,
                    "duration": self.config.click_highlight_duration,
                    "x": marker.metadata["x"],
                    "y": marker.metadata["y"],
                    "radius": self.config.click_highlight_radius,
                    "color": self.config.click_highlight_color,
                })

        return highlights

    def create_step_indicators(
        self,
        markers: list[EventMarker],
        video_size: tuple[int, int],
    ) -> list[dict[str, Any]]:
        """Create step indicator overlay definitions."""
        if not self.config.step_indicator_enabled:
            return []

        indicators = []
        step_count = 0

        for marker in markers:
            if marker.event_type == "step_start":
                step_count += 1
                # Find the corresponding step_end
                end_time = None
                for end_marker in markers:
                    if (end_marker.event_type == "step_end" and
                        end_marker.timestamp > marker.timestamp and
                        end_marker.metadata.get("index") == marker.metadata.get("index")):
                        end_time = end_marker.timestamp
                        break

                indicators.append({
                    "type": "step_indicator",
                    "timestamp": marker.timestamp,
                    "end_time": end_time,
                    "step_number": step_count,
                    "action": marker.metadata.get("action", ""),
                    "position": self.config.step_indicator_position,
                })

        return indicators

    def create_cursor_spotlights(
        self,
        markers: list[EventMarker],
        video_size: tuple[int, int],
    ) -> list[dict[str, Any]]:
        """Create cursor spotlight overlay definitions.

        Creates yellow circle spotlights at click/hover positions with
        a pulse animation on click actions.

        Args:
            markers: Event markers from test execution.
            video_size: (width, height) of the video.

        Returns:
            List of spotlight overlay specifications.
        """
        if not self.config.cursor_spotlight_enabled:
            return []

        spotlights = []
        for i, marker in enumerate(markers):
            if marker.event_type != "step_start":
                continue
            if "x" not in marker.metadata or "y" not in marker.metadata:
                continue

            # Find the corresponding step_end to get duration
            end_time = None
            for end_marker in markers:
                if (end_marker.event_type == "step_end" and
                    end_marker.timestamp > marker.timestamp and
                    end_marker.metadata.get("index") == marker.metadata.get("index")):
                    end_time = end_marker.timestamp
                    break

            if end_time is None:
                continue

            action = marker.metadata.get("action", "")
            is_click = action == "click"

            spotlights.append({
                "type": "cursor_spotlight",
                "timestamp": marker.timestamp,
                "duration": end_time - marker.timestamp,
                "x": marker.metadata["x"],
                "y": marker.metadata["y"],
                "radius": self.config.cursor_spotlight_radius,
                "color": self.config.cursor_spotlight_color,
                "opacity": self.config.cursor_spotlight_opacity,
                "is_click": is_click,
                "pulse_scale": self.config.cursor_spotlight_pulse_scale if is_click else 1.0,
            })

        return spotlights

    def apply_overlays_moviepy(
        self,
        video_clip: Any,
        overlays: list[dict[str, Any]],
    ) -> Any:
        """Apply overlays to a MoviePy video clip.

        Args:
            video_clip: MoviePy VideoClip object.
            overlays: List of overlay specifications.

        Returns:
            VideoClip with overlays applied.
        """
        try:
            from moviepy import CompositeVideoClip
        except ImportError:
            return video_clip

        overlay_clips = []

        for overlay in overlays:
            if overlay["type"] == "click_ripple":
                clip = self._create_click_ripple_clip(overlay, video_clip.size)
                if clip:
                    overlay_clips.append(clip)
            elif overlay["type"] == "step_indicator":
                clip = self._create_step_indicator_clip(overlay, video_clip)
                if clip:
                    overlay_clips.append(clip)
            elif overlay["type"] == "cursor_spotlight":
                clip = self._create_cursor_spotlight_clip(overlay, video_clip.size)
                if clip:
                    overlay_clips.append(clip)

        if overlay_clips:
            return CompositeVideoClip([video_clip, *overlay_clips])

        return video_clip

    def _create_click_ripple_clip(
        self,
        overlay: dict[str, Any],
        video_size: tuple[int, int],
    ) -> Any | None:
        """Create a click ripple effect clip."""
        try:
            from moviepy import vfx
            from moviepy import ColorClip
            import numpy as np
        except ImportError:
            return None

        x, y = overlay["x"], overlay["y"]
        radius = overlay["radius"]
        duration = overlay["duration"]
        timestamp = overlay["timestamp"]
        color = overlay["color"]

        # Create a simple colored circle that fades out
        def make_frame(t):
            # Create transparent frame
            frame = np.zeros((video_size[1], video_size[0], 4), dtype=np.uint8)

            # Calculate fade (1.0 at start, 0.0 at end)
            fade = 1.0 - (t / duration)
            current_radius = int(radius * (1 + t / duration))  # Expand over time

            # Draw circle (simplified - just a filled circle)
            y_coords, x_coords = np.ogrid[:video_size[1], :video_size[0]]
            mask = (x_coords - x) ** 2 + (y_coords - y) ** 2 <= current_radius ** 2

            frame[mask] = [*color, int(200 * fade)]

            return frame[:, :, :3]  # Return RGB only

        # For simplicity, return a basic ColorClip positioned at the click
        # In production, you'd use make_frame with a custom clip
        clip = ColorClip(
            size=(radius * 2, radius * 2),
            color=color,
            duration=duration,
        )
        clip = clip.with_position((x - radius, y - radius))
        clip = clip.with_start(timestamp)
        clip = clip.with_effects([vfx.CrossFadeOut(duration)])
        clip = clip.with_opacity(0.6)

        return clip

    def _create_step_indicator_clip(
        self,
        overlay: dict[str, Any],
        video_clip: Any,
    ) -> Any | None:
        """Create a step indicator text clip."""
        try:
            from moviepy import TextClip
        except ImportError:
            return None

        step_num = overlay["step_number"]
        action = overlay["action"]
        timestamp = overlay["timestamp"]
        end_time = overlay.get("end_time") or timestamp + 3.0
        position = overlay["position"]

        text = f"Step {step_num}"
        if action:
            text += f": {action}"

        clip = self._safe_text_clip(
            text,
            font_size=24,
            color="white",
            bg_color="rgba(0,0,0,0.7)",
            font="Arial",
        )
        if clip is None:
            return None

        # Position based on config
        pos_map = {
            "top-left": (20, 20),
            "top-right": (video_clip.w - clip.w - 20, 20),
            "bottom-left": (20, video_clip.h - clip.h - 20),
            "bottom-right": (video_clip.w - clip.w - 20, video_clip.h - clip.h - 20),
        }

        clip = clip.with_position(pos_map.get(position, (20, 20)))
        clip = clip.with_start(timestamp)
        clip = clip.with_duration(end_time - timestamp)

        return clip

    def _create_cursor_spotlight_clip(
        self,
        overlay: dict[str, Any],
        video_size: tuple[int, int],
    ) -> Any | None:
        """Create a yellow cursor spotlight with optional pulse animation on click.

        The spotlight is a semi-transparent yellow circle that appears at the
        click location. For click actions, it starts at pulse_scale size and
        shrinks to normal size, creating a visual "pulse" effect.
        """
        try:
            from moviepy import VideoClip
            import numpy as np
        except ImportError:
            return None

        x, y = overlay["x"], overlay["y"]
        base_radius = overlay["radius"]
        duration = min(overlay["duration"], 3.0)  # Cap duration for performance
        timestamp = overlay["timestamp"]
        color = overlay["color"]
        opacity = overlay["opacity"]
        is_click = overlay.get("is_click", False)
        pulse_scale = overlay.get("pulse_scale", 1.5) if is_click else 1.0

        # Create animated spotlight using make_frame
        def make_frame(t: float) -> Any:
            # Create RGBA frame
            frame = np.zeros((video_size[1], video_size[0], 4), dtype=np.uint8)

            # Calculate current radius (pulse effect: start large, shrink to normal)
            if is_click and duration > 0:
                # Pulse happens in first 0.3 seconds
                pulse_duration = min(0.3, duration)
                if t < pulse_duration:
                    # Shrink from pulse_scale to 1.0
                    scale = pulse_scale - (pulse_scale - 1.0) * (t / pulse_duration)
                else:
                    scale = 1.0
            else:
                scale = 1.0

            current_radius = int(base_radius * scale)

            # Draw filled circle with anti-aliasing approximation
            y_coords, x_coords = np.ogrid[:video_size[1], :video_size[0]]
            dist_sq = (x_coords - x) ** 2 + (y_coords - y) ** 2
            radius_sq = current_radius ** 2

            # Soft edge (anti-aliasing)
            edge_width = 3
            inner_radius_sq = max(0, current_radius - edge_width) ** 2

            # Inner solid circle
            mask_inner = dist_sq <= inner_radius_sq
            frame[mask_inner] = [color[0], color[1], color[2], int(255 * opacity)]

            # Outer gradient edge
            mask_edge = (dist_sq > inner_radius_sq) & (dist_sq <= radius_sq)
            if np.any(mask_edge):
                # Calculate alpha gradient for edge pixels
                edge_dist = np.sqrt(dist_sq[mask_edge]) - (current_radius - edge_width)
                edge_alpha = 1.0 - (edge_dist / edge_width)
                edge_alpha = np.clip(edge_alpha, 0, 1)
                frame[mask_edge, 0] = color[0]
                frame[mask_edge, 1] = color[1]
                frame[mask_edge, 2] = color[2]
                frame[mask_edge, 3] = (edge_alpha * 255 * opacity).astype(np.uint8)

            return frame

        # Create video clip
        clip = VideoClip(make_frame, duration=duration)
        clip = clip.with_start(timestamp)

        # Extract RGB and use alpha as mask
        def get_rgb(get_frame: Any) -> Any:
            def rgb_frame(t: float) -> Any:
                return get_frame(t)[:, :, :3]
            return rgb_frame

        def get_alpha(get_frame: Any) -> Any:
            def alpha_frame(t: float) -> Any:
                return get_frame(t)[:, :, 3] / 255.0
            return alpha_frame

        # Create RGB clip with alpha mask
        from moviepy import VideoClip as VC
        rgb_clip = VC(get_rgb(make_frame), duration=duration)
        alpha_clip = VC(get_alpha(make_frame), duration=duration, is_mask=True)
        rgb_clip = rgb_clip.with_mask(alpha_clip)
        rgb_clip = rgb_clip.with_start(timestamp)

        return rgb_clip

    def _safe_text_clip(self, text: str, **kwargs: Any) -> Any | None:
        """Build a TextClip but fall back to default font if the requested font is missing."""
        try:
            from moviepy import TextClip
        except ImportError:
            return None

        try:
            return TextClip(text=text, **kwargs)
        except Exception:
            font = kwargs.pop("font", None)
            try:
                return TextClip(text=text, font=None, **kwargs)
            except Exception:
                # Still failed; skip overlay gracefully.
                return None
