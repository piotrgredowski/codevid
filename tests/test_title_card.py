"""Tests for title card rendering logic."""

from pathlib import Path

from codevid.composer.editor import CompositionConfig, VideoComposer


def test_title_card_uses_explicit_caption_height(tmp_path: Path) -> None:
    composer = VideoComposer(CompositionConfig(output_path=tmp_path / "out.mp4"))

    captured: dict[str, object] = {}

    # Avoid font/text rendering in this unit test; assert parameters instead.
    def fake_create_text_clip(text: str, *, font: str | None = None, **kwargs: object):
        from moviepy import ColorClip

        captured["text"] = text
        captured["size"] = kwargs.get("size")
        return ColorClip(size=(10, 10), color=(255, 255, 255), duration=1.0)

    composer._create_text_clip = fake_create_text_clip  # type: ignore[method-assign]

    composer._create_title_card(
        "Automated Coin Flip Test With Extra Words",
        duration=1.0,
        size=(720, 405),
    )

    size = captured["size"]
    assert isinstance(size, tuple)
    assert len(size) == 2
    assert size[0] == int(720 * 0.8)
    assert size[1] is not None
