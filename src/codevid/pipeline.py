"""Main pipeline orchestration for Codevid."""

import asyncio
import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codevid import __version__
from codevid.audio.tts import AudioSegment, TTSProvider
from codevid.composer.editor import CompositionConfig, CompositionResult, VideoComposer
from codevid.llm.base import LLMProvider
from codevid.models import ActionType, NarrationSegment, ParsedTest, TestStep, VideoScript
from codevid.models.project import ProjectConfig
from codevid.parsers.base import TestParser
from codevid.recorder.screen import EventMarker


@dataclass
class PipelineConfig:
    """Configuration for the generation pipeline."""

    test_file: Path
    output: Path
    project_config: ProjectConfig
    app_name: str | None = None
    preview_mode: bool = False
    use_cached_narration_script: bool = False


@dataclass
class PipelineResult:
    """Result of pipeline execution."""

    output_path: Path | None
    script: VideoScript
    duration: float = 0.0
    success: bool = True
    error: str | None = None


class Pipeline:
    """Main orchestration pipeline for video generation."""

    def __init__(
        self,
        config: PipelineConfig,
        parser: TestParser | None = None,
        llm: LLMProvider | None = None,
        tts: TTSProvider | None = None,
    ):
        self.config = config
        self._parser = parser
        self._llm = llm
        self._tts = tts
        self._progress_callback: Callable[[int, str], None] | None = None

    def on_progress(self, callback: Callable[[int, str], None]) -> None:
        """Set progress callback (percent, message)."""
        self._progress_callback = callback

    def _report_progress(self, percent: int, message: str = "") -> None:
        if self._progress_callback:
            self._progress_callback(percent, message)

    def _get_temp_dir(self) -> Path:
        """Get the temporary directory path."""
        return self.config.output.parent / ".codevid_temp"

    def _get_script_cache_dir(self) -> Path:
        return self.config.project_config.output_dir / ".codevid_cache"

    def _get_script_cache_path(self) -> Path:
        resolved = str(self.config.test_file.resolve())
        path_hash = hashlib.sha256(resolved.encode("utf-8")).hexdigest()
        return self._get_script_cache_dir() / f"{path_hash}.json"

    def _compute_file_sha256(self, path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _script_fingerprint(self) -> dict[str, Any]:
        if self._llm is None:
            raise PipelineError("No LLM provider configured")

        return {
            "test_file_sha256": self._compute_file_sha256(self.config.test_file),
            "app_name": self.config.app_name or "the application",
            "llm_provider": self._llm.provider_name,
            "llm_model": self._llm.model_name,
            "codevid_version": __version__,
        }

    def _serialize_script(self, script: VideoScript) -> dict[str, Any]:
        return {
            "title": script.title,
            "introduction": script.introduction,
            "conclusion": script.conclusion,
            "total_estimated_duration": script.total_estimated_duration,
            "segments": [
                {
                    "text": seg.text,
                    "step_index": seg.step_index,
                    "timing_hint": seg.timing_hint,
                    "emphasis_words": list(seg.emphasis_words),
                }
                for seg in script.segments
            ],
        }

    def _deserialize_script(self, data: dict[str, Any]) -> VideoScript:
        segments_data = data.get("segments", [])
        segments: list[NarrationSegment] = []
        if isinstance(segments_data, list):
            for seg in segments_data:
                if not isinstance(seg, dict):
                    continue

                timing_hint = seg.get("timing_hint")
                segments.append(
                    NarrationSegment(
                        text=str(seg.get("text", "")),
                        step_index=int(seg.get("step_index", 0)),
                        timing_hint=float(timing_hint or 0.0),
                        emphasis_words=list(seg.get("emphasis_words") or []),
                    )
                )

        return VideoScript(
            title=str(data.get("title", "")),
            introduction=str(data.get("introduction", "")),
            segments=segments,
            conclusion=str(data.get("conclusion", "")),
            total_estimated_duration=float(data.get("total_estimated_duration", 0.0) or 0.0),
        )

    def _load_cached_script(self, parsed_test: ParsedTest) -> VideoScript | None:
        cache_path = self._get_script_cache_path()
        if not cache_path.exists():
            return None

        try:
            with cache_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return None

        if not isinstance(payload, dict):
            return None

        if payload.get("schema_version") != 1:
            return None

        expected = self._script_fingerprint()
        if payload.get("fingerprint") != expected:
            return None

        script_data = payload.get("script")
        if not isinstance(script_data, dict):
            return None

        script = self._deserialize_script(script_data)
        if any(
            seg.step_index < 0 or seg.step_index >= len(parsed_test.steps)
            for seg in script.segments
        ):
            return None

        return script

    def _save_cached_script(self, script: VideoScript) -> None:
        cache_dir = self._get_script_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._get_script_cache_path()

        payload: dict[str, Any] = {
            "schema_version": 1,
            "fingerprint": self._script_fingerprint(),
            "script": self._serialize_script(script),
        }

        tmp_file = None
        try:
            tmp_file = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(cache_dir),
                delete=False,
                prefix=cache_path.stem + ".",
                suffix=".tmp",
            )
            json.dump(payload, tmp_file, indent=2, sort_keys=True)
            tmp_file.flush()
            tmp_name = tmp_file.name
            tmp_file.close()
            Path(tmp_name).replace(cache_path)
        finally:
            if tmp_file is not None:
                try:
                    tmp_file.close()
                except Exception:
                    pass

    def _cleanup_temp(self) -> None:
        """Remove the temporary directory and all its contents."""
        temp_dir = self._get_temp_dir()
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    def run(self) -> PipelineResult:
        """Execute the pipeline synchronously."""
        return asyncio.run(self.run_async())

    async def run_async(self) -> PipelineResult:
        """Execute the full pipeline asynchronously."""
        try:
            # Step 1: Parse test file
            self._report_progress(5, "Parsing test file...")
            parsed_test = self._parse_test()
            self._report_progress(10, f"Found {len(parsed_test.steps)} test steps")

            # Step 2: Generate script with LLM (or load from cache)
            script: VideoScript | None = None
            if self.config.use_cached_narration_script:
                self._report_progress(15, "Checking for cached narration script...")
                script = self._load_cached_script(parsed_test)
                if script is not None:
                    self._report_progress(20, "Using cached narration script")

            if script is None:
                self._report_progress(15, "Generating narration script...")
                script = await self._generate_script(parsed_test)
                if self.config.use_cached_narration_script:
                    self._save_cached_script(script)

            generated_script_path = self.config.output.parent / "generated_script.txt"
            with open(generated_script_path, "w") as f:
                f.write(f"Title: {script.title}\n\n")
                f.write(f"Introduction:\n{script.introduction}\n\n")
                for i, segment in enumerate(script.segments):
                    f.write(f"Segment {i} (Step {segment.step_index}):\n{segment.text}\n\n")
                f.write(f"Conclusion:\n{script.conclusion}\n")
            self._report_progress(25, f"Script generated and saved to: {generated_script_path}")

            # Preview mode stops here
            if self.config.preview_mode:
                return PipelineResult(
                    output_path=None,
                    script=script,
                    success=True,
                )

            # Step 3: Generate audio narration
            self._report_progress(30, "Synthesizing audio...")
            audio_segments = await self._generate_audio(script)
            self._report_progress(45, f"Generated {len(audio_segments)} audio segments")

            # Step 3.5: Calculate per-step delays from audio durations
            step_delays = self._calculate_step_delays(
                script, audio_segments, num_steps=len(parsed_test.steps), test=parsed_test
            )

            # Step 4: Run test with recording (audio-synchronized timing)
            self._report_progress(50, "Recording test execution...")
            recording_path, markers = await self._record_test(parsed_test, step_delays)
            self._report_progress(75, "Recording complete")

            # Step 5: Compose final video
            self._report_progress(80, "Composing final video...")
            audio_paths = [seg.path for seg in audio_segments]
            result = self._compose_video(recording_path, script, audio_paths, markers)

            # Step 6: Cleanup temporary files
            self._report_progress(95, "Cleaning up temporary files...")
            self._cleanup_temp()
            self._report_progress(100, "Complete!")

            return PipelineResult(
                output_path=result.output_path,
                script=script,
                duration=result.duration,
                success=True,
            )

        except Exception as e:
            import traceback
            # Cleanup temp files even on failure
            self._cleanup_temp()
            return PipelineResult(
                output_path=None,
                script=VideoScript(title="", introduction="", segments=[], conclusion=""),
                success=False,
                error=f"{e}\n\n{traceback.format_exc()}",
            )

    def _parse_test(self) -> ParsedTest:
        """Parse the test file."""
        if self._parser is None:
            raise PipelineError("No parser configured")
        return self._parser.parse(self.config.test_file)

    async def _generate_script(self, test: ParsedTest) -> VideoScript:
        """Generate narration script using LLM."""
        if self._llm is None:
            raise PipelineError("No LLM provider configured")

        context = {"app_name": self.config.app_name or "the application"}
        return await self._llm.generate_script(test, context)

    async def _generate_audio(self, script: VideoScript) -> list[AudioSegment]:
        """Generate audio for all narration segments.

        Returns:
            List of AudioSegment objects with paths and durations.
            Order: [intro, segment_0, segment_1, ..., conclusion]
        """
        if self._tts is None or self._tts.provider_name == "none":
            raise PipelineError(
                "No text-to-speech provider is configured; narration audio cannot be generated. "
                "Set a TTS provider in codevid.yaml or pass --tts/--voice via the CLI."
            )

        audio_segments: list[AudioSegment] = []
        output_dir = self._get_temp_dir()
        output_dir.mkdir(exist_ok=True)

        # Generate intro audio
        intro_path = output_dir / "intro.mp3"
        intro_segment = await self._tts.synthesize(script.introduction, intro_path)
        audio_segments.append(intro_segment)

        # Generate segment audio
        for i, segment in enumerate(script.segments):
            segment_path = output_dir / f"segment_{i:03d}.mp3"
            audio_segment = await self._tts.synthesize(segment.text, segment_path)
            audio_segments.append(audio_segment)

        # Generate conclusion audio
        conclusion_path = output_dir / "conclusion.mp3"
        conclusion_segment = await self._tts.synthesize(script.conclusion, conclusion_path)
        audio_segments.append(conclusion_segment)

        return audio_segments

    def _calculate_step_delays(
        self,
        script: VideoScript,
        audio_segments: list[AudioSegment],
        num_steps: int,
        test: ParsedTest | None = None,
        min_delay: float = 0.5,
    ) -> list[float]:
        """Calculate per-step delays based on audio segment durations.

        Maps narration audio durations to test steps so each step waits
        for its narration to complete. Skipped steps get minimal delay.
        Block pauses are added after logical blocks (navigation, form submit, assertions).

        Args:
            script: The video script with segment-to-step mapping.
            audio_segments: Audio segments [intro, segment_0, ..., conclusion].
            num_steps: Total number of test steps.
            test: Optional parsed test to check skip_recording flags.
            min_delay: Minimum delay per step regardless of audio.

        Returns:
            List of delays (in seconds) for each test step.
        """
        # Build step_index -> total audio duration mapping
        # (handles multiple segments per step by summing)
        step_durations: dict[int, float] = {}

        for i, narration_seg in enumerate(script.segments):
            # Audio index is i+1 (skip intro at index 0)
            audio_idx = i + 1
            if audio_idx < len(audio_segments) - 1:  # Exclude conclusion
                duration = audio_segments[audio_idx].duration
                step_idx = narration_seg.step_index
                step_durations[step_idx] = step_durations.get(step_idx, 0.0) + duration

        # Generate delays for all steps
        delays: list[float] = []
        for step_idx in range(num_steps):
            # Skipped steps get minimal delay (just for execution stability)
            if test and step_idx < len(test.steps) and test.steps[step_idx].skip_recording:
                delays.append(0.3)
            else:
                audio_duration = step_durations.get(step_idx, 0.0)
                # Step delay should be at least audio duration, but not less than min_delay
                delay = max(audio_duration, min_delay)

                # Add block pause if enabled
                if test and step_idx < len(test.steps):
                    step = test.steps[step_idx]
                    delay += self._get_block_pause(step)

                delays.append(delay)

        return delays

    def _get_block_pause(self, step: TestStep) -> float:
        """Detect logical block end and return extra pause.

        Adds breathing room after major transitions:
        - Navigation: page loads need time to settle visually
        - Form submit: results need time to appear
        - Assertions: viewers need time to see validation

        Args:
            step: The current step.

        Returns:
            Extra pause in seconds (0.0 if no block pause needed).
        """
        block_config = self.config.project_config.block_pauses
        if not block_config.enabled:
            return 0.0

        if step.action == ActionType.NAVIGATE:
            return block_config.navigation
        if step.action == ActionType.ASSERT:
            return block_config.assertion
        if step.action == ActionType.CLICK and self._is_submit_button(step):
            return block_config.form_submit

        return 0.0

    def _is_submit_button(self, step: TestStep) -> bool:
        """Check if click target looks like a submit button.

        Detects common form submission patterns in selectors.

        Args:
            step: The click step to check.

        Returns:
            True if the target appears to be a submit button.
        """
        submit_indicators = [
            "submit",
            "save",
            "create",
            "send",
            "login",
            "sign",
            "confirm",
            "checkout",
            "complete",
            "finish",
            "register",
            "subscribe",
        ]
        target_lower = step.target.lower()
        return any(indicator in target_lower for indicator in submit_indicators)

    async def _record_test(
        self,
        test: ParsedTest,
        step_delays: list[float] | None = None,
    ) -> tuple[Path, list[EventMarker]]:
        """Execute test while recording screen.

        If the test has skip markers, uses phased execution:
        - Setup phase: execute skipped steps at beginning (no recording)
        - Recording phase: execute main steps (with recording)
        - Teardown phase: execute skipped steps at end (no recording)

        Args:
            test: The parsed test to execute.
            step_delays: Optional per-step delays (from audio durations).
        """
        from codevid.executor.playwright import ExecutorConfig, PlaywrightExecutor

        output_dir = self._get_temp_dir()
        output_dir.mkdir(exist_ok=True)
        video_dir = output_dir / "playwright_video"
        video_dir.mkdir(exist_ok=True)

        configured_resolution = self.config.project_config.recording.resolution
        if configured_resolution is not None:
            w, h = configured_resolution
            if w <= 0 or h <= 0:
                raise PipelineError(f"Invalid recording resolution: {configured_resolution}")

        # Check if anticipatory narration mode is enabled
        anticipatory = self.config.project_config.recording.narration_timing == "anticipatory"

        executor_config = ExecutorConfig(
            headless=True,
            slow_mo=100,
            viewport_width=(configured_resolution[0] if configured_resolution else 1280),
            viewport_height=(configured_resolution[1] if configured_resolution else 720),
            device_scale_factor=self.config.project_config.recording.device_scale_factor,
            step_delay=0.5,  # Fallback delay
            step_delays=step_delays,  # Per-step delays from audio durations
            record_video_dir=video_dir,
            record_video_size=configured_resolution,
            anticipatory_mode=anticipatory,
            show_cursor=self.config.project_config.recording.show_cursor,
        )
        executor = PlaywrightExecutor(executor_config)

        total_steps = len(test.steps)

        # Use phased execution if skip markers are present
        if test.has_skip_markers():
            setup_count = len(test.get_setup_steps())
            if setup_count > 0:
                self._report_progress(50, f"Executing {setup_count} setup steps (not recorded)...")

            result = await executor.execute_phased(
                test,
                on_step=lambda i, step: self._report_progress(
                    50 + int(25 * i / max(total_steps, 1)),
                    f"Executing step {i + 1}/{total_steps}: {step.description}",
                ),
            )

            if result.video_path is None:
                raise PipelineError("Playwright did not produce a video file")

            return result.video_path, result.markers
        else:
            # Original behavior for tests without skip markers
            markers, video_path = await executor.execute(
                test,
                recorder=None,
                on_step=lambda i, step: self._report_progress(
                    50 + int(25 * i / max(total_steps, 1)),
                    f"Executing step {i + 1}/{total_steps}: {step.description}",
                ),
            )

            if video_path is None:
                raise PipelineError("Playwright did not produce a video file")

            return video_path, markers

    def _compose_video(
        self,
        recording_path: Path,
        script: VideoScript,
        audio_segments: list[Path],
        markers: list[EventMarker],
    ) -> CompositionResult:
        """Compose final video from all components."""
        comp_config = CompositionConfig(
            output_path=self.config.output,
            include_captions=self.config.project_config.video.include_captions,
            theme=self.config.project_config.video.theme,
            intro_path=self.config.project_config.video.intro_template,
            outro_path=self.config.project_config.video.outro_template,
            watermark_path=self.config.project_config.video.watermark_path,
            watermark_position=self.config.project_config.video.watermark_position,
            fps=self.config.project_config.recording.fps,
            crf=self.config.project_config.video.crf,
            preset=self.config.project_config.video.preset,
            bitrate=self.config.project_config.video.bitrate,
            pixel_format=self.config.project_config.video.pixel_format,
            faststart=self.config.project_config.video.faststart,
        )

        composer = VideoComposer(comp_config)
        return composer.compose(recording_path, script, audio_segments, markers)


class PipelineError(Exception):
    """Raised when pipeline execution fails."""

    pass
