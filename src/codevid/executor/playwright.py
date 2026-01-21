"""Playwright test executor for running parsed tests."""

import asyncio
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from playwright.async_api import Browser, Page, async_playwright
from playwright.async_api import expect as async_expect

from codevid.executor.mouse_helper import MOUSE_HELPER_JS
from codevid.models import ActionType, ParsedTest, TestStep
from codevid.recorder.screen import EventMarker

if TYPE_CHECKING:
    from codevid.recorder.screen import ScreenRecorder


@dataclass
class ExecutorConfig:
    """Configuration for test execution."""

    headless: bool = False  # Screen recording requires headed; Playwright video works headless.
    browser_type: str = "chromium"
    slow_mo: int = 100  # Milliseconds between Playwright actions
    viewport_width: int = 1280
    viewport_height: int = 720
    device_scale_factor: float | None = None
    step_delay: float = 0.5  # Default delay after each step (fallback)
    step_delays: list[float] | None = None  # Per-step delays (overrides step_delay)
    record_video_dir: Path | None = None
    record_video_size: tuple[int, int] | None = None  # Defaults to viewport size
    anticipatory_mode: bool = False  # If True, wait before action (audio plays first)
    show_cursor: bool = False  # Inject visible cursor element in recordings


@dataclass
class PhasedExecutionResult:
    """Result of phased test execution."""

    markers: list[EventMarker]
    video_path: Path | None
    setup_successful: bool = True
    teardown_successful: bool = True


class PlaywrightExecutor:
    """Execute parsed Playwright tests with recording integration."""

    def __init__(self, config: ExecutorConfig | None = None):
        self.config = config or ExecutorConfig()
        self._browser: Browser | None = None
        self._page: Page | None = None

    def _get_step_delay(self, step_index: int) -> float:
        """Get the delay for a specific step.

        Uses per-step delays if available, otherwise falls back to default.
        """
        if self.config.step_delays and step_index < len(self.config.step_delays):
            return self.config.step_delays[step_index]
        return self.config.step_delay

    async def execute(
        self,
        test: ParsedTest,
        recorder: "ScreenRecorder | None" = None,
        on_step: Callable[[int, TestStep], None] | None = None,
    ) -> tuple[list["EventMarker"], Path | None]:
        """Execute all steps in a parsed test.

        Args:
            test: The parsed test to execute.
            recorder: Optional screen recorder for event marking.
            on_step: Optional callback called before each step.

        Returns:
            Tuple of (event markers, recorded video path if available).
        """
        markers: list[EventMarker] = []
        recorded_video: Path | None = None

        async with async_playwright() as p:
            # Launch browser (visible for recording)
            browser_launcher = getattr(p, self.config.browser_type)
            self._browser = await browser_launcher.launch(
                headless=self.config.headless,
                slow_mo=self.config.slow_mo,
            )

            context_kwargs: dict[str, object] = {
                "viewport": {
                    "width": self.config.viewport_width,
                    "height": self.config.viewport_height,
                }
            }
            if self.config.device_scale_factor is not None:
                context_kwargs["device_scale_factor"] = self.config.device_scale_factor

            if self.config.record_video_dir:
                context_kwargs["record_video_dir"] = str(self.config.record_video_dir)
                if self.config.record_video_size:
                    w, h = self.config.record_video_size
                    context_kwargs["record_video_size"] = {"width": w, "height": h}

            context = await self._browser.new_context(**context_kwargs)

            # Inject mouse helper script for visible cursor in recordings

            if self.config.show_cursor:
                await context.add_init_script(MOUSE_HELPER_JS)

            self._page = await context.new_page()

            # Reset timer to align with the start of video recording (which begins at page creation)
            start_time = time.time()

            try:
                for i, step in enumerate(test.steps):
                    # Notify callback
                    if on_step:
                        on_step(i, step)

                    # Mark step start (always, for video composition timing)
                    step_start_time = time.time() - start_time
                    if recorder and recorder.is_recording:
                        marker = recorder.mark_event(
                            "step_start",
                            {
                                "index": i,
                                "action": step.action.value,
                                "target": step.target,
                                "description": step.description,
                            },
                        )
                    else:
                        marker = EventMarker(
                            timestamp=step_start_time,
                            event_type="step_start",
                            metadata={
                                "index": i,
                                "action": step.action.value,
                                "target": step.target,
                                "description": step.description,
                            },
                        )
                    markers.append(marker)

                    # Get step delay
                    delay = self._get_step_delay(i)

                    # Anticipatory mode: wait for narration BEFORE action
                    # (viewer hears what will happen, then sees it)
                    if self.config.anticipatory_mode:
                        if delay > 0:
                            await asyncio.sleep(delay)
                        click_coords = await self._execute_step(step)
                        # Brief hold after action so viewer can see result
                        await asyncio.sleep(0.5)
                    else:
                        # Standard mode: execute action, then wait
                        click_coords = await self._execute_step(step)
                        if delay > 0:
                            await asyncio.sleep(delay)
                        # Safety Buffer: Add a gap between steps for visual stabilization
                        await asyncio.sleep(0.5)

                    # Update the last step_start marker with click coordinates
                    if click_coords and markers:
                        for m in reversed(markers):
                            if m.event_type == "step_start" and m.metadata.get("index") == i:
                                m.metadata["x"] = click_coords[0]
                                m.metadata["y"] = click_coords[1]
                                break

                    # Mark step end AFTER the safety buffer (for accurate composition timing)
                    step_end_time = time.time() - start_time
                    if recorder and recorder.is_recording:
                        marker = recorder.mark_event(
                            "step_end",
                            {
                                "index": i,
                            },
                        )
                    else:
                        marker = EventMarker(
                            timestamp=step_end_time,
                            event_type="step_end",
                            metadata={"index": i},
                        )
                    markers.append(marker)

            finally:
                if self._page:
                    try:
                        # Close page to ensure video is saved
                        await self._page.close()
                        # Capture video path
                        if self._page.video:
                            path = await self._page.video.path()
                            recorded_video = Path(path)
                    except Exception:
                        pass
                await context.close()
                await self._browser.close()

        return markers, recorded_video

    async def execute_segmented(
        self,
        test: ParsedTest,
        on_step: Callable[[int, TestStep], None] | None = None,
    ) -> list[Path]:
        """Execute test step-by-step, recording each segment separately.

        Each step is recorded in a separate browser context with video recording.
        State (cookies, localStorage) is persisted between steps using storage_state.

        Args:
            test: The parsed test to execute.
            on_step: Optional callback called before each step.

        Returns:
            List of video file paths, one per step.
        """
        if not self.config.record_video_dir:
            raise ExecutorError("record_video_dir must be set for segmented recording")

        segment_videos: list[Path] = []
        state_file = self.config.record_video_dir / "state.json"
        current_url: str | None = None

        # Ensure segment directories exist
        self.config.record_video_dir.mkdir(parents=True, exist_ok=True)

        async with async_playwright() as p:
            browser_launcher = getattr(p, self.config.browser_type)
            browser = await browser_launcher.launch(
                headless=self.config.headless,
                slow_mo=self.config.slow_mo,
            )

            for i, step in enumerate(test.steps):
                # Notify callback
                if on_step:
                    on_step(i, step)

                # Create directory for this step's video
                step_video_dir = self.config.record_video_dir / f"step_{i}"
                step_video_dir.mkdir(parents=True, exist_ok=True)

                # Build context kwargs with video recording
                context_kwargs: dict[str, object] = {
                    "viewport": {
                        "width": self.config.viewport_width,
                        "height": self.config.viewport_height,
                    },
                    "record_video_dir": str(step_video_dir),
                }

                if self.config.device_scale_factor is not None:
                    context_kwargs["device_scale_factor"] = self.config.device_scale_factor

                if self.config.record_video_size:
                    w, h = self.config.record_video_size
                    context_kwargs["record_video_size"] = {"width": w, "height": h}

                # Restore state from previous step if available
                if state_file.exists():
                    context_kwargs["storage_state"] = str(state_file)

                # Create new context with video recording
                context = await browser.new_context(**context_kwargs)

                # Inject mouse helper script for visible cursor in recordings
                if self.config.show_cursor:
                    await context.add_init_script(MOUSE_HELPER_JS)

                page = await context.new_page()
                self._page = page

                try:
                    # Navigate to last known URL for state continuity
                    # (skip if this step is a navigation itself)
                    if current_url and step.action != ActionType.NAVIGATE:
                        try:
                            await page.goto(current_url, wait_until="domcontentloaded")
                        except Exception:
                            pass  # URL may have changed, continue anyway

                    # Execute the step
                    await self._execute_step(step)

                    # Wait for page to stabilize (animations, network requests)
                    try:
                        await page.wait_for_load_state("networkidle", timeout=3000)
                    except Exception:
                        pass  # Timeout is fine, just continue

                    # Visual stabilization delay
                    await asyncio.sleep(1.0)

                    # Save current URL and state for next step
                    current_url = page.url
                    await context.storage_state(path=str(state_file))

                finally:
                    # Close page to finalize video
                    video_path: Path | None = None
                    try:
                        if page.video:
                            video_path = Path(await page.video.path())
                        await page.close()
                    except Exception:
                        pass

                    await context.close()

                    # Find the recorded video file
                    if video_path and video_path.exists():
                        segment_videos.append(video_path)
                    else:
                        # Fallback: search for video in directory
                        video_files = list(step_video_dir.glob("*.webm"))
                        if video_files:
                            segment_videos.append(video_files[0])

            await browser.close()

        return segment_videos

    async def execute_phased(
        self,
        test: ParsedTest,
        on_step: Callable[[int, TestStep], None] | None = None,
    ) -> PhasedExecutionResult:
        """Execute test in phases: setup (no recording) -> main (recording) -> teardown.

        Uses browser storage_state to persist session between contexts.
        Steps marked with skip_recording=True at the beginning run in setup phase,
        at the end run in teardown phase, and the middle section is recorded.

        Args:
            test: The parsed test to execute.
            on_step: Optional callback called before each step with (index, step).

        Returns:
            PhasedExecutionResult with markers, video path, and phase success flags.
        """
        if not self.config.record_video_dir:
            raise ExecutorError("record_video_dir must be set for phased execution")

        self.config.record_video_dir.mkdir(parents=True, exist_ok=True)
        state_file = self.config.record_video_dir / "session_state.json"

        setup_steps = test.get_setup_steps()
        recorded_steps = test.get_recorded_steps()
        teardown_steps = test.get_teardown_steps()

        markers: list[EventMarker] = []
        video_path: Path | None = None
        current_url: str | None = None
        setup_successful = True
        teardown_successful = True

        async with async_playwright() as p:
            browser_launcher = getattr(p, self.config.browser_type)
            browser = await browser_launcher.launch(
                headless=self.config.headless,
                slow_mo=self.config.slow_mo,
            )

            try:
                # PHASE 1: Setup (no recording)
                if setup_steps:
                    try:
                        current_url = await self._execute_setup_phase(
                            browser, setup_steps, state_file, on_step
                        )
                    except Exception as e:
                        setup_successful = False
                        raise ExecutorError(f"Setup phase failed: {e}")

                # PHASE 2: Recording
                if recorded_steps:
                    markers, video_path, current_url = await self._execute_recording_phase(
                        browser,
                        test,
                        recorded_steps,
                        state_file if setup_steps else None,
                        current_url,
                        on_step,
                        step_offset=len(setup_steps),
                    )

                # PHASE 3: Teardown (no recording)
                if teardown_steps:
                    try:
                        await self._execute_teardown_phase(
                            browser,
                            teardown_steps,
                            state_file,
                            current_url,
                            on_step,
                            step_offset=len(setup_steps) + len(recorded_steps),
                        )
                    except Exception:
                        teardown_successful = False
                        # Don't raise - teardown failure shouldn't fail the whole process

            finally:
                await browser.close()
                # Cleanup state file
                if state_file.exists():
                    state_file.unlink()

        return PhasedExecutionResult(
            markers=markers,
            video_path=video_path,
            setup_successful=setup_successful,
            teardown_successful=teardown_successful,
        )

    async def _execute_setup_phase(
        self,
        browser: Browser,
        steps: list[TestStep],
        state_file: Path,
        on_step: Callable[[int, TestStep], None] | None,
    ) -> str | None:
        """Execute setup steps without recording, save state for next phase.

        Returns the current URL after setup completes.
        """
        context_kwargs: dict[str, object] = {
            "viewport": {
                "width": self.config.viewport_width,
                "height": self.config.viewport_height,
            }
        }
        if self.config.device_scale_factor is not None:
            context_kwargs["device_scale_factor"] = self.config.device_scale_factor

        # NO record_video_dir for setup phase
        context = await browser.new_context(**context_kwargs)
        page = await context.new_page()
        self._page = page

        try:
            for i, step in enumerate(steps):
                if on_step:
                    on_step(i, step)
                await self._execute_step(step)
                await asyncio.sleep(0.3)  # Small delay for stability

            # Save state for recording phase
            current_url = page.url
            await context.storage_state(path=str(state_file))
            return current_url
        finally:
            await page.close()
            await context.close()

    async def _execute_recording_phase(
        self,
        browser: Browser,
        test: ParsedTest,
        steps: list[TestStep],
        state_file: Path | None,
        initial_url: str | None,
        on_step: Callable[[int, TestStep], None] | None,
        step_offset: int = 0,
    ) -> tuple[list[EventMarker], Path | None, str | None]:
        """Execute main recording phase with video recording enabled.

        Returns (markers, video_path, current_url).
        """
        context_kwargs: dict[str, object] = {
            "viewport": {
                "width": self.config.viewport_width,
                "height": self.config.viewport_height,
            },
            "record_video_dir": str(self.config.record_video_dir),
        }
        if self.config.device_scale_factor is not None:
            context_kwargs["device_scale_factor"] = self.config.device_scale_factor
        if self.config.record_video_size:
            w, h = self.config.record_video_size
            context_kwargs["record_video_size"] = {"width": w, "height": h}

        # Restore state from setup phase
        if state_file and state_file.exists():
            context_kwargs["storage_state"] = str(state_file)

        context = await browser.new_context(**context_kwargs)

        # Inject mouse helper script for visible cursor in recordings
        if self.config.show_cursor:
            await context.add_init_script(MOUSE_HELPER_JS)

        page = await context.new_page()
        self._page = page

        # Navigate to last known URL if we have state
        if initial_url:
            try:
                await page.goto(initial_url, wait_until="domcontentloaded")
            except Exception:
                pass  # Continue even if navigation fails

        markers: list[EventMarker] = []
        video_path: Path | None = None
        start_time = time.time()
        current_url: str | None = None

        try:
            for i, step in enumerate(steps):
                # Get original index in the full test for marker synchronization
                original_index = test.get_step_original_index(step)

                if on_step:
                    on_step(original_index, step)

                # For steps marked skip in the middle, execute with minimal delay, no markers
                if step.skip_recording:
                    await self._execute_step(step)
                    await asyncio.sleep(0.3)
                    continue

                # Mark step start
                step_start_time = time.time() - start_time
                markers.append(EventMarker(
                    timestamp=step_start_time,
                    event_type="step_start",
                    metadata={
                        "index": original_index,
                        "action": step.action.value,
                        "target": step.target,
                        "description": step.description,
                    },
                ))

                # Get step delay
                delay = self._get_step_delay(original_index)

                # Anticipatory mode: wait for narration BEFORE action
                if self.config.anticipatory_mode:
                    if delay > 0:
                        await asyncio.sleep(delay)
                    click_coords = await self._execute_step(step)
                    await asyncio.sleep(0.5)  # Brief hold after action
                else:
                    # Standard mode: execute action, then wait
                    click_coords = await self._execute_step(step)
                    if delay > 0:
                        await asyncio.sleep(delay)
                    await asyncio.sleep(0.5)  # Safety buffer

                # Update the last step_start marker with click coordinates
                if click_coords and markers:
                    for m in reversed(markers):
                        if m.event_type == "step_start" and m.metadata.get("index") == original_index:
                            m.metadata["x"] = click_coords[0]
                            m.metadata["y"] = click_coords[1]
                            break

                # Mark step end
                step_end_time = time.time() - start_time
                markers.append(EventMarker(
                    timestamp=step_end_time,
                    event_type="step_end",
                    metadata={"index": original_index},
                ))

            current_url = page.url
        finally:
            try:
                if page.video:
                    video_path = Path(await page.video.path())
                await page.close()
            except Exception:
                pass
            await context.close()

        return markers, video_path, current_url

    async def _execute_teardown_phase(
        self,
        browser: Browser,
        steps: list[TestStep],
        state_file: Path,
        initial_url: str | None,
        on_step: Callable[[int, TestStep], None] | None,
        step_offset: int = 0,
    ) -> None:
        """Execute teardown steps without recording."""
        context_kwargs: dict[str, object] = {
            "viewport": {
                "width": self.config.viewport_width,
                "height": self.config.viewport_height,
            }
        }
        if self.config.device_scale_factor is not None:
            context_kwargs["device_scale_factor"] = self.config.device_scale_factor

        # Restore state from recording phase
        if state_file.exists():
            context_kwargs["storage_state"] = str(state_file)

        context = await browser.new_context(**context_kwargs)
        page = await context.new_page()
        self._page = page

        if initial_url:
            try:
                await page.goto(initial_url, wait_until="domcontentloaded")
            except Exception:
                pass

        try:
            for i, step in enumerate(steps):
                if on_step:
                    on_step(step_offset + i, step)
                await self._execute_step(step)
                await asyncio.sleep(0.3)
        finally:
            await page.close()
            await context.close()

    async def _execute_step(self, step: TestStep) -> tuple[int, int] | None:
        """Execute a single test step.

        Returns:
            For click/hover actions, returns (x, y) center coordinates of the clicked element.
            For other actions, returns None.
        """
        if self._page is None:
            raise ExecutorError("No page available")

        page = self._page
        click_coords: tuple[int, int] | None = None

        match step.action:
            case ActionType.NAVIGATE:
                await page.goto(step.target)

            case ActionType.CLICK:
                locator = self._get_locator(page, step.target)
                # Capture element center coordinates before clicking
                try:
                    box = await locator.bounding_box()
                    if box:
                        x = int(box["x"] + box["width"] / 2)
                        y = int(box["y"] + box["height"] / 2)
                        click_coords = (x, y)
                except Exception:
                    pass  # Continue without coordinates if bounding box fails
                await locator.click()

            case ActionType.TYPE:
                locator = self._get_locator(page, step.target)
                if step.value:
                    await locator.fill(step.value)
                else:
                    await locator.fill("")

            case ActionType.PRESS:
                locator = self._get_locator(page, step.target)
                if step.value:
                    await locator.press(step.value)

            case ActionType.HOVER:
                locator = self._get_locator(page, step.target)
                # Capture element center coordinates before hovering
                try:
                    box = await locator.bounding_box()
                    if box:
                        x = int(box["x"] + box["width"] / 2)
                        y = int(box["y"] + box["height"] / 2)
                        click_coords = (x, y)
                except Exception:
                    pass
                await locator.hover()

            case ActionType.SELECT:
                locator = self._get_locator(page, step.target)
                if step.value:
                    if step.value.startswith("label="):
                        await locator.select_option(label=step.value[len("label=") :])
                    elif step.value.startswith("value="):
                        await locator.select_option(value=step.value[len("value=") :])
                    else:
                        await locator.select_option(step.value)

            case ActionType.SCROLL:
                locator = self._get_locator(page, step.target)
                await locator.scroll_into_view_if_needed()

            case ActionType.WAIT:
                await self._execute_wait(step)

            case ActionType.ASSERT:
                await self._execute_assertion(step)

            case ActionType.SCREENSHOT:
                # Screenshot doesn't need special handling for recording
                pass

            case _:
                # Unknown action - skip
                pass

        return click_coords

    def _get_locator(self, page: Page, target: str) -> Any:
        """Get a Playwright locator from a target string."""
        chain_prefixes = (
            "locator(",
            "get_by_role(",
            "get_by_text(",
            "get_by_label(",
            "get_by_placeholder(",
            "get_by_test_id(",
            "get_by_alt_text(",
        )

        # Check for chain patterns that include .first, .last, .nth(x) and locator/get_by_* calls.
        if target.startswith(chain_prefixes) or any(
            marker in target for marker in [").locator(", ").get_by_", ".first", ".last", ".nth("]
        ):
            return self._eval_locator_chain(page, target)

        # Handle explicit xpath selector
        if target.startswith("xpath="):
            return page.locator(target)

        # Default: use as CSS selector
        return page.locator(target)

    def _eval_locator_chain(self, page: Page, expr: str) -> Any:
        """Evaluate a chained locator expression.

        Handles chains like: "get_by_text('X').locator('..')"
        And now supports: .first, .last, .nth(x), and multiple arguments
        """
        result: Any = page
        remainder = expr

        while remainder:
            if remainder.startswith(".first"):
                result = result.first
                remainder = remainder[len(".first") :]
                continue

            if remainder.startswith(".last"):
                result = result.last
                remainder = remainder[len(".last") :]
                continue

            consumed = self._consume_call(remainder)
            if consumed is None:
                # Ignore stray dots, then stop.
                if remainder.startswith("."):
                    remainder = remainder[1:]
                    continue
                break

            method, args_str, remainder = consumed

            try:
                args, kwargs = self._parse_args(args_str)
                if method == "locator":
                    result = result.locator(*args, **kwargs)
                elif method == "get_by_role":
                    result = result.get_by_role(*args, **kwargs)
                elif method == "get_by_text":
                    result = result.get_by_text(*args, **kwargs)
                elif method == "get_by_label":
                    result = result.get_by_label(*args, **kwargs)
                elif method == "get_by_placeholder":
                    result = result.get_by_placeholder(*args, **kwargs)
                elif method == "get_by_test_id":
                    result = result.get_by_test_id(*args, **kwargs)
                elif method == "get_by_alt_text":
                    result = result.get_by_alt_text(*args, **kwargs)
                elif method == "nth":
                    result = result.nth(*args, **kwargs)
                else:
                    # Unknown method in locator chain.
                    break
            except Exception:
                break

        return result

    def _consume_call(self, expr: str) -> tuple[str, str, str] | None:
        s = expr
        if s.startswith("."):
            s = s[1:]

        match = re.match(r"^(\w+)\(", s)
        if match is None:
            return None

        method = match.group(1)
        open_paren_at = len(method)
        # s[open_paren_at] is '(' per regex.
        i = open_paren_at + 1
        depth = 1
        quote: str | None = None
        escape = False

        while i < len(s):
            ch = s[i]

            if quote is not None:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    quote = None
                i += 1
                continue

            if ch in ("'", '"'):
                quote = ch
                i += 1
                continue

            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    args_str = s[open_paren_at + 1 : i]
                    remainder = s[i + 1 :]
                    return method, args_str, remainder

            i += 1

        return None

    def _parse_args(self, args_str: str) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Parse argument string into args and kwargs."""

        def _capture(*args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
            return args, kwargs

        return eval(
            f"_capture({args_str})",
            {"_capture": _capture, "__builtins__": {}},
        )

    async def _execute_wait(self, step: TestStep) -> None:
        """Execute a wait step."""
        if self._page is None:
            return

        source = step.source_code.lower()

        if "wait_for_load_state" in source:
            # Extract load state if specified
            if "networkidle" in source:
                await self._page.wait_for_load_state("networkidle")
            elif "domcontentloaded" in source:
                await self._page.wait_for_load_state("domcontentloaded")
            else:
                await self._page.wait_for_load_state("load")
        elif "wait_for_url" in source:
            # URL is in the target
            if step.target:
                await self._page.wait_for_url(step.target)
        elif "wait_for_selector" in source or step.target:
            await self._page.wait_for_selector(step.target)

    async def _execute_assertion(self, step: TestStep) -> None:
        """Execute an assertion step."""
        if self._page is None:
            return

        source = step.source_code.lower()

        # Get the locator if we have a target
        locator = None
        if step.target and not step.target.startswith("expect(page)"):
            locator = self._get_locator(self._page, step.target)

        # Determine assertion type from source code
        if "to_be_visible" in source:
            if locator:
                await async_expect(locator).to_be_visible()
        elif "to_be_hidden" in source:
            if locator:
                await async_expect(locator).to_be_hidden()
        elif "to_contain_text" in source and step.value:
            if locator:
                await async_expect(locator).to_contain_text(step.value)
        elif "to_have_text" in source and step.value:
            if locator:
                await async_expect(locator).to_have_text(step.value)
        elif "to_have_url" in source:
            # URL pattern is usually in target or value
            pattern = step.target or step.value
            if pattern:
                await async_expect(self._page).to_have_url(re.compile(pattern.replace("**", ".*")))
        elif "to_have_title" in source:
            if step.value:
                await async_expect(self._page).to_have_title(step.value)


class ExecutorError(Exception):
    """Raised when test execution fails."""

    pass
