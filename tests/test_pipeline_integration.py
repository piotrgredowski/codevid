"""Integration tests for the full pipeline in preview mode.

These tests demonstrate that the complete flow works:
1. Parse a Playwright test file
2. Generate a narration script via LLM
3. Return a valid VideoScript with all components

Run with a real LLM provider:
    ANTHROPIC_API_KEY=xxx pytest tests/test_pipeline_integration.py -v -k "real_llm"

Run with mock provider (no API key needed):
    pytest tests/test_pipeline_integration.py -v -k "not real_llm"
"""

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from codevid.llm.base import LLMProvider
from codevid.models import (
    ActionType,
    NarrationSegment,
    ParsedTest,
    TestStep,
    VideoScript,
)
from codevid.models.project import LLMConfig, LLMProviderType, ProjectConfig
from codevid.parsers import PlaywrightParser
from codevid.pipeline import Pipeline, PipelineConfig


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing without API keys."""

    def __init__(self) -> None:
        self._call_count = 0

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-model"

    async def generate_script(
        self,
        test: ParsedTest,
        context: dict[str, Any] | None = None,
    ) -> VideoScript:
        """Generate a mock script based on the parsed test."""
        self._call_count += 1
        context = context or {}
        app_name = context.get("app_name", "the application")

        segments = []
        for i, step in enumerate(test.steps):
            segments.append(
                NarrationSegment(
                    text=f"Now we {step.description.lower()}",
                    step_index=i,
                    timing_hint=3.0,
                    emphasis_words=[step.action.value],
                )
            )

        return VideoScript(
            title=f"Tutorial: {test.name}",
            introduction=f"Welcome! In this video, we'll demonstrate how to use {app_name}. "
            f"We'll walk through the {test.name.replace('_', ' ')} process step by step.",
            segments=segments,
            conclusion=f"That's it! You've successfully completed the {test.name.replace('_', ' ')}. "
            "Thanks for watching!",
        )

    async def enhance_description(
        self,
        step: TestStep,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Generate a mock enhanced description."""
        return f"Enhanced: {step.description}"


@pytest.fixture
def sample_playwright_test() -> Path:
    """Create a realistic Playwright test file."""
    content = '''"""Test user registration flow."""

from playwright.sync_api import Page, expect


def test_user_registration(page: Page) -> None:
    """Test the complete user registration flow for new users."""
    # Navigate to the registration page
    page.goto("https://myapp.com/register")

    # Wait for page to load
    page.wait_for_load_state("networkidle")

    # Fill in the registration form
    page.fill("#first-name", "John")
    page.fill("#last-name", "Doe")
    page.fill("#email", "john.doe@example.com")
    page.fill("#password", "SecurePass123!")
    page.fill("#confirm-password", "SecurePass123!")

    # Accept terms and conditions
    page.locator("#terms-checkbox").check()

    # Click register button
    page.click("button[type='submit']")

    # Wait for redirect to welcome page
    page.wait_for_url("**/welcome")

    # Verify registration was successful
    expect(page.locator(".welcome-message")).to_be_visible()
    expect(page.locator(".welcome-message")).to_contain_text("Welcome, John!")
'''
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(content)
        return Path(f.name)


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    """Create a mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def project_config(tmp_path: Path) -> ProjectConfig:
    """Create a test project configuration."""
    return ProjectConfig(
        name="Test Project",
        output_dir=tmp_path / "output",
    )


class TestPipelinePreviewMode:
    """Integration tests for the pipeline in preview mode."""

    def test_full_preview_pipeline_with_mock_llm(
        self,
        sample_playwright_test: Path,
        mock_llm: MockLLMProvider,
        project_config: ProjectConfig,
        tmp_path: Path,
    ) -> None:
        """Test complete preview pipeline: parse -> generate script."""
        # Arrange
        parser = PlaywrightParser()
        pipeline_config = PipelineConfig(
            test_file=sample_playwright_test,
            output=tmp_path / "output.mp4",
            project_config=project_config,
            app_name="MyApp",
            preview_mode=True,
        )

        pipeline = Pipeline(
            config=pipeline_config,
            parser=parser,
            llm=mock_llm,
        )

        # Track progress updates
        progress_updates: list[tuple[int, str]] = []
        pipeline.on_progress(lambda p, m: progress_updates.append((p, m)))

        # Act
        result = pipeline.run()

        # Assert - Pipeline succeeded
        assert result.success is True
        assert result.error is None
        assert result.output_path is None  # Preview mode doesn't produce video

        # Assert - Script was generated correctly
        script = result.script
        assert script.title == "Tutorial: test_user_registration"
        assert "MyApp" in script.introduction
        assert "registration" in script.introduction.lower()
        assert len(script.segments) > 0
        assert "watching" in script.conclusion.lower()

        # Assert - Segments match parsed steps
        assert len(script.segments) >= 5  # Multiple fill operations + click + checks

        # Assert - Progress was reported
        assert len(progress_updates) >= 3
        assert progress_updates[0][0] == 5  # Parsing started
        assert progress_updates[-1][0] == 25  # Script generated (preview stops here)

        # Assert - LLM was called
        assert mock_llm._call_count == 1

    def test_parsed_test_contains_all_actions(
        self, sample_playwright_test: Path
    ) -> None:
        """Verify the parser extracts all expected actions from the test."""
        parser = PlaywrightParser()
        result = parser.parse(sample_playwright_test)

        # Check test metadata
        assert result.name == "test_user_registration"
        assert "registration" in result.metadata.get("docstring", "").lower()
        assert result.metadata.get("is_async") is False

        # Check all action types are present
        action_types = {step.action for step in result.steps}
        assert ActionType.NAVIGATE in action_types
        assert ActionType.TYPE in action_types
        assert ActionType.CLICK in action_types
        assert ActionType.WAIT in action_types
        assert ActionType.ASSERT in action_types

        # Check specific steps
        navigate_steps = [s for s in result.steps if s.action == ActionType.NAVIGATE]
        assert any("register" in s.target for s in navigate_steps)

        fill_steps = [s for s in result.steps if s.action == ActionType.TYPE]
        assert len(fill_steps) >= 5  # 5 form fields
        assert any(s.value == "john.doe@example.com" for s in fill_steps)

    def test_video_script_structure(
        self,
        sample_playwright_test: Path,
        mock_llm: MockLLMProvider,
        project_config: ProjectConfig,
        tmp_path: Path,
    ) -> None:
        """Verify the generated VideoScript has correct structure."""
        parser = PlaywrightParser()
        pipeline_config = PipelineConfig(
            test_file=sample_playwright_test,
            output=tmp_path / "output.mp4",
            project_config=project_config,
            preview_mode=True,
        )

        pipeline = Pipeline(
            config=pipeline_config,
            parser=parser,
            llm=mock_llm,
        )

        result = pipeline.run()
        script = result.script

        # Check VideoScript structure
        assert isinstance(script.title, str)
        assert len(script.title) > 0

        assert isinstance(script.introduction, str)
        assert len(script.introduction) > 0

        assert isinstance(script.segments, list)
        for segment in script.segments:
            assert isinstance(segment, NarrationSegment)
            assert isinstance(segment.text, str)
            assert len(segment.text) > 0
            assert isinstance(segment.step_index, int)
            assert segment.timing_hint > 0

        assert isinstance(script.conclusion, str)
        assert len(script.conclusion) > 0

        # Check calculated duration
        assert script.total_estimated_duration > 0
        assert script.segment_count == len(script.segments)

    def test_pipeline_handles_parser_error(
        self,
        mock_llm: MockLLMProvider,
        project_config: ProjectConfig,
        tmp_path: Path,
    ) -> None:
        """Test pipeline handles non-existent test file gracefully."""
        parser = PlaywrightParser()
        pipeline_config = PipelineConfig(
            test_file=Path("/nonexistent/test.py"),
            output=tmp_path / "output.mp4",
            project_config=project_config,
            preview_mode=True,
        )

        pipeline = Pipeline(
            config=pipeline_config,
            parser=parser,
            llm=mock_llm,
        )

        result = pipeline.run()

        assert result.success is False
        assert result.error is not None
        assert mock_llm._call_count == 0  # LLM should not be called

    def test_pipeline_without_llm_raises_error(
        self,
        sample_playwright_test: Path,
        project_config: ProjectConfig,
        tmp_path: Path,
    ) -> None:
        """Test pipeline requires LLM provider."""
        parser = PlaywrightParser()
        pipeline_config = PipelineConfig(
            test_file=sample_playwright_test,
            output=tmp_path / "output.mp4",
            project_config=project_config,
            preview_mode=True,
        )

        pipeline = Pipeline(
            config=pipeline_config,
            parser=parser,
            llm=None,  # No LLM provider
        )

        result = pipeline.run()

        assert result.success is False
        assert "No LLM provider" in str(result.error)


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
class TestPipelineWithRealLLM:
    """Integration tests using real LLM provider.

    These tests require ANTHROPIC_API_KEY to be set.
    Run with: ANTHROPIC_API_KEY=xxx pytest -v -k "real_llm"
    """

    def test_real_llm_preview_pipeline(
        self,
        sample_playwright_test: Path,
        project_config: ProjectConfig,
        tmp_path: Path,
    ) -> None:
        """Test preview pipeline with real Anthropic LLM."""
        from codevid.llm import create_llm_provider
        from codevid.models.project import LLMConfig, LLMProviderType

        # Create real LLM provider
        llm_config = LLMConfig(provider=LLMProviderType.ANTHROPIC)
        llm = create_llm_provider(llm_config)

        parser = PlaywrightParser()
        pipeline_config = PipelineConfig(
            test_file=sample_playwright_test,
            output=tmp_path / "output.mp4",
            project_config=project_config,
            app_name="MyApp Registration System",
            preview_mode=True,
        )

        pipeline = Pipeline(
            config=pipeline_config,
            parser=parser,
            llm=llm,
        )

        result = pipeline.run()
       
        # Assert success
        assert result.success is True, f"Pipeline failed: {result.error}"

        # Assert script quality
        script = result.script
        assert len(script.title) > 0
        assert len(script.introduction) > 20  # Real intro should be substantial
        assert len(script.segments) > 0
        assert len(script.conclusion) > 20

        # Real LLM should produce natural language
        full_text = script.get_full_text()
        assert len(full_text) > 100

        print("\n--- Generated Script ---")
        print(f"Title: {script.title}")
        print(f"\nIntroduction:\n{script.introduction}")
        print(f"\nSegments ({len(script.segments)}):")
        for i, seg in enumerate(script.segments):
            print(f"  {i + 1}. {seg.text[:80]}...")
        print(f"\nConclusion:\n{script.conclusion}")
        print(f"\nEstimated duration: {script.total_estimated_duration:.1f}s")


class TestParserLLMIntegration:
    """Test that parser output integrates correctly with LLM."""

    def test_parsed_steps_have_descriptions(
        self, sample_playwright_test: Path
    ) -> None:
        """Verify parser generates descriptions suitable for LLM."""
        parser = PlaywrightParser()
        result = parser.parse(sample_playwright_test)

        for step in result.steps:
            # Every step should have a description
            assert step.description is not None
            assert len(step.description) > 0

            # Description should be human-readable
            assert step.action.value.lower() in step.description.lower() or any(
                word in step.description.lower()
                for word in ["navigate", "type", "click", "verify", "wait"]
            )

    def test_all_step_fields_populated(
        self, sample_playwright_test: Path
    ) -> None:
        """Verify all TestStep fields are properly populated."""
        parser = PlaywrightParser()
        result = parser.parse(sample_playwright_test)

        for step in result.steps:
            assert step.action in ActionType
            assert step.target is not None
            assert step.description is not None
            assert step.line_number > 0
            assert step.source_code is not None
