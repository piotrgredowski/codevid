import hashlib
from pathlib import Path
from typing import Any

from codevid.llm.base import LLMProvider
from codevid.models import (
    ActionType,
    NarrationSegment,
    ParsedTest,
    VideoScript,
)
from codevid.models import (
    TestStep as ModelTestStep,
)
from codevid.models.project import ProjectConfig
from codevid.parsers.base import TestParser
from codevid.pipeline import Pipeline, PipelineConfig


class StubParser(TestParser):
    @property
    def framework_name(self) -> str:
        return "stub"

    def can_parse(self, file_path: str | Path) -> bool:
        return True

    def parse(self, file_path: str | Path) -> ParsedTest:
        steps = [
            ModelTestStep(
                action=ActionType.NAVIGATE,
                target="https://example.com",
                description="Navigate",
            ),
            ModelTestStep(action=ActionType.CLICK, target="#submit", description="Click submit"),
        ]
        return ParsedTest(name="test_stub", file_path=str(file_path), steps=steps)


class CountingLLMProvider(LLMProvider):
    def __init__(self, title: str) -> None:
        self.calls = 0
        self._title = title

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
        self.calls += 1
        return VideoScript(
            title=self._title,
            introduction="Intro",
            segments=[
                NarrationSegment(text="Do step 0", step_index=0, timing_hint=1.0),
                NarrationSegment(text="Do step 1", step_index=1, timing_hint=1.0),
            ],
            conclusion="Done",
        )

    async def enhance_description(
        self,
        step: ModelTestStep,
        context: dict[str, Any] | None = None,
    ) -> str:
        return step.description


class FailingLLMProvider(LLMProvider):
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
        raise AssertionError("LLM should not be called when cache is used")

    async def enhance_description(
        self,
        step: ModelTestStep,
        context: dict[str, Any] | None = None,
    ) -> str:
        raise AssertionError("enhance_description should not be called")


def _cache_path_for(test_file: Path, output_dir: Path) -> Path:
    path_hash = hashlib.sha256(str(test_file.resolve()).encode("utf-8")).hexdigest()
    return output_dir / ".codevid_cache" / f"{path_hash}.json"


def test_use_cached_narration_script_reuses_script_when_test_file_unchanged(
    tmp_path: Path,
) -> None:
    test_file = tmp_path / "test_example.py"
    test_file.write_text("# v1\nprint('hello')\n", encoding="utf-8")
    output = tmp_path / "videos" / "out.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)

    project_config = ProjectConfig(output_dir=tmp_path)
    parser = StubParser()

    llm_v1 = CountingLLMProvider(title="title-v1")
    pipeline_v1 = Pipeline(
        PipelineConfig(
            test_file=test_file,
            output=output,
            project_config=project_config,
            preview_mode=True,
            use_cached_narration_script=True,
        ),
        parser=parser,
        llm=llm_v1,
        tts=None,
    )
    result_v1 = pipeline_v1.run()
    assert result_v1.success is True
    assert llm_v1.calls == 1
    assert result_v1.script.title == "title-v1"
    assert _cache_path_for(test_file, tmp_path).exists()

    pipeline_v1_cached = Pipeline(
        PipelineConfig(
            test_file=test_file,
            output=output,
            project_config=project_config,
            preview_mode=True,
            use_cached_narration_script=True,
        ),
        parser=parser,
        llm=FailingLLMProvider(),
        tts=None,
    )
    result_cached = pipeline_v1_cached.run()
    assert result_cached.success is True
    assert result_cached.script.title == "title-v1"


def test_use_cached_narration_script_invalidates_when_test_file_changes(tmp_path: Path) -> None:
    test_file = tmp_path / "test_example.py"
    test_file.write_text("# v1\nprint('hello')\n", encoding="utf-8")
    output = tmp_path / "videos" / "out.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    project_config = ProjectConfig(output_dir=tmp_path)
    parser = StubParser()

    pipeline_v1 = Pipeline(
        PipelineConfig(
            test_file=test_file,
            output=output,
            project_config=project_config,
            preview_mode=True,
            use_cached_narration_script=True,
        ),
        parser=parser,
        llm=CountingLLMProvider(title="title-v1"),
        tts=None,
    )
    assert pipeline_v1.run().success is True

    test_file.write_text("# v2\nprint('hello world')\n", encoding="utf-8")

    llm_v2 = CountingLLMProvider(title="title-v2")
    pipeline_v2 = Pipeline(
        PipelineConfig(
            test_file=test_file,
            output=output,
            project_config=project_config,
            preview_mode=True,
            use_cached_narration_script=True,
        ),
        parser=parser,
        llm=llm_v2,
        tts=None,
    )
    result_v2 = pipeline_v2.run()
    assert result_v2.success is True
    assert llm_v2.calls == 1
    assert result_v2.script.title == "title-v2"
