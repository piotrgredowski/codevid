# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Codevid generates video tutorials from automated tests using LLMs. The tool parses test files (currently Playwright), generates narration scripts via LLM, synthesizes audio with TTS, records test execution, and composes the final video with captions and overlays.

## Development Commands

### Installation & Setup
```bash
# Install with dev dependencies
pip install -e .[dev]

# Install Playwright browsers
python -m playwright install chromium
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_playwright_parser.py

# Run Playwright integration tests (requires demo app)
pytest examples/demo_app/test_demo_login.py --browser chromium
```

### Code Quality
```bash
# Lint code
ruff check .

# Type check
mypy .
```

### Using Codevid CLI
```bash
# Generate video from test
codevid generate tests/test_login.py -o tutorial.mp4

# Preview script only (no recording)
codevid preview tests/test_login.py

# Initialize config file
codevid init
```

## Architecture

### Pipeline Flow (src/codevid/pipeline.py)
The core `Pipeline` class orchestrates the entire video generation process:
1. **Parse** test file → `ParsedTest` (via `TestParser`)
2. **Generate** narration script → `VideoScript` (via `LLMProvider`)
3. **Synthesize** audio → audio segments (via `TTSProvider`)
4. **Record** test execution → video file + event markers (via `PlaywrightExecutor`)
5. **Compose** final video with narration, captions, overlays (via `VideoComposer`)

### Key Abstractions

**Parser Layer** (`src/codevid/parsers/`)
- `TestParser` (base): Abstract interface for test file parsing
- `PlaywrightParser`: AST-based parser that converts Playwright test code into `TestStep` objects
- Uses `ACTION_MAP` to translate Playwright methods (click, fill, goto) to `ActionType` enum
- Parser registry pattern allows extending to other frameworks

**LLM Layer** (`src/codevid/llm/`)
- `LLMProvider` (base): Abstract interface for script generation
- Implementations: `AnthropicProvider`, `OpenAIProvider`, `OllamaProvider`
- `SimpleLLM`: Rule-based fallback when LLM unavailable
- Factory pattern (`llm/factory.py`) creates providers from config
- Prompts system (`llm/prompts.py`) manages prompt templates

**Audio Layer** (`src/codevid/audio/`)
- `TTSProvider` (base): Abstract interface for text-to-speech
- Implementations: `EdgeTTSProvider`, `OpenAITTSProvider`
- Factory pattern creates providers from config
- Audio segments stored as temporary files

**Executor Layer** (`src/codevid/executor/`)
- `PlaywrightExecutor`: Executes `ParsedTest` steps using Playwright async API
- Records video via Playwright's built-in `record_video_dir` context option
- Returns `EventMarker` list for syncing narration with actions
- Configuration controls headless mode, slow_mo, viewport size, delays

**Composer Layer** (`src/codevid/composer/`)
- `VideoComposer`: Assembles final video using moviepy
- `captions.py`: SRT subtitle generation
- `overlays.py`: Click highlights, mouse spotlight effects
- `templates.py`: Intro/outro template handling

### Data Models (`src/codevid/models/`)
- `TestStep`: Single action (ActionType, target, value, description)
- `ParsedTest`: Collection of steps + metadata
- `VideoScript`: Title, intro, segments (text + step indices), conclusion
- `ProjectConfig`: All configuration (LLM, TTS, recording, video settings)

### Configuration (`src/codevid/config.py`)
- Loads from `codevid.yaml` (or `.yml`, `.codevid.yaml`, `.codevid.yml`)
- Searches current dir → parent dirs
- Merges YAML config with environment variables (`CODEVID_*` prefix)
- API keys sourced from env vars or explicit config

## Testing Strategy

- **Parser tests** (`test_playwright_parser.py`): AST parsing accuracy, action mapping
- **LLM tests** (`test_llm_providers.py`): Provider initialization, script generation (mocked)
- **TTS tests** (`test_tts_providers.py`): Provider initialization, audio synthesis (mocked)
- **Integration tests** (`test_pipeline_integration.py`): Full pipeline with mocked providers
- **Composer tests** (`test_composer.py`): Video composition, captions, overlays

Tests use pytest + pytest-asyncio. Playwright tests require demo Flask app (`examples/demo_app/`) running on expected port.

## Important Implementation Details

### Playwright AST Parsing
`PlaywrightParser` walks Python AST to find:
- Test functions (prefix `test_`)
- Playwright method calls (page.goto, page.click, locator().fill)
- Assertion chains (expect().to_be_visible)
- Extracts selector strings, URLs, input values from call arguments

### Pipeline Progress Reporting
Pipeline emits progress callbacks (0-100%) for CLI progress bars. Key stages:
- 5-10%: Parsing
- 15-25%: Script generation
- 30-45%: Audio synthesis
- 50-75%: Test recording
- 80-100%: Video composition

### Video Recording Flow
1. Playwright launches with `record_video_dir` enabled
2. Test executes step by step with delays
3. Video saved to temp dir (`.codevid_temp/`)
4. Composer reads raw video, overlays narration audio synced to event markers
5. Final video written to output path

### Error Handling
- `LLMError`: Raised when LLM provider fails (API key missing, model not found)
- `TTSError`: Raised when TTS provider fails (hard failure to avoid silent videos)
- `ParseError`: Raised when test parsing fails
- `PipelineError`: Raised for pipeline-level failures
- CLI falls back to `SimpleLLM` if LLM unavailable

## Code Style

- Python 3.11+ with strict type hints (mypy strict mode)
- Ruff formatting, line length 100
- `snake_case` for modules, functions, variables
- Tests named `test_*.py`, functions `test_*`
- Prefer small, pure functions
- Use Pydantic for settings and config models
