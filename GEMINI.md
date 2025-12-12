# GEMINI.md - Codevid Project

This document provides a comprehensive overview of the Codevid project, intended to be used as a quick reference for developers and AI assistants.

## Project Overview

Codevid is a Python-based command-line tool that generates video tutorials from automated tests. It uses Large Language Models (LLMs) to generate narration scripts from test code and then combines screen recordings, text-to-speech (TTS), and other assets to create a complete video tutorial.

**Core Technologies:**

*   **CLI:** Typer
*   **Configuration:** Pydantic, YAML
*   **LLM Integration:** Anthropic, OpenAI
*   **Test Framework Parsing:** Playwright is the primary supported framework.
*   **Video Generation:** MoviePy
*   **Screen Recording:** Playwright's built-in recording capabilities.
*   **Audio Generation:** edge-tts, openai-tts
*   **Development:** Pytest for testing, Ruff for linting, MyPy for type checking.

## Getting Started

### Installation

The project is packaged using Hatchling and can be installed via pip.

```bash
pip install -e .[dev]
```

This installs the project in editable mode with all development dependencies.

### Running the Application

The main entry point is the `codevid` command-line tool.

**Generate a tutorial:**

```bash
codevid generate <path/to/test.py> -o <output_video.mp4>
```

**Preview a script (no video rendering):**

```bash
codevid preview <path/to/test.py>
```

**Initialize a project:**

This creates a `codevid.yaml` configuration file in the current directory.

```bash
codevid init
```

### Configuration

The project uses a `codevid.yaml` file for configuration. This file allows you to specify the LLM provider, TTS provider and voice, video settings, and more. The `codevid init` command generates a default configuration file.

## Development

### Testing

The project uses `pytest` for testing. Tests are located in the `tests/` directory.

To run the tests:

```bash
pytest
```

### Linting and Formatting

The project uses `ruff` for linting and formatting.

To check for linting errors:

```bash
ruff check .
```

To format the code:

```bash
ruff format .
```

### Type Checking

The project uses `mypy` for static type checking.

To run the type checker:

```bash
mypy src/
```

## Project Structure

*   `src/codevid/`: Main source code for the application.
    *   `cli.py`: The Typer-based command-line interface.
    *   `pipeline.py`: The main orchestration logic for video generation.
    *   `config.py`: Pydantic models for configuration.
    *   `llm/`: Modules for interacting with LLM providers.
    *   `tts/`: Modules for text-to-speech generation.
    *   `parsers/`: Modules for parsing test files (e.g., Playwright).
    *   `recorder/`: Screen recording logic.
    *   `composer/`: Video composition logic using MoviePy.
*   `tests/`: Pytest tests.
*   `examples/`: Example test files and a demo Flask application.
*   `pyproject.toml`: Project metadata, dependencies, and tool configurations (pytest, ruff, mypy).
*   `codevid.yaml`: Project configuration file.

## Architectural Flow

The video generation process is orchestrated by the `Pipeline` class in `src/codevid/pipeline.py` and follows these steps:

1.  **Parse Test:** The input test file is parsed to extract steps and actions.
2.  **Generate Script:** The parsed test steps are sent to an LLM to generate a narration script.
3.  **Generate Audio:** The narration script is converted to audio using a TTS provider.
4.  **Record Test:** The test is executed by Playwright, which records the browser interaction to a video file.
5.  **Compose Video:** The recorded video, audio narration, captions, and other assets (like intros/outros and watermarks) are combined into the final video file using MoviePy.
