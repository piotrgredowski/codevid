Repository Guidelines
=====================

Project Structure & Module Organization
---------------------------------------
- `src/codevid/`: Core library code (CLI, parsers, executor, pipeline).
- `examples/`: Sample Playwright tests and demo Flask app (`examples/demo_app`).
- `tests/`: Pytest suites covering parsers, pipeline integration, and providers.
- `pyproject.toml`: Dependencies, tooling config (pytest, mypy, ruff).

Build, Test, and Development Commands
-------------------------------------
- Install (dev): `pip install -e .[dev]` — editable install with test/lint tooling.
- Playwright browsers: `python -m playwright install chromium` (or `... install`).
- Run all tests: `pytest` (asyncio auto mode enabled).
- Target suite: `pytest tests/test_playwright_parser.py` or `pytest examples/demo_app/test_demo_login.py --browser chromium`.
- Lint: `ruff check .` ; Type-check: `mypy .`.

Coding Style & Naming Conventions
---------------------------------
- Python 3.11+, type hints required (mypy strict). Keep functions small and pure where possible.
- Formatting via ruff defaults; line length 100. Use descriptive names; prefer `snake_case` for modules, functions, and variables.
- Tests mirror source naming; fixtures live near usage.

Testing Guidelines
------------------
- Framework: pytest (+ pytest-asyncio, pytest-playwright). Mark async tests appropriately; rely on built-in `page` fixture for Playwright.
- Name tests `test_*.py` and functions `test_*`. Keep assertions explicit and stable.
- For demo app tests, ensure Flask demo is running on the port the test expects before executing.

Commit & Pull Request Guidelines
--------------------------------
- Write clear, imperative commit messages (e.g., "Add Playwright parser validation").
- PRs should describe intent, key changes, and test coverage; link issues when applicable. Include screenshots for UI/template changes (e.g., `examples/demo_app/templates`).

Security & Configuration Tips
-----------------------------
- Do not commit secrets. Use `.env` or config files kept local.
- Network calls use httpx; handle timeouts and errors explicitly.
- When adding providers (LLM/TTS), validate config via pydantic settings and document any required env vars.
