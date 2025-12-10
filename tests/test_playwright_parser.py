"""Tests for the Playwright parser."""

import tempfile
from pathlib import Path

import pytest

from codevid.models import ActionType
from codevid.parsers import PlaywrightParser, get_parser, parse_test
from codevid.parsers.base import ParseError


@pytest.fixture
def parser() -> PlaywrightParser:
    return PlaywrightParser()


@pytest.fixture
def sample_test_file() -> Path:
    """Create a temporary test file."""
    content = '''
from playwright.sync_api import Page, expect

def test_example(page: Page):
    """Test example flow."""
    page.goto("https://example.com")
    page.fill("#username", "testuser")
    page.click("button[type='submit']")
    expect(page.locator(".welcome")).to_be_visible()
'''
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(content)
        return Path(f.name)


class TestPlaywrightParser:
    def test_can_parse_playwright_file(self, parser: PlaywrightParser, sample_test_file: Path):
        assert parser.can_parse(sample_test_file)

    def test_cannot_parse_non_python_file(self, parser: PlaywrightParser, tmp_path: Path):
        js_file = tmp_path / "test.js"
        js_file.write_text("console.log('hello')")
        assert not parser.can_parse(js_file)

    def test_cannot_parse_non_playwright_python(self, parser: PlaywrightParser, tmp_path: Path):
        py_file = tmp_path / "regular.py"
        py_file.write_text("def hello():\n    print('hello')")
        assert not parser.can_parse(py_file)

    def test_parse_basic_test(self, parser: PlaywrightParser, sample_test_file: Path):
        result = parser.parse(sample_test_file)

        assert result.name == "test_example"
        assert result.file_path == str(sample_test_file)
        assert len(result.steps) >= 3

    def test_parse_navigation(self, parser: PlaywrightParser, tmp_path: Path):
        test_file = tmp_path / "test_nav.py"
        test_file.write_text('''
from playwright.sync_api import Page

def test_nav(page: Page):
    page.goto("https://example.com/page")
''')
        result = parser.parse(test_file)

        assert len(result.steps) == 1
        step = result.steps[0]
        assert step.action == ActionType.NAVIGATE
        assert step.target == "https://example.com/page"

    def test_parse_click(self, parser: PlaywrightParser, tmp_path: Path):
        test_file = tmp_path / "test_click.py"
        test_file.write_text('''
from playwright.sync_api import Page

def test_click(page: Page):
    page.click("#submit-button")
''')
        result = parser.parse(test_file)

        assert len(result.steps) == 1
        step = result.steps[0]
        assert step.action == ActionType.CLICK
        assert step.target == "#submit-button"

    def test_parse_fill(self, parser: PlaywrightParser, tmp_path: Path):
        test_file = tmp_path / "test_fill.py"
        test_file.write_text('''
from playwright.sync_api import Page

def test_fill(page: Page):
    page.fill("#email", "test@example.com")
''')
        result = parser.parse(test_file)

        assert len(result.steps) == 1
        step = result.steps[0]
        assert step.action == ActionType.TYPE
        assert step.target == "#email"
        assert step.value == "test@example.com"

    def test_parse_locator_chain(self, parser: PlaywrightParser, tmp_path: Path):
        test_file = tmp_path / "test_locator.py"
        test_file.write_text('''
from playwright.sync_api import Page

def test_locator(page: Page):
    page.locator("#form").fill("value")
''')
        result = parser.parse(test_file)

        assert len(result.steps) == 1
        step = result.steps[0]
        assert step.action == ActionType.TYPE
        assert "#form" in step.target

    def test_parse_get_by_role(self, parser: PlaywrightParser, tmp_path: Path):
        test_file = tmp_path / "test_role.py"
        test_file.write_text('''
from playwright.sync_api import Page

def test_role(page: Page):
    page.get_by_role("button", name="Submit").click()
''')
        result = parser.parse(test_file)

        assert len(result.steps) == 1
        step = result.steps[0]
        assert step.action == ActionType.CLICK
        assert "get_by_role" in step.target
        assert "button" in step.target

    def test_parse_expect_assertion(self, parser: PlaywrightParser, tmp_path: Path):
        test_file = tmp_path / "test_expect.py"
        test_file.write_text('''
from playwright.sync_api import Page, expect

def test_expect(page: Page):
    expect(page.locator(".message")).to_be_visible()
''')
        result = parser.parse(test_file)

        assert len(result.steps) == 1
        step = result.steps[0]
        assert step.action == ActionType.ASSERT

    def test_parse_async_test(self, parser: PlaywrightParser, tmp_path: Path):
        test_file = tmp_path / "test_async.py"
        test_file.write_text('''
from playwright.async_api import Page

async def test_async_example(page: Page):
    await page.goto("https://example.com")
    await page.click("#button")
''')
        result = parser.parse(test_file)

        assert result.metadata["is_async"] is True
        assert len(result.steps) == 2

    def test_parse_preserves_line_numbers(self, parser: PlaywrightParser, tmp_path: Path):
        test_file = tmp_path / "test_lines.py"
        test_file.write_text('''
from playwright.sync_api import Page

def test_lines(page: Page):
    # Line 5
    page.goto("https://example.com")
    # Line 7
    page.click("#button")
''')
        result = parser.parse(test_file)

        assert len(result.steps) == 2
        assert result.steps[0].line_number == 6
        assert result.steps[1].line_number == 8

    def test_parse_generates_descriptions(self, parser: PlaywrightParser, tmp_path: Path):
        test_file = tmp_path / "test_desc.py"
        test_file.write_text('''
from playwright.sync_api import Page

def test_desc(page: Page):
    page.goto("https://example.com")
    page.fill("#email", "test@test.com")
''')
        result = parser.parse(test_file)

        assert "Navigate" in result.steps[0].description
        assert "Type" in result.steps[1].description

    def test_parse_no_test_functions_raises_error(self, parser: PlaywrightParser, tmp_path: Path):
        test_file = tmp_path / "no_tests.py"
        test_file.write_text('''
from playwright.sync_api import Page

def helper_function():
    pass
''')
        with pytest.raises(ParseError) as exc_info:
            parser.parse(test_file)

        assert "No test functions found" in str(exc_info.value)

    def test_parse_syntax_error_raises_error(self, parser: PlaywrightParser, tmp_path: Path):
        test_file = tmp_path / "syntax_error.py"
        test_file.write_text('''
def test_broken(
    # Missing closing paren
''')
        with pytest.raises(ParseError) as exc_info:
            parser.parse(test_file)

        assert "Syntax error" in str(exc_info.value)


class TestParserRegistry:
    def test_get_parser_for_playwright(self, sample_test_file: Path):
        parser = get_parser(sample_test_file)
        assert isinstance(parser, PlaywrightParser)

    def test_parse_test_convenience_function(self, sample_test_file: Path):
        result = parse_test(sample_test_file)
        assert result.name == "test_example"
        assert len(result.steps) >= 3

    def test_unknown_file_raises_error(self, tmp_path: Path):
        unknown_file = tmp_path / "unknown.xyz"
        unknown_file.write_text("something")

        with pytest.raises(ParseError) as exc_info:
            get_parser(unknown_file)

        assert "No parser found" in str(exc_info.value)
