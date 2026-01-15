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

    def test_parse_get_by_role_with_first(self, parser: PlaywrightParser, tmp_path: Path):
        test_file = tmp_path / "test_role_first.py"
        test_file.write_text('''
from playwright.sync_api import Page

def test_role_first(page: Page):
    page.get_by_role("link", name="Coin Flipper").first.click()
''')
        result = parser.parse(test_file)

        assert len(result.steps) == 1
        step = result.steps[0]
        assert step.action == ActionType.CLICK
        assert step.target.startswith("get_by_role(")
        assert ".first" in step.target
        assert "name=" in step.target

    def test_parse_locator_chain_with_first(self, parser: PlaywrightParser, tmp_path: Path):
        test_file = tmp_path / "test_chain_first.py"
        test_file.write_text('''
from playwright.sync_api import Page

def test_chain_first(page: Page):
    page.locator("#navigation").get_by_role("link", name="Coin Flipper").first.click()
''')
        result = parser.parse(test_file)

        assert len(result.steps) == 1
        step = result.steps[0]
        assert step.action == ActionType.CLICK
        assert step.target.startswith("locator(")
        assert ".get_by_role" in step.target
        assert ".first" in step.target

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


class TestSkipMarkers:
    """Tests for skip recording markers (# codevid: skip)."""

    @pytest.fixture
    def parser(self) -> PlaywrightParser:
        return PlaywrightParser()

    def test_parse_block_skip_markers(self, parser: PlaywrightParser, tmp_path: Path):
        """Test that block skip markers correctly mark steps."""
        test_file = tmp_path / "test_skip_block.py"
        test_file.write_text('''
from playwright.sync_api import Page

def test_with_skip_block(page: Page):
    # codevid: skip-start
    page.goto("https://example.com/login")
    page.fill("#username", "admin")
    page.click("#login")
    # codevid: skip-end
    page.click("#main-content")
    page.fill("#form", "data")
''')
        result = parser.parse(test_file)

        assert len(result.steps) == 5
        # First 3 steps should be marked as skip
        assert result.steps[0].skip_recording is True
        assert result.steps[1].skip_recording is True
        assert result.steps[2].skip_recording is True
        # Last 2 steps should NOT be skipped
        assert result.steps[3].skip_recording is False
        assert result.steps[4].skip_recording is False

    def test_parse_inline_skip_marker(self, parser: PlaywrightParser, tmp_path: Path):
        """Test that inline skip markers correctly mark single steps."""
        test_file = tmp_path / "test_skip_inline.py"
        test_file.write_text('''
from playwright.sync_api import Page

def test_with_inline_skip(page: Page):
    page.goto("https://example.com")
    page.fill("#password", "secret")  # codevid: skip
    page.click("#submit")
''')
        result = parser.parse(test_file)

        assert len(result.steps) == 3
        assert result.steps[0].skip_recording is False  # goto
        assert result.steps[1].skip_recording is True   # fill with skip marker
        assert result.steps[2].skip_recording is False  # click

    def test_parse_no_skip_markers(self, parser: PlaywrightParser, tmp_path: Path):
        """Test that files without skip markers have all steps with skip_recording=False."""
        test_file = tmp_path / "test_no_skip.py"
        test_file.write_text('''
from playwright.sync_api import Page

def test_without_skip(page: Page):
    page.goto("https://example.com")
    page.click("#button")
''')
        result = parser.parse(test_file)

        assert len(result.steps) == 2
        assert all(not step.skip_recording for step in result.steps)
        assert result.has_skip_markers() is False

    def test_has_skip_markers(self, parser: PlaywrightParser, tmp_path: Path):
        """Test the has_skip_markers() method on ParsedTest."""
        test_file = tmp_path / "test_has_skip.py"
        test_file.write_text('''
from playwright.sync_api import Page

def test_with_skip(page: Page):
    page.goto("https://example.com")  # codevid: skip
    page.click("#button")
''')
        result = parser.parse(test_file)

        assert result.has_skip_markers() is True

    def test_get_setup_steps(self, parser: PlaywrightParser, tmp_path: Path):
        """Test get_setup_steps() returns contiguous skipped steps at beginning."""
        test_file = tmp_path / "test_setup.py"
        test_file.write_text('''
from playwright.sync_api import Page

def test_with_setup(page: Page):
    # codevid: skip-start
    page.goto("https://example.com/login")
    page.fill("#user", "admin")
    page.click("#login")
    # codevid: skip-end
    page.click("#main")
    page.fill("#form", "data")
''')
        result = parser.parse(test_file)

        setup = result.get_setup_steps()
        assert len(setup) == 3
        assert all(step.skip_recording for step in setup)

    def test_get_recorded_steps(self, parser: PlaywrightParser, tmp_path: Path):
        """Test get_recorded_steps() returns steps between first and last non-skipped."""
        test_file = tmp_path / "test_recorded.py"
        test_file.write_text('''
from playwright.sync_api import Page

def test_with_recorded(page: Page):
    # codevid: skip-start
    page.goto("https://example.com/login")
    page.fill("#user", "admin")
    # codevid: skip-end
    page.click("#main")
    page.fill("#form", "data")
''')
        result = parser.parse(test_file)

        recorded = result.get_recorded_steps()
        assert len(recorded) == 2
        assert all(not step.skip_recording for step in recorded)

    def test_get_teardown_steps(self, parser: PlaywrightParser, tmp_path: Path):
        """Test get_teardown_steps() returns contiguous skipped steps at end."""
        test_file = tmp_path / "test_teardown.py"
        test_file.write_text('''
from playwright.sync_api import Page

def test_with_teardown(page: Page):
    page.goto("https://example.com")
    page.click("#main")
    # codevid: skip-start
    page.click("#logout")
    page.wait_for_load_state("networkidle")
    # codevid: skip-end
''')
        result = parser.parse(test_file)

        teardown = result.get_teardown_steps()
        assert len(teardown) == 2
        assert all(step.skip_recording for step in teardown)

    def test_mixed_setup_and_teardown(self, parser: PlaywrightParser, tmp_path: Path):
        """Test file with both setup and teardown skipped steps."""
        test_file = tmp_path / "test_mixed.py"
        test_file.write_text('''
from playwright.sync_api import Page

def test_mixed(page: Page):
    # codevid: skip-start
    page.goto("https://example.com/login")
    page.fill("#user", "admin")
    # codevid: skip-end
    page.click("#main")
    page.fill("#form", "data")
    # codevid: skip-start
    page.click("#logout")
    # codevid: skip-end
''')
        result = parser.parse(test_file)

        setup = result.get_setup_steps()
        recorded = result.get_recorded_steps()
        teardown = result.get_teardown_steps()

        assert len(setup) == 2
        assert len(recorded) == 2
        assert len(teardown) == 1

    def test_all_steps_skipped(self, parser: PlaywrightParser, tmp_path: Path):
        """Test when all steps are skipped."""
        test_file = tmp_path / "test_all_skipped.py"
        test_file.write_text('''
from playwright.sync_api import Page

def test_all_skipped(page: Page):
    # codevid: skip-start
    page.goto("https://example.com")
    page.click("#button")
    # codevid: skip-end
''')
        result = parser.parse(test_file)

        assert result.has_skip_markers() is True
        assert len(result.get_recorded_steps()) == 0

    def test_codevid_skip_without_space(self, parser: PlaywrightParser, tmp_path: Path):
        """Test that codevid:skip (without space) also works."""
        test_file = tmp_path / "test_no_space.py"
        test_file.write_text('''
from playwright.sync_api import Page

def test_no_space(page: Page):
    page.fill("#secret", "password")  #codevid:skip
    page.click("#submit")
''')
        result = parser.parse(test_file)

        assert result.steps[0].skip_recording is True
        assert result.steps[1].skip_recording is False
