"""Parser for Playwright Python tests."""

import ast
import io
import tokenize
from pathlib import Path
from typing import NamedTuple

from codevid.models import ActionType, ParsedTest, TestStep
from codevid.parsers.base import ParseError, TestParser


class SkipRegion(NamedTuple):
    """A region of lines that should be skipped from recording."""

    start_line: int
    end_line: int


class PlaywrightParser(TestParser):
    """Parse Playwright Python test files using AST analysis."""

    # Map Playwright methods to ActionTypes
    ACTION_MAP: dict[str, ActionType] = {
        # Navigation
        "goto": ActionType.NAVIGATE,
        "go_back": ActionType.NAVIGATE,
        "go_forward": ActionType.NAVIGATE,
        "reload": ActionType.NAVIGATE,
        # Clicks
        "click": ActionType.CLICK,
        "dblclick": ActionType.CLICK,
        "tap": ActionType.CLICK,
        # Input
        "fill": ActionType.TYPE,
        "type": ActionType.TYPE,
        "press": ActionType.PRESS,
        "press_sequentially": ActionType.TYPE,
        "clear": ActionType.TYPE,
        "set_input_files": ActionType.TYPE,
        # Selection
        "select_option": ActionType.SELECT,
        "select_text": ActionType.SELECT,
        "check": ActionType.CLICK,
        "uncheck": ActionType.CLICK,
        "set_checked": ActionType.CLICK,
        # Hover/Focus
        "hover": ActionType.HOVER,
        "focus": ActionType.HOVER,
        # Scroll
        "scroll_into_view_if_needed": ActionType.SCROLL,
        # Wait
        "wait_for_selector": ActionType.WAIT,
        "wait_for_load_state": ActionType.WAIT,
        "wait_for_url": ActionType.WAIT,
        "wait_for_timeout": ActionType.WAIT,
        # Assertions (from expect)
        "to_be_visible": ActionType.ASSERT,
        "to_be_hidden": ActionType.ASSERT,
        "to_be_enabled": ActionType.ASSERT,
        "to_be_disabled": ActionType.ASSERT,
        "to_have_text": ActionType.ASSERT,
        "to_have_value": ActionType.ASSERT,
        "to_have_attribute": ActionType.ASSERT,
        "to_have_class": ActionType.ASSERT,
        "to_have_count": ActionType.ASSERT,
        "to_have_url": ActionType.ASSERT,
        "to_have_title": ActionType.ASSERT,
        "to_contain_text": ActionType.ASSERT,
        # Screenshot
        "screenshot": ActionType.SCREENSHOT,
    }

    # Methods that are on page.locator() chain
    LOCATOR_METHODS = {
        "click", "dblclick", "tap", "fill", "type", "press", "press_sequentially",
        "clear", "hover", "focus", "check", "uncheck", "set_checked", "select_option",
        "scroll_into_view_if_needed", "screenshot", "set_input_files",
    }

    # Skip marker patterns
    SKIP_START_MARKERS = {"codevid: skip-start", "codevid:skip-start"}
    SKIP_END_MARKERS = {"codevid: skip-end", "codevid:skip-end"}
    SKIP_INLINE_MARKERS = {"codevid: skip", "codevid:skip"}

    def _extract_skip_regions(self, source: str) -> list[SkipRegion]:
        """Extract line ranges marked for skip using tokenize.

        Detects both block markers (skip-start/skip-end) and inline markers (skip).
        Returns a list of SkipRegion tuples indicating which lines should be skipped.
        """
        regions: list[SkipRegion] = []
        block_start: int | None = None

        try:
            tokens = tokenize.generate_tokens(io.StringIO(source).readline)
            for tok in tokens:
                if tok.type == tokenize.COMMENT:
                    comment_text = tok.string.lstrip("#").strip().lower()
                    line_no = tok.start[0]

                    # Check for block start
                    if any(marker in comment_text for marker in self.SKIP_START_MARKERS):
                        block_start = line_no

                    # Check for block end
                    elif any(marker in comment_text for marker in self.SKIP_END_MARKERS):
                        if block_start is not None:
                            regions.append(SkipRegion(block_start, line_no))
                            block_start = None

                    # Check for inline skip marker
                    elif any(marker in comment_text for marker in self.SKIP_INLINE_MARKERS):
                        # Inline marker applies to the same line
                        regions.append(SkipRegion(line_no, line_no))

        except tokenize.TokenError:
            pass  # Gracefully handle malformed files

        return regions

    def _is_line_in_skip_region(
        self,
        line_number: int,
        skip_regions: list[SkipRegion],
    ) -> bool:
        """Check if a line falls within any skip region."""
        for region in skip_regions:
            if region.start_line <= line_number <= region.end_line:
                return True
        return False

    @property
    def framework_name(self) -> str:
        return "playwright"

    def can_parse(self, file_path: str | Path) -> bool:
        """Check if this is a Playwright Python test file."""
        path = Path(file_path)

        if path.suffix != ".py":
            return False

        try:
            content = path.read_text()
            # Look for Playwright imports or fixtures
            return any(marker in content for marker in [
                "from playwright",
                "import playwright",
                "def test_",
                "async def test_",
                "page: Page",
                "page.goto",
            ])
        except Exception:
            return False

    def parse(self, file_path: str | Path) -> ParsedTest:
        """Parse a Playwright test file and extract steps."""
        path = Path(file_path)

        try:
            source = path.read_text()
        except Exception as e:
            raise ParseError(f"Failed to read file: {e}", file_path)

        # Extract skip regions BEFORE AST parsing (AST doesn't preserve comments)
        skip_regions = self._extract_skip_regions(source)

        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            raise ParseError(f"Syntax error: {e.msg}", file_path, e.lineno)

        # Find test functions
        test_functions = self._find_test_functions(tree)

        if not test_functions:
            raise ParseError("No test functions found", file_path)

        # Parse the first test function (or combine multiple)
        test_func = test_functions[0]
        steps = self._extract_steps(test_func, source)

        # Apply skip markers to steps based on their line numbers
        for step in steps:
            step.skip_recording = self._is_line_in_skip_region(
                step.line_number, skip_regions
            )

        # Extract test metadata
        test_name = test_func.name
        docstring = ast.get_docstring(test_func) or ""

        return ParsedTest(
            name=test_name,
            file_path=str(path),
            steps=steps,
            setup_code="",
            teardown_code="",
            metadata={
                "docstring": docstring,
                "is_async": isinstance(test_func, ast.AsyncFunctionDef),
                "function_count": len(test_functions),
                "has_skip_markers": len(skip_regions) > 0,
            },
        )

    def _find_test_functions(self, tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
        """Find all test functions in the AST."""
        tests = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_"):
                    tests.append(node)

        return tests

    def _extract_steps(
        self,
        func: ast.FunctionDef | ast.AsyncFunctionDef,
        source: str,
    ) -> list[TestStep]:
        """Extract test steps from a function body."""
        steps = []
        # Track processed call node ids to avoid duplicates from await
        processed_calls: set[int] = set()

        # Simple local variable context (name -> value)
        # We only track string constants assigned to variables
        context: dict[str, str] = {}

        for node in func.body:
            # Handle Assignments: var = "value"
            if isinstance(node, ast.Assign):
                self._update_context(node, context)
                continue

            # Walk the node to find calls
            for child in ast.walk(node):
                # Handle await expressions - mark the inner call as processed
                if isinstance(child, ast.Await) and isinstance(child.value, ast.Call):
                    call_node = child.value
                    if id(call_node) not in processed_calls:
                        processed_calls.add(id(call_node))
                        step = self._parse_call(call_node, source, context)
                        if step:
                            steps.append(step)
                elif isinstance(child, ast.Call):
                    # Skip if already processed via await
                    if id(child) not in processed_calls:
                        processed_calls.add(id(child))
                        step = self._parse_call(child, source, context)
                        if step:
                            steps.append(step)

        # Sort by line number
        steps.sort(key=lambda s: s.line_number)

        return steps

    def _update_context(self, node: ast.Assign, context: dict[str, str]) -> None:
        """Update context with variable assignments if value is a string constant."""
        # Only handle simple assignment to a single target: x = "str"
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                context[var_name] = node.value.value

    def _parse_call(self, node: ast.Call, source: str, context: dict[str, str]) -> TestStep | None:
        """Parse a function call into a TestStep."""
        method_name = self._get_method_name(node)
        if method_name is None:
            return None

        action_type = self.ACTION_MAP.get(method_name)
        if action_type is None:
            return None

        # Get target (selector or URL)
        target = self._extract_target(node, method_name, context)

        # Get value (for fill, type, etc.)
        value = self._extract_value(node, method_name, context)

        # Get source code for this node
        source_code = self._get_source_segment(source, node)

        # Generate description
        description = self._generate_description(action_type, method_name, target, value)

        return TestStep(
            action=action_type,
            target=target,
            value=value,
            description=description,
            line_number=node.lineno,
            source_code=source_code,
        )

    def _get_method_name(self, node: ast.Call) -> str | None:
        """Extract the method name from a call node."""
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def _extract_target(self, node: ast.Call, method_name: str, context: dict[str, str]) -> str:
        """Extract the target selector or URL from arguments."""
        # For navigation methods, first arg is URL
        if method_name in ("goto", "wait_for_url"):
            return self._get_string_arg(node, 0, context) or ""

        # For locator methods, we need to find the selector
        # This could be from page.locator("selector") or page.click("selector")
        if method_name in self.LOCATOR_METHODS:
            # First check if this is a chained call: page.locator("selector").click()
            selector = self._find_locator_selector(node, context)
            if selector:
                return selector

            # Direct page method: page.click("selector")
            selector = self._get_string_arg(node, 0, context)
            if selector:
                return selector

        # For expect assertions, find the locator
        if method_name.startswith("to_"):
            selector = self._find_expect_locator(node, context)
            if selector:
                return selector

        return self._get_string_arg(node, 0, context) or ""

    def _extract_value(self, node: ast.Call, method_name: str, context: dict[str, str]) -> str | None:
        """Extract the value argument for input methods."""
        if method_name in ("fill", "type", "press_sequentially"):
            # Check if chained (page.locator().fill("value")) - value is arg 0
            # Or direct (page.fill("#sel", "value")) - value is arg 1
            if self._is_chained_locator_call(node):
                return self._get_string_arg(node, 0, context)
            return self._get_string_arg(node, 1, context)
        if method_name == "press":
            return self._get_string_arg(node, 0, context)
        if method_name in ("to_have_text", "to_contain_text", "to_have_value"):
            return self._get_string_arg(node, 0, context)
        if method_name == "select_option":
            # Same chain detection for select_option
            if self._is_chained_locator_call(node):
                return self._get_string_arg(node, 0, context) or self._get_select_option_kw(node, context)

            return (
                self._get_string_arg(node, 1, context)
                or self._get_select_option_kw(node, context)
                or self._get_string_arg(node, 0, context)
            )
        return None

    def _get_select_option_kw(self, node: ast.Call, context: dict[str, str]) -> str | None:
        for kw in node.keywords:
            if kw.arg not in ("label", "value"):
                continue

            prefix = f"{kw.arg}="

            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                return prefix + kw.value.value
            if isinstance(kw.value, ast.Name):
                resolved = context.get(kw.value.id)
                return (prefix + resolved) if resolved is not None else None
            if isinstance(kw.value, ast.JoinedStr):
                return prefix + self._extract_fstring(kw.value, context)

        return None

    def _is_chained_locator_call(self, node: ast.Call) -> bool:
        """Check if this call is chained from a locator method."""
        if not isinstance(node.func, ast.Attribute):
            return False
        value: ast.AST = node.func.value

        # Unwrap locator properties like .first / .last
        # e.g., page.get_by_role(...).first.fill("x")
        if isinstance(value, ast.Attribute) and value.attr in ("first", "last"):
            value = value.value

        if not isinstance(value, ast.Call):
            return False

        inner_method = self._get_method_name(value)
        return inner_method in (
            "locator",
            "get_by_role",
            "get_by_text",
            "get_by_label",
            "get_by_placeholder",
            "get_by_test_id",
            "get_by_alt_text",
            "nth",
        )

    def _get_string_arg(self, node: ast.Call, index: int, context: dict[str, str]) -> str | None:
        """Get a string argument at the given index."""
        if index < len(node.args):
            arg = node.args[index]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                return arg.value
            if isinstance(arg, ast.Name):
                # Resolve variable
                return context.get(arg.id, None)
            if isinstance(arg, ast.JoinedStr):
                # f-string - try to extract static parts
                return self._extract_fstring(arg, context)
        return None

    def _extract_fstring(self, node: ast.JoinedStr, context: dict[str, str]) -> str:
        """Extract a simplified representation of an f-string."""
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue) and isinstance(value.value, ast.Name):
                # Try to resolve variable in f-string
                var_value = context.get(value.value.id)
                parts.append(var_value if var_value is not None else "{...}")
            else:
                parts.append("{...}")
        return "".join(parts)

    # Locator methods that can be chained
    LOCATOR_CHAIN_METHODS = {
        "locator", "get_by_role", "get_by_text", "get_by_label",
        "get_by_placeholder", "get_by_test_id", "get_by_alt_text",
    }

    def _find_locator_selector(self, node: ast.Call, context: dict[str, str]) -> str | None:
        """Find selector from chained locator call like page.locator("sel").click().

        Handles chains like: page.get_by_text("X").locator("..").click()
        Returns the full chain expression: "get_by_text('X').locator('..')"
        """
        if not isinstance(node.func, ast.Attribute):
            return None

        # node.func.value is the receiver the action is called on.
        # Example: page.get_by_role(...).first.click()
        # node.func = Attribute(attr='click', value=Attribute(attr='first', value=Call(get_by_role...)))
        return self._format_locator_expr(node.func.value, context)

    def _format_locator_expr(self, expr: ast.AST, context: dict[str, str]) -> str | None:
        """Format a locator expression (without the leading `page.`) into a string.

        Examples:
        - page.locator("#form") -> "locator('#form')"
        - page.get_by_role("link", name="X").first -> "get_by_role('link', name='X').first"
        - page.get_by_text("X").locator("..")-> "get_by_text('X').locator('..')"
        """
        # Properties like .first / .last
        if isinstance(expr, ast.Attribute):
            if expr.attr in ("first", "last"):
                base = self._format_locator_expr(expr.value, context)
                if base is None:
                    return None
                return f"{base}.{expr.attr}"
            return None

        # Method calls like .locator(...), .get_by_role(...), .nth(...)
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
            method = expr.func.attr
            if method not in self.LOCATOR_CHAIN_METHODS and method != "nth":
                return None

            base = self._format_locator_expr(expr.func.value, context)
            args_str = self._get_all_args_string(expr, context)
            part = f"{method}({args_str})"

            if base is None:
                return part
            return f"{base}.{part}"

        # Stop at names (e.g., the `page` fixture)
        if isinstance(expr, ast.Name):
            return None

        return None

    def _get_all_args_string(self, node: ast.Call, context: dict[str, str]) -> str:
        """Get string representation of all arguments for a call."""
        args: list[str] = []
        # Positional args
        for arg in node.args:
            val = self._eval_arg(arg, context)
            args.append(repr(val))

        # Keyword args
        for kw in node.keywords:
            val = self._eval_arg(kw.value, context)
            args.append(f"{kw.arg}={repr(val)}")

        return ", ".join(args)

    def _eval_arg(self, arg: ast.AST, context: dict[str, str]) -> str | bool | int | float | None:
        """Evaluate an argument node to a python value."""
        if isinstance(arg, ast.Constant):
            return arg.value
        if isinstance(arg, ast.Name):
            return context.get(arg.id, arg.id)  # Return value or name if unknown
        return "..."  # Fallback for complex expressions

    def _find_expect_locator(self, node: ast.Call, context: dict[str, str]) -> str | None:
        """Find the locator from an expect() assertion chain."""
        # expect(page.locator("sel")).to_be_visible()
        current = node.func
        while isinstance(current, ast.Attribute):
            if isinstance(current.value, ast.Call):
                inner = current.value
                inner_method = self._get_method_name(inner)
                is_expect_name = isinstance(inner.func, ast.Name) and inner.func.id == "expect"
                if inner_method == "expect" or is_expect_name:
                    # Found expect(), check its argument
                    if inner.args:
                        arg = inner.args[0]
                        if isinstance(arg, ast.Call):
                            return self._find_locator_selector_from_call(arg, context)
                current = inner.func
            else:
                break
        return None

    def _find_locator_selector_from_call(self, node: ast.Call, context: dict[str, str]) -> str | None:
        """Extract selector from a locator call."""
        method = self._get_method_name(node)
        if method in ("locator", "get_by_role", "get_by_text",
                      "get_by_label", "get_by_placeholder",
                      "get_by_test_id", "get_by_alt_text"):
            args_str = self._get_all_args_string(node, context)
            return f"{method}({args_str})"
        return None

    def _get_source_segment(self, source: str, node: ast.AST) -> str:
        """Get the source code for an AST node."""
        try:
            return ast.get_source_segment(source, node) or ""
        except Exception:
            return ""

    def _generate_description(
        self,
        action: ActionType,
        method: str,
        target: str,
        value: str | None,
    ) -> str:
        """Generate a human-readable description of the step."""
        descriptions = {
            ActionType.NAVIGATE: f"Navigate to {target}",
            ActionType.CLICK: f"Click on {self._humanize_selector(target)}",
            ActionType.TYPE: f"Type '{value}' into {self._humanize_selector(target)}" if value else f"Clear {self._humanize_selector(target)}",
            ActionType.PRESS: f"Press {value} key",
            ActionType.HOVER: f"Hover over {self._humanize_selector(target)}",
            ActionType.SELECT: f"Select '{value}' from {self._humanize_selector(target)}" if value else f"Select option from {self._humanize_selector(target)}",
            ActionType.SCROLL: f"Scroll to {self._humanize_selector(target)}",
            ActionType.WAIT: f"Wait for {self._humanize_selector(target)}",
            ActionType.ASSERT: self._describe_assertion(method, target, value),
            ActionType.SCREENSHOT: f"Take screenshot of {self._humanize_selector(target)}" if target else "Take screenshot",
        }
        return descriptions.get(action, f"{method} on {target}")

    def _humanize_selector(self, selector: str) -> str:
        """Convert a selector to a more human-readable form."""
        if not selector:
            return "element"

        # Handle get_by_* methods
        if selector.startswith("get_by_"):
            return selector.replace("get_by_", "").replace("_", " ").replace("(", " ").rstrip(")")

        # Handle common selector patterns
        if selector.startswith("#"):
            return f"'{selector[1:]}' element"
        if selector.startswith("."):
            return f"element with class '{selector[1:]}'"
        if selector.startswith("["):
            return f"element matching {selector}"
        if selector.startswith("text="):
            return f"text '{selector[5:]}'"
        if selector.startswith("//"):
            return "element"  # XPath is too complex to humanize

        # Data-testid pattern
        if "data-testid" in selector or "test-id" in selector:
            return f"'{selector}' test element"

        return f"'{selector}'"

    def _describe_assertion(self, method: str, target: str, value: str | None) -> str:
        """Generate description for assertion methods."""
        target_desc = self._humanize_selector(target)

        assertions = {
            "to_be_visible": f"Verify {target_desc} is visible",
            "to_be_hidden": f"Verify {target_desc} is hidden",
            "to_be_enabled": f"Verify {target_desc} is enabled",
            "to_be_disabled": f"Verify {target_desc} is disabled",
            "to_have_text": f"Verify {target_desc} has text '{value}'",
            "to_contain_text": f"Verify {target_desc} contains '{value}'",
            "to_have_value": f"Verify {target_desc} has value '{value}'",
            "to_have_url": f"Verify URL is '{value}'",
            "to_have_title": f"Verify page title is '{value}'",
        }
        return assertions.get(method, f"Assert {method} on {target_desc}")
