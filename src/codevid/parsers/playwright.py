"""Parser for Playwright Python tests."""

import ast
from pathlib import Path

from codevid.models import ActionType, ParsedTest, TestStep
from codevid.parsers.base import ParseError, TestParser


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
                return self._get_string_arg(node, 0, context)
            return self._get_string_arg(node, 1, context) or self._get_string_arg(node, 0, context)
        return None

    def _is_chained_locator_call(self, node: ast.Call) -> bool:
        """Check if this call is chained from a locator method."""
        if not isinstance(node.func, ast.Attribute):
            return False
        if not isinstance(node.func.value, ast.Call):
            return False
        inner_method = self._get_method_name(node.func.value)
        return inner_method in ("locator", "get_by_role", "get_by_text",
                                "get_by_label", "get_by_placeholder",
                                "get_by_test_id", "get_by_alt_text")

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
        # Collect all locator methods in the chain (in reverse order)
        chain_parts: list[tuple[str, str]] = []
        current = node.func

        while isinstance(current, ast.Attribute):
            if isinstance(current.value, ast.Call):
                inner_call = current.value
                inner_method = self._get_method_name(inner_call)

                if inner_method in self.LOCATOR_CHAIN_METHODS:
                    selector = self._get_string_arg(inner_call, 0, context)
                    if selector:
                        chain_parts.insert(0, (inner_method, selector))

                # Continue up the chain to find more locator methods
                current = inner_call.func
            else:
                current = current.value

        if not chain_parts:
            return None

        # Single simple locator: return as before for backwards compatibility
        if len(chain_parts) == 1:
            method, selector = chain_parts[0]
            if method == "locator":
                return selector
            return f"{method}({selector!r})"

        # Multiple chain: return full expression
        # e.g., "get_by_text('Atlas Runner Sneaker').locator('..')"
        return ".".join(f"{m}({s!r})" for m, s in chain_parts)

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
            selector = self._get_string_arg(node, 0, context)
            if selector:
                if method != "locator":
                    return f"{method}({selector!r})"
                return selector
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
