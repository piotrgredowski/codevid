"""Parser registry for automatic framework detection."""

from pathlib import Path

from codevid.parsers.base import ParseError, TestParser


class ParserRegistry:
    """Registry for test file parsers with auto-detection."""

    def __init__(self) -> None:
        self._parsers: list[TestParser] = []

    def register(self, parser: TestParser) -> None:
        """Register a parser."""
        self._parsers.append(parser)

    def get_parser(self, file_path: str | Path) -> TestParser:
        """Get the appropriate parser for a file.

        Args:
            file_path: Path to the test file.

        Returns:
            A parser that can handle the file.

        Raises:
            ParseError: If no parser can handle the file.
        """
        path = Path(file_path)

        for parser in self._parsers:
            if parser.can_parse(path):
                return parser

        raise ParseError(
            f"No parser found for file type. "
            f"Supported frameworks: {', '.join(p.framework_name for p in self._parsers)}",
            file_path,
        )

    def list_frameworks(self) -> list[str]:
        """List all registered framework names."""
        return [p.framework_name for p in self._parsers]


# Global registry instance
_registry: ParserRegistry | None = None


def get_registry() -> ParserRegistry:
    """Get the global parser registry, initializing if needed."""
    global _registry

    if _registry is None:
        _registry = ParserRegistry()
        _register_default_parsers(_registry)

    return _registry


def _register_default_parsers(registry: ParserRegistry) -> None:
    """Register all built-in parsers."""
    from codevid.parsers.playwright import PlaywrightParser

    registry.register(PlaywrightParser())


def get_parser(file_path: str | Path) -> TestParser:
    """Get the appropriate parser for a file.

    This is a convenience function that uses the global registry.
    """
    return get_registry().get_parser(file_path)


def parse_test(file_path: str | Path):
    """Parse a test file using the appropriate parser.

    This is a convenience function for quick parsing.
    """
    parser = get_parser(file_path)
    return parser.parse(file_path)
