"""Base parser interface for test files."""

from abc import ABC, abstractmethod
from pathlib import Path

from codevid.models import ParsedTest


class TestParser(ABC):
    """Abstract base class for test file parsers."""

    @abstractmethod
    def parse(self, file_path: str | Path) -> ParsedTest:
        """Parse a test file and extract steps.

        Args:
            file_path: Path to the test file.

        Returns:
            ParsedTest object containing extracted test information.

        Raises:
            ParseError: If the file cannot be parsed.
        """
        pass

    @abstractmethod
    def can_parse(self, file_path: str | Path) -> bool:
        """Check if this parser can handle the given file.

        Args:
            file_path: Path to the test file.

        Returns:
            True if this parser can handle the file.
        """
        pass

    @property
    @abstractmethod
    def framework_name(self) -> str:
        """Return the name of the testing framework this parser handles."""
        pass


class ParseError(Exception):
    """Raised when a test file cannot be parsed."""

    def __init__(self, message: str, file_path: str | Path, line_number: int | None = None):
        self.file_path = str(file_path)
        self.line_number = line_number
        location = f"{file_path}"
        if line_number:
            location += f":{line_number}"
        super().__init__(f"{location}: {message}")
