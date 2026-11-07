"""The mechanical half of CODING_STANDARDS.md "Test through the MCP surface": which package modules tests may import."""

import ast
from pathlib import Path

# Pure modules, the stores tests seed state through, and the error type tests catch.
ANY_TEST = {"bars", "errors", "portfolio", "prompt", "watchlists"}
# Files that need more, each for a reason no tool can give.
EXCEPTIONS = {
    "conftest.py": {"cache", "server", "signal_maker"},  # builds the fixtures
    "fakes.py": {"signals"},  # seed_signals creates the signals table
    "test_classifier.py": {"classifier"},  # tests the OpenRouter transport itself
    "test_golden_prompt.py": {"classifier"},  # pins what the real transport sends
    "test_logging_setup.py": {"__main__"},  # logging is set up at the entry point
}


def _imported_modules(path: Path) -> set[str]:
    modules = set()
    for node in ast.walk(ast.parse(path.read_text())):
        names = [node.module or ""] if isinstance(node, ast.ImportFrom) else []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        modules |= {name.split(".")[1] for name in names if name.startswith("stock_market_mcp.")}
    return modules


def test_tests_import_only_allowed_package_modules():
    tests = Path(__file__).parent
    breaches = {
        path.name: sorted(extra)
        for path in sorted(tests.glob("*.py"))
        if (extra := _imported_modules(path) - ANY_TEST - EXCEPTIONS.get(path.name, set()))
    }
    assert not breaches, (
        f"{breaches}: test through the MCP surface or the signal_maker fixture, or add the file to EXCEPTIONS with "
        "the reason no tool can give"
    )
