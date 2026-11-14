"""Verify that all eval CLI options are documented in the README.

Uses pytest-subtests so each flag is reported individually on failure.
"""

import argparse
import re
from pathlib import Path
from unittest.mock import patch

# Flags that argparse adds automatically and don't need README docs
_ARGPARSE_BUILTINS = {"-h", "--help"}

README_PATH = Path(__file__).parent.parent / "evals" / "README-technical.md"

# Map: module import path -> section heading text in README-technical.md
_CLI_MODULES = {
    "evals.harvest": "### harvest",
    "evals.review": "### review",
    "evals.run_eval": "### run_eval",
    "evals.report": "### report",
    "evals.compare": "### compare",
}


def _capture_parser(module_name: str) -> argparse.ArgumentParser:
    """Import an eval module and capture its ArgumentParser before parse_args runs."""
    captured = {}

    original_parse_args = argparse.ArgumentParser.parse_args

    def intercept(self, args=None, namespace=None):
        captured["parser"] = self
        raise SystemExit(0)  # Stop cli() from proceeding

    with patch.object(argparse.ArgumentParser, "parse_args", intercept):
        import importlib
        mod = importlib.import_module(module_name)
        try:
            mod.cli()
        except SystemExit:
            pass

    argparse.ArgumentParser.parse_args = original_parse_args
    return captured["parser"]


def _get_parser_flags(parser: argparse.ArgumentParser) -> set[str]:
    """Extract all --flag names from a parser.

    Excludes -h/--help and hidden flags (help=argparse.SUPPRESS), which are
    intentionally undocumented — e.g. deprecated no-op aliases kept only so old
    automation doesn't error.
    """
    flags = set()
    for action in parser._actions:
        if action.help == argparse.SUPPRESS:
            continue
        for opt in action.option_strings:
            if opt.startswith("--") and opt not in _ARGPARSE_BUILTINS:
                flags.add(opt)
    return flags


def _get_readme_section(full_text: str, heading: str) -> str:
    """Extract the README section starting at heading, ending at the next ### or ##."""
    pattern = re.escape(heading) + r".*?(?=\n###?\s|\Z)"
    match = re.search(pattern, full_text, re.DOTALL)
    return match.group(0) if match else ""


def _get_documented_flags(section_text: str) -> set[str]:
    """Extract all `--flag` patterns from backtick-quoted text in a README section."""
    return set(re.findall(r"`(--[\w-]+)`", section_text))


def test_all_eval_cli_options_documented(subtests):
    readme_text = README_PATH.read_text()

    for module_name, heading in _CLI_MODULES.items():
        parser = _capture_parser(module_name)
        parser_flags = _get_parser_flags(parser)
        section = _get_readme_section(readme_text, heading)
        documented_flags = _get_documented_flags(section)

        for flag in sorted(parser_flags):
            with subtests.test(module=module_name, flag=flag):
                assert flag in documented_flags, (
                    f"{flag} from {module_name} is not documented in README "
                    f"under '{heading}'"
                )
