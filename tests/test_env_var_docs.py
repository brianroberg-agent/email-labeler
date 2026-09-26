"""Verify that all environment variables used by the daemon are documented in the README.

Scans daemon source files for os.environ references and config.toml for
{env.VAR} placeholders, then checks each variable appears in the README's
Environment Variables section.

Uses pytest-subtests so each variable is reported individually on failure.
"""

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
README_PATH = ROOT / "README-technical.md"
CONFIG_PATH = ROOT / "config.toml"

# Daemon runtime source files (not tests, not evals)
_DAEMON_SOURCES = [p for p in ROOT.glob("*.py") if p.name != "setup.py"]

# Pattern: os.environ.get("VAR_NAME") or os.environ["VAR_NAME"]
_ENVIRON_GET = re.compile(r'os\.environ\.get\(\s*["\'](\w+)["\']')
_ENVIRON_BRACKET = re.compile(r'os\.environ\[\s*["\'](\w+)["\']')
# Pattern: resolve_int_env("VAR_NAME", ...) — daemon.py's numeric overrides
# (MAX_FAILURES, LOCAL_PARALLEL, WRITE_PARALLEL, MAX_EMAILS_PER_CYCLE,
# HALT_PROBE_INTERVAL_SECONDS, BALANCE_HALT_STRIKES) read the environment
# through that helper, not os.environ directly (review of #81, Opus F5).
_RESOLVE_INT_ENV = re.compile(r'resolve_int_env\(\s*["\'](\w+)["\']')

# Pattern: {env.VAR_NAME} in config.toml
_CONFIG_ENV = re.compile(r'\{env\.(\w+)\}')


def _collect_env_vars() -> set[str]:
    """Collect all environment variable names referenced by daemon code."""
    env_vars: set[str] = set()

    for source_file in _DAEMON_SOURCES:
        text = source_file.read_text()
        env_vars.update(_ENVIRON_GET.findall(text))
        env_vars.update(_ENVIRON_BRACKET.findall(text))
        env_vars.update(_RESOLVE_INT_ENV.findall(text))

    if CONFIG_PATH.exists():
        env_vars.update(_CONFIG_ENV.findall(CONFIG_PATH.read_text()))

    return env_vars


def _get_env_vars_section(readme_text: str) -> str:
    """Extract the Environment Variables section from the README."""
    match = re.search(
        r"## Environment Variables.*?(?=\n## |\Z)", readme_text, re.DOTALL
    )
    return match.group(0) if match else ""


def test_all_daemon_env_vars_documented(subtests):
    readme_text = README_PATH.read_text()
    env_section = _get_env_vars_section(readme_text)
    env_vars = _collect_env_vars()

    assert env_vars, "No environment variables found — scan may be broken"
    assert env_section, "No '## Environment Variables' section found in README"

    for var in sorted(env_vars):
        with subtests.test(var=var):
            assert var in env_section, (
                f"`{var}` is referenced in daemon source code but not "
                f"documented in the README's Environment Variables section"
            )
