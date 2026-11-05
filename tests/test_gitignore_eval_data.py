"""Guard: eval and labeling data that holds real email content stays out of git.

These files carry full newsletter bodies, story texts and sender names. They are
created at runtime on the machine that labels, so the only thing standing
between them and a broad ``git add`` is ``.gitignore``. The check asks git
itself (``git check-ignore``) rather than matching the file's text, so a
pattern that looks right but does not match is still caught.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Paths the tools write by default. Each must be ignored.
CONTENT_BEARING_PATHS = [
    "evals/golden_set.jsonl",
    "evals/results/run.jsonl",
    "evals/cache/llm_cache.jsonl",
    "data/newsletter_assessments.jsonl",
    "evals/newsletter_golden_set.jsonl",
    "evals/newsletter_results/run.jsonl",
    "evals/newsletter_results/run.thinking.jsonl",
    "evals/newsletter_story_labels.jsonl",
]


def _git_work_tree() -> bool:
    if shutil.which("git") is None:
        return False
    probe = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    return probe.returncode == 0 and probe.stdout.strip() == "true"


@pytest.mark.skipif(not _git_work_tree(), reason="needs git and a work tree")
@pytest.mark.parametrize("path", CONTENT_BEARING_PATHS)
def test_content_bearing_eval_path_is_gitignored(path):
    result = subprocess.run(
        ["git", "check-ignore", "-q", path], cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"{path} is not gitignored: it holds real email content and a broad "
        "`git add` would stage it"
    )
