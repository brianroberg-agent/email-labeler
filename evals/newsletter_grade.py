"""Grade ready-made newsletter stories into an append-only label log.

The production daemon already extracts story texts from every newsletter it
grades (the assessments JSONL). This module turns those texts into a queue of
stories for the operator to grade by the rubric, blind to what the model said,
and records each judgment as one self-contained line in a label log.

Cutting and rating are separate jobs: a story whose boundaries are wrong is
marked *miscut* and set aside for recutting, never graded and never discarded.
"""

import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

from evals.newsletter_schemas import StoryLabel
from newsletter import _DIMENSIONS, _VALID_THEMES

# Two texts from one thread count as the same story at or above this overlap.
# Same value the report uses to match extracted stories to golden ones.
OVERLAP_THRESHOLD = 0.6

# A shared run shorter than this many words is coincidence ("in the fall of"),
# not the same passage, and does not count toward overlap.
_MIN_SHARED_WORDS = 4


@dataclass(frozen=True)
class Candidate:
    """One story waiting to be graded. Carries nothing the model said about it."""

    story_key: str
    thread_id: str
    text: str
    source_timestamp: str = ""
    recut_of: str = ""


@dataclass(frozen=True)
class Rubric:
    """The rubric wording shown beside a story while it is graded."""

    grades: dict[str, str]  # poor/ok/good -> anchor
    dimensions: dict[str, str]  # dimension -> the question it asks
    theme_grades: dict[str, str]  # absent/present/emphasized -> anchor
    themes: dict[str, str]  # theme -> definition


_RUBRIC_KEYS = {
    "grades": ("poor", "ok", "good"),
    "dimensions": _DIMENSIONS,
    "theme_grades": ("absent", "present", "emphasized"),
    "themes": tuple(sorted(name.lower() for name in _VALID_THEMES)),
}


def load_rubric(config: dict) -> Rubric:
    """Read ``[newsletter.rubric]`` from a loaded config.

    Raises ValueError naming the table and entry when one is missing or
    unexpected: a card with a blank question is worse than no card.
    """
    table = config.get("newsletter", {}).get("rubric")
    if not table:
        raise ValueError("config has no [newsletter.rubric] table")
    for name, expected in _RUBRIC_KEYS.items():
        entries = table.get(name, {})
        missing = [key for key in expected if not entries.get(key)]
        if missing:
            raise ValueError(
                f"[newsletter.rubric.{name}] is missing {', '.join(missing)}"
            )
        unknown = sorted(set(entries) - set(expected))
        if unknown:
            raise ValueError(
                f"[newsletter.rubric.{name}] has unknown entries {', '.join(unknown)}"
            )
    return Rubric(
        grades=dict(table["grades"]),
        dimensions=dict(table["dimensions"]),
        theme_grades=dict(table["theme_grades"]),
        themes=dict(table["themes"]),
    )


def story_key(text: str) -> str:
    """A story's identity: a hash of its text with whitespace normalized.

    Text-only on purpose. A recut of the same newsletter is different text and
    so a different story; the same text arriving in two threads is one story.
    """
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def read_label_log(path: Path) -> list[StoryLabel]:
    """Every label line in *path*, in file order. A missing file is an empty log.

    Blank lines are skipped. A line that is not JSON, or not a valid label,
    raises ValueError naming the file and line, so a bad hand-merge is located
    rather than silently dropped.
    """
    path = Path(path)
    if not path.exists():
        return []
    labels = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                labels.append(StoryLabel.from_dict(json.loads(line)))
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return labels


def append_label(path: Path, label: StoryLabel) -> None:
    """Append one label line to *path* and sync it to disk.

    Earlier lines are never rewritten. If the file does not end in a newline
    (a hand-merge, an interrupted write), one is added first so the new label
    lands on its own line.
    """
    line = (json.dumps(label.to_dict()) + "\n").encode()
    with open(path, "a+b") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _labeled_at(label: StoryLabel) -> datetime | None:
    try:
        stamp = datetime.fromisoformat(label.labeled_at or "")
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def latest_labels(labels: list[StoryLabel]) -> dict[str, StoryLabel]:
    """The label in force for each story: the newest by ``labeled_at``.

    File order is only the tie-break (a later line wins) when two lines carry
    equal or unreadable times, because two machines' logs are merged by
    concatenation and file order is then not time order.
    """
    latest: dict[str, StoryLabel] = {}
    for label in labels:
        kept = latest.get(label.story_key)
        if kept is not None:
            new_at, kept_at = _labeled_at(label), _labeled_at(kept)
            if new_at is not None and kept_at is not None and new_at < kept_at:
                continue
        latest[label.story_key] = label
    return latest


def text_overlap(a: str, b: str) -> float:
    """How much of the shorter text also appears in the other, from 0.0 to 1.0.

    Counted in words over shared runs of at least ``_MIN_SHARED_WORDS``, and
    measured against the SHORTER text: a cut that stops early is wholly inside
    its repaired version, so the two overlap fully even though one is several
    times the other's length.
    """
    words_a, words_b = a.split(), b.split()
    shorter = min(len(words_a), len(words_b))
    if shorter == 0:
        return 0.0
    matcher = SequenceMatcher(None, words_a, words_b, autojunk=False)
    shared = sum(
        block.size for block in matcher.get_matching_blocks()
        if block.size >= min(_MIN_SHARED_WORDS, shorter)
    )
    return shared / shorter


def build_queue(
    assessments: list[dict],
    curated_thread_ids: set[str],
    labels: dict[str, StoryLabel],
) -> list[Candidate]:
    """The stories still to grade, in a stable order.

    *assessments* are production assessment records, one per thread (as
    ``newsletter_review.tui.load_assessments`` returns them). Only each story's
    text, its thread and the record's timestamp are read; nothing the model said
    about the story is carried into a ``Candidate``.

    Left out: threads in the hand-curated golden set (their stories are already
    ground truth), stories already labeled under their key, blank texts, and a
    text already queued. A story that overlaps one already *graded* in the same
    thread is left out as a near-duplicate; one that overlaps a *miscut* story
    in the same thread is that story's recut and comes back linked to it.

    Sorted by story key, which is a hash: the order survives restarts and one
    newsletter's stories are not graded back to back.
    """
    by_thread: dict[str, list[StoryLabel]] = defaultdict(list)
    for label in labels.values():
        by_thread[label.thread_id].append(label)

    queued: dict[str, Candidate] = {}
    for record in assessments:
        thread_id = record.get("thread_id") or ""
        if thread_id in curated_thread_ids:
            continue
        for story in record.get("stories", []):
            text = story.get("text") or ""
            if not text.strip():
                continue
            key = story_key(text)
            if key in labels or key in queued:
                continue
            recut_of = ""
            already_graded = False
            for label in by_thread.get(thread_id, ()):
                if text_overlap(text, label.text) < OVERLAP_THRESHOLD:
                    continue
                if label.status == "graded":
                    already_graded = True
                    break
                recut_of = recut_of or label.story_key
            if already_graded:
                continue
            queued[key] = Candidate(
                story_key=key,
                thread_id=thread_id,
                text=text,
                source_timestamp=record.get("timestamp") or "",
                recut_of=recut_of,
            )
    return sorted(queued.values(), key=lambda candidate: candidate.story_key)


def miscut_bucket(labels: dict[str, StoryLabel]) -> list[StoryLabel]:
    """Miscut stories still waiting for a usable recut, grouped by thread.

    A miscut story leaves the bucket when a story recorded as its recut has
    been graded, or when it is itself graded later (the newest label wins).
    """
    redressed = {
        label.recut_of for label in labels.values()
        if label.status == "graded" and label.recut_of
    }
    waiting = [
        label for label in labels.values()
        if label.status == "miscut" and label.story_key not in redressed
    ]
    return sorted(waiting, key=lambda label: (label.thread_id, label.labeled_at, label.story_key))
