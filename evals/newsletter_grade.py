"""Grade ready-made newsletter stories into an append-only label log.

The production daemon already extracts story texts from every newsletter it
grades (the assessments JSONL). This module turns those texts into a queue of
stories for the operator to grade by the rubric, blind to what the model said,
and records each judgment as one self-contained line in a label log.

Cutting and rating are separate jobs: a story whose boundaries are wrong is
marked *miscut* and set aside for recutting: it is not graded, and it stays in
the log with its text so it is not lost.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import tomllib
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Input, Static

from config_utils import substitute_env_vars
from daemon import load_config
from evals.newsletter_schemas import MISCUT_REASONS, StoryLabel
from newsletter import _DIMENSIONS, _VALID_THEMES
from newsletter_review.tui import load_assessments
from tui_common import truncate

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


def _parse_time(value) -> datetime | None:
    try:
        stamp = datetime.fromisoformat(value or "")
    except (TypeError, ValueError):
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def _later(record_timestamp, label_source_timestamp) -> bool:
    """Whether an assessment record is a later extraction than the one a label's
    story came from. False when either time is unknown: without knowing, no
    story is treated as another's recut or duplicate."""
    record_at, source_at = _parse_time(record_timestamp), _parse_time(label_source_timestamp)
    return record_at is not None and source_at is not None and record_at > source_at


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
    text already queued. When the thread has been extracted again since a story
    was labeled, a story of the newer extraction that overlaps one already
    *graded* is left out as a near-duplicate, and one that overlaps a *miscut*
    story is that story's recut and comes back linked to it. Stories from the
    same extraction as a labeled one are its neighbors and are queued plainly.

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
                # Only a LATER extraction can recut or duplicate a labeled story.
                # A story from the same extraction is its neighbor, however much
                # text the two share (a cut that ran long contains the next one).
                if not _later(record.get("timestamp"), label.source_timestamp):
                    continue
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


# The nine judgments asked about a story, in order: the four dimensions, then
# the five themes in the order the theme prompt lists them.
THEMES = ("scripture", "christlikeness", "church", "vocation_family", "disciple_making")
QUESTIONS = tuple(_DIMENSIONS) + THEMES


@dataclass(frozen=True)
class Card:
    """One story being graded: the answers so far, oldest first."""

    candidate: Candidate
    answers: tuple[int, ...] = ()  # 1/2/3 per answered question, in QUESTIONS order
    torn: frozenset[str] = frozenset()  # questions the operator was torn on
    note: str = ""


def current_question(card: Card) -> str | None:
    """The question the next answer goes to, or None once all nine are answered."""
    if len(card.answers) >= len(QUESTIONS):
        return None
    return QUESTIONS[len(card.answers)]


def answer(card: Card, value: int) -> Card:
    """Record 1/2/3 (low/middle/high) for the current question.

    Anything else, or an answer to a finished card, leaves the card unchanged.
    """
    if value not in (1, 2, 3) or current_question(card) is None:
        return card
    return replace(card, answers=card.answers + (value,))


def answer_rest_absent(card: Card) -> Card:
    """Answer Absent for every theme not yet answered.

    Only while a theme is being asked: a dimension has no "none" answer.
    """
    if current_question(card) not in THEMES:
        return card
    remaining = len(QUESTIONS) - len(card.answers)
    return replace(card, answers=card.answers + (1,) * remaining)


def back(card: Card) -> Card:
    """Take back the last answer. Torn flags stay with their question."""
    if not card.answers:
        return card
    return replace(card, answers=card.answers[:-1])


def toggle_torn(card: Card) -> Card:
    """Flag (or unflag) the current question as one the operator is torn on."""
    question = current_question(card)
    if question is None:
        return card
    return replace(card, torn=card.torn ^ {question})


_THEME_GRADE_BY_ANSWER = {2: "present", 3: "emphasized"}  # 1 = absent, stored by omission


def graded_label(card: Card, *, labeled_at: str, seconds: float | None) -> StoryLabel:
    """The label line for a fully answered card."""
    if current_question(card) is not None:
        raise ValueError(
            f"story {card.candidate.story_key!r} has unanswered questions "
            f"(next: {current_question(card)})"
        )
    by_question = dict(zip(QUESTIONS, card.answers))
    return StoryLabel(
        story_key=card.candidate.story_key,
        thread_id=card.candidate.thread_id,
        text=card.candidate.text,
        status="graded",
        scores={dim: by_question[dim] for dim in _DIMENSIONS},
        themes={
            theme: _THEME_GRADE_BY_ANSWER[by_question[theme]]
            for theme in THEMES if by_question[theme] in _THEME_GRADE_BY_ANSWER
        },
        torn=[question for question in QUESTIONS if question in card.torn],
        note=card.note,
        recut_of=card.candidate.recut_of,
        labeled_at=labeled_at,
        seconds=seconds,
        source_timestamp=card.candidate.source_timestamp,
    )


def miscut_label(card: Card, reason: str, note: str, *, labeled_at: str,
                 seconds: float | None) -> StoryLabel:
    """The label line that sets a story aside for recutting. Any answers already
    given on the card are dropped: a miscut story carries no grades."""
    return StoryLabel(
        story_key=card.candidate.story_key,
        thread_id=card.candidate.thread_id,
        text=card.candidate.text,
        status="miscut",
        note=note,
        miscut_reason=reason,
        recut_of=card.candidate.recut_of,
        labeled_at=labeled_at,
        seconds=seconds,
        source_timestamp=card.candidate.source_timestamp,
    )


_ANSWER_BY_THEME_GRADE = {"present": 2, "emphasized": 3}


def card_from_label(label: StoryLabel) -> Card:
    """Reopen a story from its label, for stepping back into it.

    A graded story reopens at its last question with the earlier answers, torn
    flags and note in place; a miscut story reopens as a fresh card.
    """
    candidate = Candidate(
        story_key=label.story_key,
        thread_id=label.thread_id,
        text=label.text,
        source_timestamp=label.source_timestamp,
        recut_of=label.recut_of,
    )
    if label.status != "graded":
        return Card(candidate)
    answers = tuple(label.scores[dim] for dim in _DIMENSIONS) + tuple(
        _ANSWER_BY_THEME_GRADE.get(label.themes.get(theme), 1) for theme in THEMES
    )
    return Card(candidate, answers[:-1], frozenset(label.torn), label.note)


_DIMENSION_GRADES = ((1, "Poor", "poor"), (2, "OK", "ok"), (3, "Good", "good"))
_THEME_GRADES = ((1, "Absent", "absent"), (2, "Present", "present"), (3, "Emphasized", "emphasized"))

# Rubric section 1, Decided 2026-09-09. Shown when a question is flagged torn.
GRADE_DOWN_RULE = "torn: when torn, grade down and say why"


def question_panel(card: Card, rubric: Rubric) -> list[str]:
    """The lines shown beside the story for the question being asked: its name,
    the rubric's wording for it, and the three grades with their anchors."""
    question = current_question(card)
    if question is None:
        return []
    if question in THEMES:
        wording, grades, anchors = rubric.themes[question], _THEME_GRADES, rubric.theme_grades
    else:
        wording, grades, anchors = rubric.dimensions[question], _DIMENSION_GRADES, rubric.grades
    lines = [f"{question.upper()}   {wording}", ""]
    lines += [f"  {key}  {name:<11}{anchors[anchor]}" for key, name, anchor in grades]
    if question in card.torn:
        lines += ["", f"  {GRADE_DOWN_RULE}"]
    return lines


def answers_strip(card: Card) -> str:
    """One line of the answers so far; ``?`` marks the question being asked."""
    parts = []
    for index, question in enumerate(QUESTIONS):
        names = _THEME_GRADES if question in THEMES else _DIMENSION_GRADES
        if index < len(card.answers):
            value = names[card.answers[index] - 1][1]
        elif index == len(card.answers):
            value = "?"
        else:
            value = "-"
        flag = " (torn)" if question in card.torn else ""
        parts.append(f"{question} {value}{flag}")
    return "   ".join(parts)


def count_labeled_on(labels: dict[str, StoryLabel], day, tz=None) -> int:
    """How many stories' current label was made on *day*, in timezone *tz*
    (the local timezone when None)."""
    count = 0
    for label in labels.values():
        stamp = _labeled_at(label)
        if stamp is not None and stamp.astimezone(tz).date() == day:
            count += 1
    return count


# What the operator can say is wrong with a cut, in key order (1-4).
MISCUT_REASON_TEXT = {
    "cut_short": "cut short: the story continues past where this ends",
    "runs_long": "runs past its end: it takes in text that is not this story",
    "several_stories": "several stories in one",
    "not_a_story": "not a story",
}

_KEYS = {
    "grade": "1/2/3 answer   0 rest absent   t torn   backspace back   n note   "
             "x miscut   s skip   q quit",
    "reason": "1-4 choose   esc cancel",
    "note": "enter save   esc cancel",
    "done": "backspace reopen the last story   q quit",
}

_NOTE_PROMPTS = {
    "note": "Note for this story:",
    "torn": "Why (you were torn: grade down and say why):",
    "miscut": "What is wrong with the cut (optional):",
}


def miscut_menu() -> list[str]:
    """The lines shown in place of the question while a miscut reason is chosen."""
    lines = ["MISCUT   What is wrong with this cut? It will be set aside for recutting.", ""]
    lines += [
        f"  {key}  {MISCUT_REASON_TEXT[reason]}"
        for key, reason in enumerate(MISCUT_REASONS, 1)
    ]
    return lines


def _displayable(text: str) -> str:
    """Story text as drawn on screen: line breaks kept, tabs widened, every other
    control character dropped. The label still stores the text as extracted."""
    return "".join(
        "    " if ch == "\t" else ch
        for ch in text if ch in "\n\t" or ch.isprintable()
    )


class GradeApp(App):
    """One story at a time: the text, the question being asked with the rubric's
    wording for it, and nine single-key answers.

    Nothing here is modal over the story, and scrolling never touches the
    answers. While a note is being typed the arrow and page keys still scroll
    the story; Home and End then move within the note.
    Every finished card is appended to the label log before the next story is
    shown; a card that cannot be saved stays on screen.
    """

    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        Binding("1", "digit(1)", show=False),
        Binding("2", "digit(2)", show=False),
        Binding("3", "digit(3)", show=False),
        Binding("4", "digit(4)", show=False),
        Binding("0", "rest_absent", show=False),
        Binding("t", "torn", show=False),
        Binding("backspace", "back", show=False),
        Binding("n", "note", show=False),
        Binding("x", "miscut", show=False),
        Binding("s", "skip", show=False),
        Binding("q", "quit_app", show=False),
        Binding("escape", "cancel", show=False),
        Binding("up", "scroll_story('up')", show=False),
        Binding("down", "scroll_story('down')", show=False),
        Binding("pageup,ctrl+b", "scroll_story('page_up')", show=False),
        Binding("pagedown,ctrl+f", "scroll_story('page_down')", show=False),
        Binding("home", "scroll_story('home')", show=False),
        Binding("end", "scroll_story('end')", show=False),
    ]

    DEFAULT_CSS = """
    GradeApp #progress { text-style: bold; }
    GradeApp #story { height: 1fr; border-top: solid $accent; border-bottom: solid $accent; }
    GradeApp #story-text { max-width: 88; padding: 0 1; }
    GradeApp #question { height: auto; padding: 0 1; }
    GradeApp #strip { padding: 0 1; color: $text-muted; }
    GradeApp #status { padding: 0 1; }
    GradeApp #note-prompt { display: none; padding: 0 1; }
    GradeApp #note { display: none; }
    """

    def __init__(self, queue, rubric: Rubric, label_path, *, labeled_today: int = 0,
                 clock=time.monotonic, now=None):
        super().__init__()
        self.queue: list[Candidate] = list(queue)
        self.rubric = rubric
        self.label_path = Path(label_path)
        self.labeled_today = labeled_today
        self.session_labeled = 0
        self._clock = clock
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._done: list[StoryLabel] = []  # labels written this session, oldest first
        self._card: Card | None = None
        self._opened_at = 0.0
        self._mode = "done"  # grade | reason | note | done
        self._note_purpose = "note"
        self._miscut_reason = ""
        self._status = ""

    def compose(self) -> ComposeResult:
        yield Static(id="progress", markup=False)
        yield VerticalScroll(Static(id="story-text", markup=False), id="story")
        yield Static(id="question", markup=False)
        yield Static(id="strip", markup=False)
        yield Static(id="status", markup=False)
        yield Static(id="note-prompt", markup=False)
        yield Input(id="note")

    def on_mount(self) -> None:
        self._open_next()

    # -- state ---------------------------------------------------------------

    def _open_next(self) -> None:
        if self.queue:
            self._open(Card(self.queue[0]))
        else:
            self._card = None
            self._mode = "done"
            self._refresh()

    def _open(self, card: Card) -> None:
        self._card = card
        self._mode = "grade"
        self._opened_at = self._clock()
        self.query_one("#story-text", Static).update(_displayable(card.candidate.text))
        self.query_one("#story", VerticalScroll).scroll_home(animate=False)
        self._refresh()

    def _refresh(self) -> None:
        card = self._card
        progress = f"{self.labeled_today} labeled today · {len(self.queue)} queued"
        if card is not None and card.candidate.recut_of:
            progress += " · a recut of a story you marked miscut"
        self.query_one("#progress", Static).update(progress)
        if self._mode == "done":
            self.query_one("#story-text", Static).update("")
            question = [f"Nothing left to grade. {self.session_labeled} labeled this session."]
            strip = ""
        elif self._mode == "reason":
            question, strip = miscut_menu(), answers_strip(card)
        else:
            question, strip = question_panel(card, self.rubric), answers_strip(card)
        self.query_one("#question", Static).update("\n".join(question))
        self.query_one("#strip", Static).update(strip)
        status = _KEYS[self._mode]
        if self._status:
            status = f"{self._status}\n{status}"
        self.query_one("#status", Static).update(status)
        editing = self._mode == "note"
        prompt = self.query_one("#note-prompt", Static)
        prompt.update(_NOTE_PROMPTS[self._note_purpose])
        prompt.display = editing
        self.query_one("#note", Input).display = editing

    def _seconds(self) -> float:
        return round(self._clock() - self._opened_at, 1)

    def _stamp(self) -> str:
        return self._now().isoformat()

    def _commit(self, label: StoryLabel) -> None:
        """Append *label*; only then leave the story. A failed write keeps it."""
        try:
            append_label(self.label_path, label)
        except OSError as exc:
            self._status = f"Could not save: {exc}. Nothing was written; the story stays here."
            if current_question(self._card) is None:
                self._card = back(self._card)
            self._mode = "grade"
            self._refresh()
            return
        self._status = ""
        self._done.append(label)
        self.queue.pop(0)
        self.labeled_today += 1
        self.session_labeled += 1
        self._open_next()

    def _finish(self) -> None:
        """All nine answered: write the label, asking why first if torn."""
        if self._card.torn and not self._card.note:
            self._ask_note("torn")
        else:
            self._commit(graded_label(
                self._card, labeled_at=self._stamp(), seconds=self._seconds()))

    def _ask_note(self, purpose: str) -> None:
        self._note_purpose = purpose
        self._mode = "note"
        note = self.query_one("#note", Input)
        note.value = "" if purpose == "miscut" else self._card.note
        self._refresh()
        note.focus()

    # -- keys ----------------------------------------------------------------

    def action_digit(self, value: int) -> None:
        if self._mode == "grade" and value in (1, 2, 3):
            self._status = ""
            self._card = answer(self._card, value)
            if current_question(self._card) is None:
                self._finish()
            else:
                self._refresh()
        elif self._mode == "reason":
            self._miscut_reason = MISCUT_REASONS[value - 1]
            self._ask_note("miscut")

    def action_rest_absent(self) -> None:
        if self._mode != "grade":
            return
        filled = answer_rest_absent(self._card)
        if filled != self._card:
            self._card = filled
            self._finish()

    def action_torn(self) -> None:
        if self._mode == "grade":
            self._card = toggle_torn(self._card)
            self._refresh()

    def action_back(self) -> None:
        if self._mode == "grade" and self._card.answers:
            self._card = back(self._card)
            self._refresh()
        elif self._mode in ("grade", "done") and self._done:
            # Step back into the story labeled just before. Its earlier label
            # stays in the log; finishing it again appends one that supersedes.
            card = card_from_label(self._done.pop())
            self.queue.insert(0, card.candidate)
            self.labeled_today -= 1
            self.session_labeled -= 1
            self._open(card)

    def action_note(self) -> None:
        if self._mode == "grade":
            self._ask_note("note")

    def action_miscut(self) -> None:
        if self._mode == "grade":
            self._mode = "reason"
            self._refresh()

    def action_skip(self) -> None:
        if self._mode == "grade":
            self.queue.append(self.queue.pop(0))
            self._open_next()

    def action_cancel(self) -> None:
        if self._mode == "note":
            if self._note_purpose == "torn":
                self._card = back(self._card)  # back to the last question
            self._mode = "grade"
            self.set_focus(None)
            self._refresh()
        elif self._mode == "reason":
            self._mode = "grade"
            self._refresh()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self._mode != "note":
            return
        text = "".join(ch for ch in event.value if ch.isprintable()).strip()
        self.set_focus(None)
        if self._note_purpose == "miscut":
            self._commit(miscut_label(
                self._card, self._miscut_reason, text,
                labeled_at=self._stamp(), seconds=self._seconds()))
            return
        self._card = replace(self._card, note=text)
        if self._note_purpose == "torn":
            self._commit(graded_label(
                self._card, labeled_at=self._stamp(), seconds=self._seconds()))
        else:
            self._mode = "grade"
            self._refresh()

    def action_scroll_story(self, how: str) -> None:
        story = self.query_one("#story", VerticalScroll)
        getattr(story, f"scroll_{how}")(animate=False)

    def action_quit_app(self) -> None:
        self.exit(self.session_labeled)


def _fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)


def _curated_thread_ids(path: Path) -> set[str]:
    """Thread ids in the hand-curated golden set; none if the file is absent."""
    if not path.exists():
        return set()
    ids = set()
    with open(path) as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line).get("thread_id", ""))
    return ids


def _printable(text: str) -> str:
    """*text* without control characters. Story text comes from email and is
    printed to a terminal here, where an escape sequence would be obeyed."""
    return "".join(ch for ch in text if ch.isprintable())


def format_miscut_bucket(bucket: list[StoryLabel]) -> str:
    """The miscut bucket as text: each thread, then its stories with the reason,
    the opening words and the operator's note."""
    if not bucket:
        return "No miscut stories."
    count = len(bucket)
    lines = [f"{count} miscut {'story' if count == 1 else 'stories'} waiting for a recut."]
    thread = None
    for label in bucket:
        if label.thread_id != thread:
            thread = label.thread_id
            lines += ["", _printable(thread)]
        reason = MISCUT_REASON_TEXT[label.miscut_reason].split(":")[0]
        opening = truncate(_printable(" ".join(label.text.split())), 70)
        lines.append(f"  {reason}: {opening}")
        if label.note:
            lines.append(f"    note: {_printable(label.note)}")
    return "\n".join(lines)


def cli(argv=None):
    parser = argparse.ArgumentParser(
        description="Grade ready-made newsletter stories, blind, into the story-label log")
    parser.add_argument("--assessments",
                        help="Production assessments JSONL to draw stories from "
                             "(default: [newsletter] output_file in the config)")
    parser.add_argument("--labels", default="evals/newsletter_story_labels.jsonl",
                        help="Story-label log to append to")
    parser.add_argument("--golden-set", default="evals/newsletter_golden_set.jsonl",
                        help="Hand-curated golden set whose newsletters are left out of "
                             "the queue (a missing file means none)")
    parser.add_argument("--config",
                        help="Path to config.toml (default: the repo-root config.toml, "
                             "regardless of CWD)")
    parser.add_argument("--miscut", action="store_true",
                        help="List the stories set aside as miscut, then exit")
    args = parser.parse_args(argv)

    labels_path = Path(args.labels)
    try:
        labels = latest_labels(read_label_log(labels_path))
    except ValueError as exc:
        _fail(f"Cannot read the story-label log: {exc}")
    if args.miscut:
        print(format_miscut_bucket(miscut_bucket(labels)))
        return

    try:
        if args.config:
            with open(args.config, "rb") as f:
                config = substitute_env_vars(tomllib.load(f))
        else:
            config = load_config()
        rubric = load_rubric(config)
    except (OSError, ValueError) as exc:
        _fail(f"Cannot load the config: {exc}")

    assessments_path = Path(args.assessments or config["newsletter"]["output_file"])
    if not assessments_path.exists():
        _fail(f"No assessments file at {assessments_path}. Pass --assessments PATH, "
              "or run this where the daemon writes its newsletter assessments.")
    try:
        assessments = load_assessments(assessments_path)
        curated = _curated_thread_ids(Path(args.golden_set))
    except ValueError as exc:
        _fail(f"Cannot read the stories to grade: {exc}")

    queue = build_queue(assessments, curated, labels)
    if not queue:
        print(f"Nothing to grade: every story in {assessments_path} is already labeled "
              "or belongs to a curated newsletter.")
        sys.exit(0)

    app = GradeApp(queue, rubric, labels_path,
                   labeled_today=count_labeled_on(labels, date.today()))
    labeled = app.run()
    if labeled is not None:
        print(f"{labeled} labeled this session. Labels are in {labels_path}.")


if __name__ == "__main__":
    cli()
