"""Dataclasses for the newsletter golden set and evaluation results.

All types support JSONL serialization via to_dict() / from_dict().
from_dict is tolerant of missing optional keys (via d.get(...) defaults) so
older golden sets and result files keep loading as the schema grows.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from newsletter import _DIMENSIONS, _VALID_THEMES, compute_tier

# Grades that ever appear in a stored theme dict. Kept in lockstep with
# newsletter.parse_themes (which only stores present/emphasized; absent is
# omitted) and evals.newsletter_label._THEME_CYCLE / _GRADE_ABBR (which only
# cycle through / abbreviate exactly these). "absent" and case variants are
# NOT valid stored grades.
_VALID_THEME_GRADES = frozenset({"present", "emphasized"})

# Valid dimension scores after the 1-3 (Poor/OK/Good) migration; legacy 1-5
# exports are out of range and must be re-migrated (issue #53).
_VALID_SCORES = frozenset({1, 2, 3})


def _coerce_themes(value, *, story_id: str = "", field_name: str = "themes") -> dict[str, str]:
    """Normalize a stored themes value to a grade dict (theme -> "present"/
    "emphasized"), defaulting a missing/empty value to ``{}`` (issue #53).

    Backward compatibility with the pre-#53 schema was deliberately removed, so
    this raises a clear, actionable ValueError (rather than silently coercing or
    dying with an opaque dict-update error) when it meets old-scheme data:

    * ``value`` is a list — the pre-migration theme scheme stored themes as a
      list of names; the file must be re-migrated to the theme->grade dict
      scheme (Finding 1).
    * a grade is anything other than exactly "present"/"emphasized" — a case
      variant, "absent", or junk would later KeyError the labeling TUI mid
      session (Finding 2).

    ``story_id``/``field_name`` are woven into the message so the offending
    record and field are locatable.
    """
    where = f"{field_name} for story {story_id!r}" if story_id else field_name
    if isinstance(value, list):
        raise ValueError(
            f"{where} is an old-scheme list {value!r}; this file predates the "
            "theme-dict migration (issue #53) and must be re-migrated to the new "
            "theme->grade dict scheme (theme -> 'present'/'emphasized')."
        )
    if not value:
        return {}
    coerced = dict(value)
    for theme, grade in coerced.items():
        if grade not in _VALID_THEME_GRADES:
            raise ValueError(
                f"{where}: theme {theme!r} has invalid grade {grade!r}; expected "
                "exactly 'present' or 'emphasized' ('absent' is represented by "
                "omitting the theme). Re-migrate this file to the new theme-grade "
                "scheme (issue #53)."
            )
    return coerced


def _coerce_scores(value, *, story_id: str = "", field_name: str = "scores"):
    """Validate a stored dimension-score dict: every value must be an int in
    1..3 (Poor/OK/Good) after the #53 migration. A missing/None value is passed
    through unchanged (unlabeled). Legacy 1-5 scores are rejected loudly at load
    rather than silently corrupting report metrics / the labeling TUI (Finding
    3). ``story_id``/``field_name`` locate the offending record and field."""
    if value is None:
        return None
    where = f"{field_name} for story {story_id!r}" if story_id else field_name
    for dimension, score in value.items():
        if isinstance(score, bool) or not isinstance(score, int) or score not in _VALID_SCORES:
            raise ValueError(
                f"{where}: dimension {dimension!r} has invalid value {score!r}; "
                "expected an int in 1..3 (Poor/OK/Good). This file may be an "
                "old-scheme (1-5) export needing re-migration to the 1-3 scale "
                "(issue #53)."
            )
    return value


@dataclass
class GoldenStory:
    """One story within a golden newsletter (ground truth)."""

    story_id: str  # stable, f"{thread_id}:{index}"
    text: str
    expected_scores: dict[str, int] | None = None  # simple/concrete/personal/dynamic, 1-3 (Poor/OK/Good)
    expected_tier: str | None = None  # "excellent" / "good" / "fair" / "poor"
    expected_themes: dict[str, str] = field(default_factory=dict)  # theme -> "present"/"emphasized"
    reviewed: bool = False
    notes: str = ""
    excluded: bool = False  # drop from quality/theme scoring, keep as extraction truth

    def to_dict(self) -> dict:
        return {
            "story_id": self.story_id,
            "text": self.text,
            "expected_scores": self.expected_scores,
            "expected_tier": self.expected_tier,
            "expected_themes": self.expected_themes,
            "reviewed": self.reviewed,
            "notes": self.notes,
            "excluded": self.excluded,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GoldenStory":
        # ``title`` was dropped from the schema; older golden sets that still
        # carry it load fine because the extra key is simply ignored here.
        sid = d.get("story_id", "")
        return cls(
            story_id=d["story_id"],
            text=d["text"],
            expected_scores=_coerce_scores(
                d.get("expected_scores"), story_id=sid, field_name="expected_scores"
            ),
            expected_tier=d.get("expected_tier"),
            expected_themes=_coerce_themes(
                d.get("expected_themes"), story_id=sid, field_name="expected_themes"
            ),
            reviewed=d.get("reviewed", False),
            notes=d.get("notes", ""),
            excluded=d.get("excluded", False),
        )


# A story-label line's status. "graded" carries ground truth; "miscut" sets the
# story aside for recutting and carries none.
STORY_LABEL_STATUSES = ("graded", "miscut")

# What the operator said is wrong with a miscut story. "not_a_story" is the one
# reason no recut can fix; the others describe a boundary to redraw.
MISCUT_REASONS = ("cut_short", "runs_long", "several_stories", "not_a_story")

_THEME_NAMES = frozenset(name.lower() for name in _VALID_THEMES)


@dataclass
class StoryLabel:
    """One line of the append-only story-label log written by
    ``evals.newsletter_grade``: the operator's judgment of one ready-made story.

    Self-contained on purpose (it carries the story text), so the log is a
    grading golden set by itself and two machines' logs merge by concatenation.
    A later line for the same ``story_key`` supersedes an earlier one.
    Validated on load, theme names included, because the file is written by a
    tool and merged by hand.
    """

    story_key: str  # hash of the normalized story text
    thread_id: str
    text: str
    status: str  # one of STORY_LABEL_STATUSES
    scores: dict[str, int] | None = None  # all four dimensions, 1-3, when graded
    themes: dict[str, str] = field(default_factory=dict)  # theme -> "present"/"emphasized"
    torn: list[str] = field(default_factory=list)  # dimensions/themes the operator was torn on
    note: str = ""
    miscut_reason: str = ""  # one of MISCUT_REASONS when status == "miscut"
    recut_of: str = ""  # story_key of the miscut story this one replaces
    labeled_at: str = ""  # ISO-8601 UTC
    seconds: float | None = None  # time the card was open
    source_timestamp: str = ""  # the assessment record the text came from

    def to_dict(self) -> dict:
        return {
            "type": "story_label",
            "story_key": self.story_key,
            "thread_id": self.thread_id,
            "text": self.text,
            "status": self.status,
            "scores": self.scores,
            "themes": self.themes,
            "torn": self.torn,
            "note": self.note,
            "miscut_reason": self.miscut_reason,
            "recut_of": self.recut_of,
            "labeled_at": self.labeled_at,
            "seconds": self.seconds,
            "source_timestamp": self.source_timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StoryLabel":
        """Build a label from one decoded log line, validating all of it.

        Every problem is a ValueError that names the field, so the reader can
        report the file and line: the log is merged by hand, and a malformed
        line must not surface later as a crash or as a silently wrong score.
        """
        if not isinstance(d, dict):
            raise ValueError("a story-label line must be a JSON object")
        missing = [name for name in ("story_key", "thread_id", "text", "status") if name not in d]
        if missing:
            raise ValueError(f"story label is missing {', '.join(missing)}")
        key = d["story_key"]

        def text_field(name: str, default: str = "", required: bool = False) -> str:
            value = d.get(name, default)
            if not isinstance(value, str) or (required and not value.strip()):
                kind = "a non-empty string" if required else "a string"
                raise ValueError(f"story label {key!r}: {name} must be {kind}, got {value!r}")
            return value

        text_field("story_key", required=True)
        thread_id = text_field("thread_id")
        text = text_field("text", required=True)
        status = d["status"]
        if status not in STORY_LABEL_STATUSES:
            raise ValueError(
                f"story label {key!r}: status {status!r} is not one of "
                f"{', '.join(STORY_LABEL_STATUSES)}"
            )
        raw_scores = d.get("scores")
        if raw_scores is not None and not isinstance(raw_scores, dict):
            raise ValueError(f"scores for story {key!r} must be a mapping, got {raw_scores!r}")
        scores = _coerce_scores(raw_scores, story_id=key, field_name="scores")
        raw_themes = d.get("themes")
        if raw_themes is not None and not isinstance(raw_themes, (dict, list)):
            raise ValueError(f"themes for story {key!r} must be a mapping, got {raw_themes!r}")
        themes = _coerce_themes(raw_themes, story_id=key, field_name="themes")
        unknown = sorted(set(themes) - _THEME_NAMES)
        if unknown:
            raise ValueError(
                f"themes for story {key!r}: unknown theme name(s) {', '.join(unknown)}; "
                f"expected only {', '.join(sorted(_THEME_NAMES))}"
            )
        torn = d.get("torn") or []
        if not isinstance(torn, list):
            raise ValueError(f"torn for story {key!r} must be a list, got {torn!r}")
        stray = sorted({str(name) for name in torn} - _THEME_NAMES - set(_DIMENSIONS))
        if stray:
            raise ValueError(
                f"torn for story {key!r}: {', '.join(stray)} is not a dimension or a theme"
            )
        seconds = d.get("seconds")
        if seconds is not None and (isinstance(seconds, bool) or not isinstance(seconds, (int, float))):
            raise ValueError(f"story label {key!r}: seconds must be a number, got {seconds!r}")
        reason = text_field("miscut_reason")
        if status == "graded":
            if scores is None:
                raise ValueError(f"story label {key!r}: a graded label needs scores")
            missing_dims = [dim for dim in _DIMENSIONS if dim not in scores]
            if missing_dims:
                raise ValueError(
                    f"scores for story {key!r}: missing dimension(s) {', '.join(missing_dims)}"
                )
            extra = sorted(set(scores) - set(_DIMENSIONS))
            if extra:
                raise ValueError(
                    f"scores for story {key!r}: {', '.join(extra)} is not a dimension"
                )
        else:
            if reason not in MISCUT_REASONS:
                raise ValueError(
                    f"story label {key!r}: miscut_reason {reason!r} is not one of "
                    f"{', '.join(MISCUT_REASONS)}"
                )
            if scores is not None or themes:
                raise ValueError(
                    f"story label {key!r}: a miscut story carries no scores or themes"
                )
        return cls(
            story_key=key,
            thread_id=thread_id,
            text=text,
            status=status,
            scores=scores,
            themes=themes,
            torn=list(torn),
            note=text_field("note"),
            miscut_reason=reason,
            recut_of=text_field("recut_of"),
            labeled_at=text_field("labeled_at"),
            seconds=seconds,
            source_timestamp=text_field("source_timestamp"),
        )

    def to_golden_story(self) -> "GoldenStory":
        """This label as grading ground truth, in the shape the eval runs on."""
        if self.status != "graded":
            raise ValueError(
                f"story label {self.story_key!r} is {self.status}, not graded: "
                "a miscut story has no ground truth"
            )
        return GoldenStory(
            story_id=self.story_key,
            text=self.text,
            expected_scores=dict(self.scores),
            expected_tier=compute_tier(self.scores).value,
            expected_themes=dict(self.themes),
            reviewed=True,
            notes=self.note,
        )


@dataclass
class GoldenNewsletter:
    """One newsletter in the golden set; owns its list of GoldenStory."""

    thread_id: str
    message_id: str
    sender: str
    subject: str
    body: str  # raw body fed verbatim to extract_stories
    stories: list[GoldenStory] = field(default_factory=list)
    source: str = "harvested"
    harvested_at: str = ""  # ISO 8601
    reviewed: bool = False  # story list confirmed = authoritative extraction truth
    notes: str = ""
    excluded: bool = False

    def to_dict(self) -> dict:
        return {
            "type": "golden_newsletter",
            "thread_id": self.thread_id,
            "message_id": self.message_id,
            "sender": self.sender,
            "subject": self.subject,
            "body": self.body,
            "stories": [s.to_dict() for s in self.stories],
            "source": self.source,
            "harvested_at": self.harvested_at,
            "reviewed": self.reviewed,
            "notes": self.notes,
            "excluded": self.excluded,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GoldenNewsletter":
        return cls(
            thread_id=d["thread_id"],
            message_id=d["message_id"],
            sender=d["sender"],
            subject=d["subject"],
            body=d["body"],
            stories=[GoldenStory.from_dict(s) for s in d.get("stories", [])],
            source=d.get("source", "harvested"),
            harvested_at=d.get("harvested_at", ""),
            reviewed=d.get("reviewed", False),
            notes=d.get("notes", ""),
            excluded=d.get("excluded", False),
        )


@dataclass
class StoryPrediction:
    """One prediction per golden story in quality/theme mode."""

    story_id: str
    thread_id: str
    expected_scores: dict[str, int] | None = None
    expected_tier: str | None = None
    expected_themes: dict[str, str] = field(default_factory=dict)
    predicted_scores: dict[str, int] | None = None
    predicted_tier: str | None = None
    predicted_themes: dict[str, str] = field(default_factory=dict)
    scores_raw: str | None = None
    themes_raw: str | None = None
    duration_seconds: float = 0.0
    error: str | None = None
    # Where the ground truth came from: "curated" (the hand-curated golden set)
    # or "story_labels" (the story-label log written by newsletter_grade).
    source: str = "curated"

    def to_dict(self) -> dict:
        return {
            "type": "story_prediction",
            "story_id": self.story_id,
            "thread_id": self.thread_id,
            "expected_scores": self.expected_scores,
            "expected_tier": self.expected_tier,
            "expected_themes": self.expected_themes,
            "predicted_scores": self.predicted_scores,
            "predicted_tier": self.predicted_tier,
            "predicted_themes": self.predicted_themes,
            "scores_raw": self.scores_raw,
            "themes_raw": self.themes_raw,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StoryPrediction":
        sid = d.get("story_id", "")
        return cls(
            story_id=d["story_id"],
            thread_id=d["thread_id"],
            expected_scores=_coerce_scores(
                d.get("expected_scores"), story_id=sid, field_name="expected_scores"
            ),
            expected_tier=d.get("expected_tier"),
            expected_themes=_coerce_themes(
                d.get("expected_themes"), story_id=sid, field_name="expected_themes"
            ),
            predicted_scores=_coerce_scores(
                d.get("predicted_scores"), story_id=sid, field_name="predicted_scores"
            ),
            predicted_tier=d.get("predicted_tier"),
            predicted_themes=_coerce_themes(
                d.get("predicted_themes"), story_id=sid, field_name="predicted_themes"
            ),
            scores_raw=d.get("scores_raw"),
            themes_raw=d.get("themes_raw"),
            duration_seconds=d.get("duration_seconds", 0.0),
            error=d.get("error"),
            source=d.get("source", "curated"),
        )


@dataclass
class ExtractionPrediction:
    """One prediction per newsletter in extraction mode."""

    thread_id: str
    golden_stories: list[dict] = field(default_factory=list)  # [{"story_id","text"}]
    predicted_stories: list[dict] = field(default_factory=list)  # [{"text"}]
    duration_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "type": "extraction_prediction",
            "thread_id": self.thread_id,
            "golden_stories": self.golden_stories,
            "predicted_stories": self.predicted_stories,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExtractionPrediction":
        return cls(
            thread_id=d["thread_id"],
            golden_stories=d.get("golden_stories", []),
            predicted_stories=d.get("predicted_stories", []),
            duration_seconds=d.get("duration_seconds", 0.0),
            error=d.get("error"),
        )


@dataclass
class NewsletterThinkingEntry:
    """Chain-of-thought content for one eval-run unit of work.

    Story-level entries (quality/theme scoring) set story_id; newsletter-level
    entries (extraction segmentation) set thread_id + extraction_cot instead.
    Stored in sidecar .cot.jsonl files alongside main results.
    """

    story_id: str = ""
    quality_cot: str = ""
    theme_cot: str = ""
    thread_id: str = ""
    extraction_cot: str = ""

    def to_dict(self) -> dict:
        return {
            "type": "thinking",
            "story_id": self.story_id,
            "quality_cot": self.quality_cot,
            "theme_cot": self.theme_cot,
            "thread_id": self.thread_id,
            "extraction_cot": self.extraction_cot,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NewsletterThinkingEntry":
        return cls(
            story_id=d.get("story_id", ""),
            quality_cot=d.get("quality_cot", ""),
            theme_cot=d.get("theme_cot", ""),
            thread_id=d.get("thread_id", ""),
            extraction_cot=d.get("extraction_cot", ""),
        )


@dataclass
class NewsletterRunMeta:
    """Metadata for a newsletter evaluation run (first line of results JSONL)."""

    run_id: str
    timestamp: str  # ISO 8601
    config_hash: str
    config_path: str
    newsletter_model: str
    golden_set_path: str
    golden_set_count: int  # number of newsletters evaluated
    # Stories the run evaluated: Phase-B-labeled, non-excluded stories for story
    # modes (quality/themes/all); ALL golden stories for extraction-only runs,
    # since extraction matches against the full confirmed story list.
    story_count: int
    mode: str = "all"  # "extraction" / "quality" / "themes" / "all"
    prompt_hash: str = ""
    temperature: float = 0.0
    max_tokens: int = 0
    extra_body: dict | None = None
    parallelism: int = 1
    tag: str = ""
    # System prompts (constant across a run, deterministic from config)
    extraction_system_prompt: str = ""
    quality_system_prompt: str = ""
    theme_system_prompt: str = ""
    # Stories taken from the story-label log (newsletter_grade); they are part
    # of story_count. Zero and "" for runs that used none.
    story_labels_path: str = ""
    story_label_count: int = 0

    def to_dict(self) -> dict:
        return {
            "type": "run_meta",
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "config_hash": self.config_hash,
            "config_path": self.config_path,
            "newsletter_model": self.newsletter_model,
            "golden_set_path": self.golden_set_path,
            "golden_set_count": self.golden_set_count,
            "story_count": self.story_count,
            "mode": self.mode,
            "prompt_hash": self.prompt_hash,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "extra_body": self.extra_body,
            "parallelism": self.parallelism,
            "tag": self.tag,
            "extraction_system_prompt": self.extraction_system_prompt,
            "quality_system_prompt": self.quality_system_prompt,
            "theme_system_prompt": self.theme_system_prompt,
            "story_labels_path": self.story_labels_path,
            "story_label_count": self.story_label_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NewsletterRunMeta":
        return cls(
            run_id=d["run_id"],
            timestamp=d["timestamp"],
            config_hash=d["config_hash"],
            config_path=d["config_path"],
            newsletter_model=d["newsletter_model"],
            golden_set_path=d["golden_set_path"],
            golden_set_count=d["golden_set_count"],
            story_count=d["story_count"],
            mode=d.get("mode", "all"),
            prompt_hash=d.get("prompt_hash", ""),
            temperature=d.get("temperature", 0.0),
            max_tokens=d.get("max_tokens", 0),
            extra_body=d.get("extra_body"),
            parallelism=d.get("parallelism", 1),
            tag=d.get("tag", ""),
            extraction_system_prompt=d.get("extraction_system_prompt", ""),
            quality_system_prompt=d.get("quality_system_prompt", ""),
            theme_system_prompt=d.get("theme_system_prompt", ""),
            story_labels_path=d.get("story_labels_path", ""),
            story_label_count=d.get("story_label_count", 0),
        )


# --- The story-label log -----------------------------------------------------
# File handling for the log lives here, beside StoryLabel, so the headless eval
# runner can read it without loading the grading screen.


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
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
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


def parse_time(value) -> datetime | None:
    """An ISO-8601 time as an aware datetime (naive is read as UTC), or None
    when *value* is missing or unreadable."""
    try:
        stamp = datetime.fromisoformat(value or "")
    except (TypeError, ValueError):
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
            new_at, kept_at = parse_time(label.labeled_at), parse_time(kept.labeled_at)
            if new_at is not None and kept_at is not None and new_at < kept_at:
                continue
        latest[label.story_key] = label
    return latest
