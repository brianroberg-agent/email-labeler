"""Newsletter story classification pipeline.

Extracts stories from ministry newsletters, scores them on a quality rubric,
and tags them with Ends Statement themes. All LLM calls use the cloud endpoint
(newsletter content is not privacy-sensitive).
"""

import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import getaddresses, parsedate_to_datetime
from enum import Enum
from pathlib import Path

from gmail_utils import get_header
from llm_client import LLMBalanceError, LLMClient, LLMContentError, LLMUnavailableError

log = logging.getLogger(__name__)

# Errors that affect every story, not just the one being graded: propagate out
# of the per-story isolation handlers (to the daemon's cycle attribution, or to
# the newsletter function's halt) instead of committing a permanently mis-graded
# newsletter. Single-sourced so the quality and theme arms can't drift apart.
_PIPELINE_WIDE_ERRORS = (LLMUnavailableError, LLMContentError, LLMBalanceError)


class NewsletterTier(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


_VALID_THEMES = {
    "SCRIPTURE",
    "CHRISTLIKENESS",
    "CHURCH",
    "VOCATION_FAMILY",
    "DISCIPLE_MAKING",
}

_DIMENSIONS = ("simple", "concrete", "personal", "dynamic")

# Storytelling dimensions are graded on a 3-value rubric (issue #53). The tokens
# map to integers so the tier average and eval metrics stay simple arithmetic.
_SCORE_TOKENS = {"POOR": 1, "OK": 2, "GOOD": 3}

# Ends-Statement themes are graded on a 3-value rubric (issue #53). All three
# grades are *recognized*; "absent" is represented by omission, so only present/
# emphasized ever appear in a stored dict (see classify_theme_line / parse_themes).
_THEME_GRADE_TOKENS = {"ABSENT": "absent", "PRESENT": "present", "EMPHASIZED": "emphasized"}

# One-per-line theme matcher, single-sourced here so the production parser
# (parse_themes) and the eval anomaly detectors share identical format knowledge
# and cannot drift (Finding 2). Unanchored ``search`` tolerates leading list
# markers/markdown (mirrors parse_quality_scores); ``([A-Za-z]+)`` captures ANY
# grade word so a misspelled grade is still recognized as a line yet rejected as
# a grade — surfacing it as a near-miss instead of being silently skipped.
_THEME_LINE_RE = re.compile(r"([A-Za-z_]+)\s*:\s*([A-Za-z]+)")


def classify_theme_line(line: str) -> tuple[str, str | None, str | None]:
    """Classify one line of an LLM theme-grading reply. Returns ``(kind, name, grade)``.

    The single shared theme-line classifier: ``parse_themes`` and the eval
    anomaly detectors (``newsletter_run._theme_line_anomalies`` /
    ``newsletter_report.theme_parse_anomalies``) all route through it, so the
    response-format knowledge lives in one place (Finding 2).

    Kinds:
      - ``"blank"`` — whitespace-only line; ignore it.
      - ``"theme"`` — an in-taxonomy grading. ``name`` is the lowercased theme,
        ``grade`` one of "absent"/"present"/"emphasized". parse_themes stores it
        iff the grade is not "absent".
      - ``"off_taxonomy"`` — a ``NAME: GRADE`` line with a valid grade token but an
        off-taxonomy ``name`` (returned UPPERCASE); parse_themes drops it.
      - ``"anomalous"`` — a non-blank line that is not a recognizable grading (no
        ``NAME: WORD`` pair, or a WORD that is not a grade token); parse_themes
        skips it. Off-taxonomy and anomalous lines are exactly the non-blank lines
        parse_themes does NOT accept — the diagnostics surface both (Finding 3).
    """
    if not line.strip():
        return "blank", None, None
    match = _THEME_LINE_RE.search(line)
    if not match:
        return "anomalous", None, None
    name = match.group(1).upper()
    grade = _THEME_GRADE_TOKENS.get(match.group(2).upper())
    if grade is None:
        return "anomalous", None, None
    if name in _VALID_THEMES:
        return "theme", name.lower(), grade
    return "off_taxonomy", name, grade


@dataclass
class StoryResult:
    text: str
    scores: dict[str, int] | None = None
    average_score: float | None = None
    tier: NewsletterTier | None = None
    themes: dict[str, str] = field(default_factory=dict)
    quality_cot: str = ""
    theme_cot: str = ""


def parse_stories(raw: str) -> list[str]:
    """Parse LLM story extraction output into a list of story texts.

    Expected format (one story per block, blocks separated by blank lines):
        STORY: <story text, possibly multi-line>

    Returns an empty list ONLY for an explicit ``NO_STORIES`` reply — the genuine
    zero-story extraction, the one thing that may become an
    ``agent/newsletter/no-stories`` outcome. Any other reply that yields no
    parsed story is unusable output and raises ``LLMContentError`` (D5 Rule 1 /
    D20): conflating the two let a *failed* extraction commit a false
    no-stories label. Empty/whitespace-only input is unreachable in production
    (llm_client's content guard raises first) but is treated the same way rather
    than letting this contract lie.
    """
    # Normalize CRLF/CR to LF first so the blank-line split and the verbatim
    # story slices behave identically regardless of the model's line endings.
    stripped = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
    if stripped.upper() == "NO_STORIES":
        return []

    stories = []
    # Split only at a STORY: that begins the text or follows a blank line (the
    # documented "one story per block, separated by blank lines" format). This
    # keeps a story whose own body contains a line starting with "STORY:" intact
    # instead of splitting it in two.
    blocks = re.split(r"(?:\A|\n[ \t]*\n)(?=STORY:)", stripped)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # ``search`` (not ``match``): a preamble line the model glued onto the
        # first STORY: with a single newline is skipped rather than dropping the
        # whole block. The first STORY: wins and DOTALL keeps any later
        # "STORY:"-prefixed body line as part of this story's text.
        match = re.search(r"^STORY:\s*(.*)", block, flags=re.DOTALL | re.MULTILINE)
        if match:
            text = match.group(1).strip()
            if text:
                stories.append(text)

    if not stories:
        raise LLMContentError(
            "story extraction reply was neither NO_STORIES nor parsable into "
            f"stories: {stripped[:200]!r}"
        )
    return stories


def parse_quality_scores(raw: str) -> dict[str, int] | None:
    """Parse LLM quality assessment output into dimension scores.

    Each dimension is graded POOR/OK/GOOD (mapped to 1/2/3). Returns None if any
    dimension is missing or its value is not one of those tokens (a legacy digit
    or an unknown word is a parse failure, not a clamped score).
    """
    if not raw.strip():
        return None

    scores = {}
    for dim in _DIMENSIONS:
        pattern = rf"{dim.upper()}\s*:\s*(POOR|OK|GOOD)\b"
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if not match:
            return None
        scores[dim] = _SCORE_TOKENS[match.group(1).upper()]

    return scores


def parse_themes(raw: str) -> dict[str, str]:
    """Parse LLM theme classification output into graded theme labels.

    Each recognized theme line is ``THEME: ABSENT|PRESENT|EMPHASIZED``. Returns a
    dict mapping lowercase theme name -> "present"/"emphasized"; Absent themes and
    unrecognized names are omitted. Returns an empty dict for a NONE response.
    """
    stripped = raw.strip()
    if not stripped or stripped.upper() == "NONE":
        return {}

    themes: dict[str, str] = {}
    for line in stripped.splitlines():
        kind, name, grade = classify_theme_line(line)
        # Store only in-taxonomy present/emphasized gradings; absent is omitted,
        # off-taxonomy/anomalous lines are dropped (surfaced by the eval detectors).
        if kind == "theme" and grade != "absent":
            themes[name] = grade

    return themes


def compute_tier(scores: dict[str, int]) -> NewsletterTier:
    """Derive quality tier from the 3-value dimension scores.

    Dimensions are 1/2/3 (Poor/OK/Good); the tier is banded off their mean:
    excellent >= 2.75, good >= 2.25, fair >= 1.75, else poor (issue #53).
    """
    avg = sum(scores.values()) / len(scores)
    if avg >= 2.75:
        return NewsletterTier.EXCELLENT
    if avg >= 2.25:
        return NewsletterTier.GOOD
    if avg >= 1.75:
        return NewsletterTier.FAIR
    return NewsletterTier.POOR


# Ordering of theme grades for cross-story aggregation (higher = stronger).
THEME_GRADE_RANK = {"present": 1, "emphasized": 2}


def aggregate_theme_grades(stories: list["StoryResult"]) -> dict[str, str]:
    """Merge per-story graded themes into one dict for the whole newsletter.

    Each theme takes the strongest grade seen across all stories
    (emphasized > present); themes absent from every story are omitted.
    """
    merged: dict[str, str] = {}
    for story in stories:
        for theme, grade in story.themes.items():
            if THEME_GRADE_RANK.get(grade, 0) > THEME_GRADE_RANK.get(merged.get(theme, ""), 0):
                merged[theme] = grade
    return merged


def parse_send_date(date_header: str, internal_date_ms: str | None = None) -> str | None:
    """Normalize an email's send date to an ISO-8601 UTC string.

    Prefers the RFC-2822 ``Date`` header (a header with no timezone is assumed
    UTC); falls back to Gmail's ``internalDate`` (epoch milliseconds) when the
    header is missing or unparseable. Returns None if neither yields a valid date.
    ISO-8601 UTC sorts lexicographically = chronologically, which the review TUI
    relies on for date sorting/filtering (issue #35/#36).
    """
    if date_header:
        try:
            dt = parsedate_to_datetime(date_header)
            if dt is not None:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                # astimezone can OverflowError near datetime's min/max year.
                return dt.astimezone(timezone.utc).isoformat()
        except (ValueError, TypeError, OverflowError):
            pass

    if internal_date_ms:
        try:
            seconds = int(internal_date_ms) / 1000
            return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()
        except (ValueError, TypeError, OverflowError, OSError):
            pass

    return None


class AssessmentSinkError(Exception):
    """The assessments JSONL could not be written (read-only mount, full disk,
    bad permissions, sink path is a directory).

    Raised by the daemon at the ``write_assessment`` call site, wrapping the
    underlying ``OSError`` as its cause. Its whole purpose is to be
    *distinguishable*: under decision D5's sink corollary a sink fault is
    shared-cause (the disk, not the thread), so it must never count toward
    give-up — and a bare ``OSError`` cannot be caught for that without also
    catching ``TimeoutError``, which subclasses it and *is* a strike candidate.
    """


def write_assessment(
    output_file: str,
    message_id: str,
    thread_id: str,
    sender: str,
    subject: str,
    overall_tier: NewsletterTier | None,
    stories: list[StoryResult],
    *,
    send_date: str | None = None,
    model: str | None = None,
) -> None:
    """Append a newsletter assessment record to the JSONL output file.

    ``timestamp`` is the *processed* time (UTC now). ``send_date`` is the email's
    own send date (ISO-8601 UTC, email-intrinsic) and ``model`` is the classifier
    model — both used by the review TUI (issue #35/#36). Old records lacking these
    keys are read with ``.get()`` fallbacks by consumers.

    The record shape is documented in README-technical's "Assessment record
    schema" (decision D12); ``schema_version`` is bumped there when the shape
    changes. Absence of the key marks a pre-versioning record.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "message_id": message_id,
        "thread_id": thread_id,
        "from": sender,
        "subject": subject,
        "send_date": send_date,
        "model": model,
        "overall_tier": overall_tier.value if overall_tier else None,
        "stories": [
            {
                "text": s.text,
                "scores": s.scores,
                "average_score": s.average_score,
                "tier": s.tier.value if s.tier else None,
                "themes": s.themes,
                "quality_cot": s.quality_cot,
                "theme_cot": s.theme_cot,
            }
            for s in stories
        ],
    }

    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Assessment sink diagnostics
#
# The JSONL sink is the only durable copy of a newsletter's grading, and the
# ways it goes wrong are all silent: writes to an unmounted container path
# succeed right up until the container is recreated, and a read-only mount only
# surfaces per-thread, mid-cycle. These preflight the sink at startup so a
# misconfigured deployment announces itself instead of quietly grading into the
# void.
# ---------------------------------------------------------------------------

# Marker files every mainstream container runtime drops in the root filesystem.
_CONTAINER_MARKERS = (Path("/.dockerenv"), Path("/run/.containerenv"))
_MOUNTINFO = Path("/proc/self/mountinfo")


def running_in_container() -> bool:
    """True when this process looks like it is running inside a container."""
    return any(marker.exists() for marker in _CONTAINER_MARKERS)


def read_mountinfo(path: Path = _MOUNTINFO) -> str | None:
    """Read ``/proc/self/mountinfo``, or None where it isn't available (macOS)."""
    try:
        return path.read_text()
    except OSError:
        return None


# Filesystem types that do not outlive the container, whatever they are mounted
# over: RAM-backed, or another copy-on-write layer discarded with it. A mount of
# one of these is not persistence, so a mount-point-only check would clear a
# tmpfs over the sink directory that loses every record on restart.
_EPHEMERAL_FSTYPES = frozenset({"tmpfs", "ramfs", "overlay", "overlayfs", "aufs", "devtmpfs"})


def parse_mounts(mountinfo: str) -> dict[str, tuple[str, str]]:
    """Map mount point -> ``(source, fstype)`` from ``/proc/self/mountinfo`` text.

    The mount point is field 5. The source is field 4: the path *within its own
    filesystem* — for a Docker bind mount that is the host directory
    (``/var/lib/docker/volumes/<name>/_data`` for a named volume), which is what
    makes "am I appending to the directory I actually review?" answerable from
    inside the container. Note it is relative to the ROOT OF THAT FILESYSTEM, not
    to the host's ``/``: if the host keeps ``/srv`` on its own filesystem, a bind
    of ``/srv/stack/data`` reports ``/stack/data``. The fstype is the first field
    after the ``-`` separator, and is what distinguishes a durable bind mount from
    a tmpfs.

    Paths containing spaces, tabs, newlines, or backslashes are octal-escaped in
    mountinfo; ``\\040`` (space) is the one worth decoding for a bind-mounted
    host directory. Malformed lines are skipped rather than raising — this is a
    diagnostic, and it must never be the thing that breaks startup.
    """
    mounts = {}
    for line in mountinfo.splitlines():
        fields = line.split(" ")
        if len(fields) < 10:  # a real mountinfo line has at least 10 fields
            continue
        # Optional fields sit between field 7 and the "-", so the fstype's index
        # varies: find the separator rather than assuming a position.
        try:
            fstype = fields[fields.index("-", 6) + 1]
        except (ValueError, IndexError):
            continue
        mounts[fields[4].replace("\\040", " ")] = (fields[3].replace("\\040", " "), fstype)
    return mounts


def covering_mount(path: Path, mountinfo: str | None) -> tuple[str, str, str] | None:
    """The mount actually holding *path*, as ``(mount_point, source, fstype)``.

    The nearest enclosing mount: the longest mount point that is *path* itself or
    one of its ancestors. Compared as paths, not strings, so ``/app/dataset`` is
    never mistaken for a mount covering ``/app/data``. Returns None when
    mountinfo is unreadable (non-Linux) — no evidence, no claim.
    """
    if mountinfo is None:
        return None

    nearest: tuple[str, str, str] | None = None
    for point, (source, fstype) in parse_mounts(mountinfo).items():
        # is_relative_to is already true for an equal path, so this covers a bind
        # mount of the sink file itself as well as one of a parent directory.
        if path.is_relative_to(Path(point)):
            if nearest is None or len(point) > len(nearest[0]):
                nearest = (point, source, fstype)
    return nearest


def sink_persistence_warning(path: Path, mountinfo: str | None) -> str | None:
    """:func:`mount_persistence_warning` against raw ``mountinfo`` text."""
    return mount_persistence_warning(path, covering_mount(path, mountinfo))


def mount_persistence_warning(path: Path, mount: tuple[str, str, str] | None) -> str | None:
    """Warn when *mount* — the mount covering *path* — is not keeping it.

    The daemon's ``output_file`` is relative to the working directory (``/app``
    in the image), so without a volume mount the assessments land in the
    container layer: every write succeeds, the review TUI on the host sees
    nothing new, and the records are destroyed on the next ``docker compose up``.
    Two shapes mean exactly that, and both look fine to a mount-point-only check:

      * the mount covering the path IS the container root — nothing was mounted
        over it; and
      * something *was* mounted over it, but with an ephemeral filesystem: a
        tmpfs is RAM, another overlay is another throwaway layer.

    Takes the already-resolved mount so a caller that also wants to *report* the
    mount (the startup preflight does) reads and parses ``mountinfo`` once.
    Returns None when there's no evidence to act on (mountinfo unreadable), so
    the check never cries wolf on a platform it can't inspect.
    """
    if mount is None:
        return None
    point, _source, fstype = mount
    if point != "/" and fstype not in _EPHEMERAL_FSTYPES:
        return None

    covers = (
        "which no volume covers — it is inside the container's writable layer"
        if point == "/"
        else f"which is covered only by a {fstype} mount at {point} — that filesystem "
        "does not outlive the container"
    )
    return (
        f"Newsletter assessments resolve to {path}, {covers}. Writes will appear to "
        "succeed, but the records are invisible on the host and are DESTROYED when the "
        "container is recreated. Mount the directory you review, e.g. "
        "'- ./data:/app/data' under the service's volumes."
    )


def sink_writability_warning(path: Path) -> str | None:
    """Warn when *path* cannot be appended to (read-only mount, permissions).

    ``write_assessment`` creates missing parents, so a not-yet-existing file is
    fine as long as the nearest existing ancestor is writable — that ancestor is
    what gets checked, and named, when the file itself isn't there yet.

    An ``output_file`` naming an existing DIRECTORY is caught here too: it is
    fatal for every grading (``open(dir, "a")`` raises ``IsADirectoryError``) yet
    ``os.access`` reports a directory as perfectly writable, so the permission
    check alone would wave the one guaranteed-fatal misconfiguration through.
    """
    if path.is_dir():
        return (
            f"Newsletter assessments sink is a directory, not a file: {path}. Every "
            "grading will fail to save. Point [newsletter] output_file at a file "
            "inside it (e.g. data/newsletter_assessments.jsonl)."
        )
    if path.exists():
        if os.access(path, os.W_OK):
            return None
        target = path
    else:
        target = next((p for p in path.parents if p.exists()), Path(path.anchor or "."))
        if os.access(target, os.W_OK | os.X_OK):
            return None

    return (
        f"Newsletter assessments sink is not writable: {target}. Newsletters will be "
        "left unprocessed (retried every cycle until this is fixed, never abandoned) "
        "rather than graded into a record that cannot be saved."
    )


def count_records(path: Path) -> int | None:
    """Count non-blank lines in the assessments JSONL, or None if unreadable.

    Reported at startup: a daemon that has been grading for weeks against a sink
    holding 0 records is writing somewhere other than the file being reviewed —
    the single number that makes a misdirected sink obvious.
    """
    try:
        with open(path) as f:
            return sum(1 for line in f if line.strip())
    except FileNotFoundError:
        return 0
    except OSError:
        return None


def is_newsletter(messages: list[dict], recipient: str) -> bool:
    """Check if any message in a thread was sent to the newsletter address.

    Checks both To and Cc headers. The match is an exact addr-spec comparison
    (decision D3): each header value is parsed as an address list, and a
    message matches only when one of its addresses equals the configured
    recipient, case-insensitively. Display names never match, and neither do
    superstring addresses.
    """
    target = recipient.casefold()
    for msg in messages:
        headers = msg.get("payload", {}).get("headers", [])
        for header_name in ("To", "Cc"):
            value = get_header(headers, header_name)
            if not value:
                continue
            for _display_name, addr in getaddresses([value]):
                if addr.casefold() == target:
                    return True
    return False


class NewsletterClassifier:
    """Classifies newsletter stories for quality and themes."""

    def __init__(self, cloud_llm: LLMClient, config: dict):
        self.cloud_llm = cloud_llm
        nl_config = config["newsletter"]
        self.extraction_config = nl_config["prompts"]["story_extraction"]
        self.quality_config = nl_config["prompts"]["quality_assessment"]
        self.theme_config = nl_config["prompts"]["theme_classification"]

    async def extract_stories(self, body: str) -> list[str]:
        """Extract individual story texts from a newsletter body."""
        user_content = self.extraction_config["user_template"].format(body=body)
        raw, _ = await self.cloud_llm.complete(
            self.extraction_config["system"],
            user_content,
            include_thinking=True,
        )
        return parse_stories(raw)

    async def assess_quality(self, text: str) -> tuple[dict[str, int] | None, str]:
        """Score a story on the 4-dimension quality rubric."""
        user_content = self.quality_config["user_template"].format(text=text)
        raw, cot = await self.cloud_llm.complete(
            self.quality_config["system"],
            user_content,
            include_thinking=True,
        )
        scores = parse_quality_scores(raw)
        return scores, cot

    async def classify_themes(self, text: str) -> tuple[dict[str, str], str]:
        """Tag a story with graded Ends Statement themes (theme -> grade)."""
        user_content = self.theme_config["user_template"].format(text=text)
        raw, cot = await self.cloud_llm.complete(
            self.theme_config["system"],
            user_content,
            include_thinking=True,
        )
        return parse_themes(raw), cot

    async def classify_newsletter(
        self, body: str, *, on_answer: Callable[[], None] | None = None
    ) -> list[StoryResult]:
        """Run the full newsletter classification pipeline over story texts.

        ``on_answer``, when given, is called after every LLM call the provider
        answers (extraction, each quality assessment, each theme classification)
        — the daemon binds it to the newsletter halt slot's ``record_success`` so
        a provider that answers and then fails a later call has still been seen
        answering (decision D22 item 2; review of PR #81).

        Individual *per-story* failures are isolated — a quality failure doesn't
        prevent theme classification, and vice versa — but only while some story
        still grades: if stories were extracted and *not one* produced scores,
        the whole grading is content-less and raises ``LLMContentError``
        (D5 Rule 1 / D20, issue #30's parse-to-None route). Committing it would
        label the newsletter `no-stories` off a failure, indistinguishable from
        a genuine zero-story extraction. An empty result list therefore means
        exactly one thing: extraction succeeded and found no stories.

        A *transient* LLM outage (LLMUnavailableError) is NOT isolated: it
        propagates so the daemon retries the whole newsletter thread next cycle,
        rather than committing a permanently mis-graded newsletter (empty
        tier/themes) and marking it processed. Mirrors the email pipeline's
        transient-outage guarantee. An out-of-funds provider (LLMBalanceError)
        propagates likewise: the thread stays unprocessed and the NEWSLETTER
        function halts (on the third consecutive fault) until its re-probe
        finds the provider answering again (D22) — email triage keeps running
        on its own tiers (decision D5's scope rule, D19).
        """
        stories = await self.extract_stories(body)
        if on_answer is not None:
            on_answer()
        if not stories:
            return []

        results = []
        for text in stories:
            result = StoryResult(text=text)

            try:
                scores, quality_cot = await self.assess_quality(text)
                if on_answer is not None:
                    on_answer()
                result.quality_cot = quality_cot
                if scores:
                    result.scores = scores
                    result.average_score = sum(scores.values()) / len(scores)
                    result.tier = compute_tier(scores)
            except _PIPELINE_WIDE_ERRORS:
                # transient outage, a content-less response (issue #30), or an
                # out-of-funds provider: all affect every story, so propagate
                # (to the cycle attribution, or to the newsletter function's
                # halt) rather than commit a permanently mis-graded (empty)
                # newsletter.
                raise
            except Exception:
                log.warning("Quality assessment failed for story: %s", text[:60])

            try:
                themes, theme_cot = await self.classify_themes(text)
                if on_answer is not None:
                    on_answer()
                result.themes = themes
                result.theme_cot = theme_cot
            except _PIPELINE_WIDE_ERRORS:
                raise
            except Exception:
                log.warning("Theme classification failed for story: %s", text[:60])

            results.append(result)

        # Stories exist but none of them graded (every quality reply unparseable
        # or per-story-failed): a content-less grading, not a `no-stories`
        # outcome — raise so the daemon defers the thread instead of committing
        # a tier-less record plus the no-stories label (D5 Rule 1 / D20).
        if all(r.scores is None for r in results):
            raise LLMContentError(
                f"no story graded: all {len(results)} quality replies were unusable"
            )

        return results
