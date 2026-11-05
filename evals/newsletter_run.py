"""Replay the newsletter golden set through the real NewsletterClassifier.

Usage:
    python -m evals.newsletter_run
    python -m evals.newsletter_run --golden-set evals/newsletter_golden_set.jsonl --mode all
    python -m evals.newsletter_run --prompts alt.toml --tag variant
    python -m evals.newsletter_run --mode quality --no-cache

Mirrors evals/run_eval.py: load config (tomllib + env substitution), resolve the
newsletter endpoint, build one LLMClient from [newsletter.llm], wrap in the disk
LLM cache unless --no-cache, preflight the endpoint, then replay the golden set.
"""

import argparse
import asyncio
import copy
import hashlib
import json
import os
import sys
import time
import tomllib
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

from config_utils import substitute_env_vars
from daemon import quiet_http_logging, resolve_newsletter_llm_endpoint
from evals import format_network_error, plural
from evals.llm_cache import CachedLLMClient
from evals.newsletter_grade import latest_labels, read_label_log
from evals.newsletter_schemas import (
    ExtractionPrediction,
    GoldenNewsletter,
    GoldenStory,
    NewsletterRunMeta,
    NewsletterThinkingEntry,
    StoryPrediction,
)
from llm_client import LLMClient
from newsletter import NewsletterClassifier, classify_theme_line, compute_tier


def load_golden_set(
    path: Path, reviewed_only: bool = True,
) -> tuple[list[GoldenNewsletter], dict]:
    """Load newsletters from the golden set JSONL.

    Drops ``excluded`` newsletters unconditionally; drops unreviewed ones unless
    ``reviewed_only`` is False. Blank lines are skipped.

    Returns (kept newsletters, stats) where stats counts what the file held and
    what filtering dropped: {"total", "excluded", "unreviewed"} — so the CLI can
    explain an empty result instead of dead-ending.
    """
    newsletters = []
    stats = {"total": 0, "excluded": 0, "unreviewed": 0}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            gn = GoldenNewsletter.from_dict(json.loads(line))
            stats["total"] += 1
            if gn.excluded:
                stats["excluded"] += 1
                continue
            if not gn.reviewed:
                stats["unreviewed"] += 1
                if reviewed_only:
                    continue
            newsletters.append(gn)
    return newsletters, stats


def format_load_summary(kept: int, stats: dict, path) -> str:
    """Human-readable summary of what the golden-set load kept vs dropped.

    Covers the three UX dead-ends: an empty file (points at harvest), an
    all-filtered file (points at the labeling TUI and --include-unreviewed),
    and a partial load (reports the unreviewed/excluded breakdown).
    """
    total = stats["total"]
    if kept == 0:
        if total == 0:
            return (
                f"No newsletters to evaluate: golden set {path} is empty.\n"
                "Harvest newsletters first with: python -m evals.newsletter_harvest"
            )
        return (
            f"No newsletters to evaluate: {total} loaded from {path}, "
            f"but {stats['unreviewed']} unreviewed and {stats['excluded']} excluded.\n"
            "Label them with: python -m evals.newsletter_label\n"
            "(or pass --include-unreviewed to run extraction on uncurated newsletters)"
        )
    dropped = []
    if total - stats["excluded"] - kept > 0:
        dropped.append(f"{stats['unreviewed']} unreviewed skipped")
    if stats["excluded"]:
        dropped.append(f"{stats['excluded']} excluded")
    if dropped:
        return f"Loaded {kept} of {total} newsletters from {path} ({', '.join(dropped)})"
    return f"Loaded {kept} newsletters from {path}"


def compute_prompt_hash(config: dict) -> str:
    """Content-hash the newsletter prompt block so prompt A/Bs are self-identifying.

    Hashes config["newsletter"]["prompts"] deterministically (sorted keys) — any
    change to a prompt string (system or user_template) changes the hash, and an
    identical prompt block always hashes the same.
    """
    prompts = config["newsletter"]["prompts"]
    raw = json.dumps(prompts, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def describe_endpoint() -> str:
    """The resolved newsletter LLM URL + which env var supplied it.

    Mirrors daemon.resolve_newsletter_llm_endpoint()'s precedence so error
    messages can say WHICH endpoint was probed — the NEWSLETTER_LLM_URL vs
    CLOUD_LLM_URL fallback is a documented source of confusion.
    """
    url = os.environ.get("NEWSLETTER_LLM_URL")
    if url:
        return f"{url} (from NEWSLETTER_LLM_URL)"
    return f"{os.environ.get('CLOUD_LLM_URL') or '(unset)'} (from CLOUD_LLM_URL)"


def _endpoint_url(llm) -> str | None:
    """Best-effort base_url of a (possibly cache-wrapped) LLM client."""
    url = getattr(llm, "base_url", None)
    if url:
        return url
    inner = getattr(llm, "inner", None)
    return getattr(inner, "base_url", None) if inner is not None else None


class _RawCapture:
    """Wraps an LLM client to record the raw (non-thinking) response of each call.

    The NewsletterClassifier parses scores/themes internally and only returns the
    parsed result + chain-of-thought, discarding the raw model text. To store the
    raw text in a StoryPrediction (for debugging a parse failure) without a second
    LLM call, temporarily route the classifier's client through this recorder — it
    delegates every call to the inner client and keeps the last response.
    """

    def __init__(self, inner):
        self.inner = inner
        self.last_raw = None
        self.last_thinking = None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    async def complete(self, system_prompt, user_content, include_thinking=False):
        result = await self.inner.complete(system_prompt, user_content, include_thinking=True)
        response, thinking = result
        self.last_raw = response
        self.last_thinking = thinking
        if include_thinking:
            return response, thinking
        return response


def _elapsed(llm, start: float) -> float:
    """Actual LLM call time when caching (0 for pure cache hits), else wall time.

    Mirrors run_eval: a CachedLLMClient exposes take_llm_seconds() (miss-only
    time); an un-cached raw client falls back to wall-clock.
    """
    if hasattr(llm, "take_llm_seconds"):
        return round(llm.take_llm_seconds(), 3)
    return round(time.monotonic() - start, 3)


async def evaluate_extraction(
    newsletter: GoldenNewsletter, classifier: NewsletterClassifier,
) -> tuple[ExtractionPrediction, NewsletterThinkingEntry]:
    """Run extract_stories on the raw body; compare predicted vs golden story list.

    Extraction is scored at the newsletter-body level: the raw body goes in, and
    the predicted [{text}] list is matched (in the report) against the
    newsletter's confirmed golden stories. The extraction chain-of-thought is
    captured into a newsletter-level ``NewsletterThinkingEntry`` (thread_id +
    extraction_cot) so segmentation mangles can be prompt-debugged from the
    ``.cot.jsonl`` sidecar, just like quality/theme scoring.
    """
    pred = ExtractionPrediction(
        thread_id=newsletter.thread_id,
        golden_stories=[
            {"story_id": s.story_id, "text": s.text}
            for s in newsletter.stories
        ],
    )
    thinking = NewsletterThinkingEntry(thread_id=newsletter.thread_id)

    real_client = classifier.cloud_llm
    capture = _RawCapture(real_client)
    # Shallow-copy for the same concurrency reason as evaluate_story: swapping
    # the shared classifier's client would race under parallelism > 1.
    local_classifier = copy.copy(classifier)
    local_classifier.cloud_llm = capture
    start = time.monotonic()
    try:
        stories = await local_classifier.extract_stories(newsletter.body)
        pred.predicted_stories = [{"text": x} for x in stories]
        thinking.extraction_cot = capture.last_thinking or ""
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        pred.error = format_network_error(exc, "newsletter LLM")
    except Exception as exc:
        pred.error = str(exc)
    if pred.error:
        url = _endpoint_url(real_client)
        if url:
            pred.error += f" (endpoint: {url})"
    pred.duration_seconds = _elapsed(real_client, start)
    return pred, thinking


async def evaluate_story(
    story: GoldenStory, thread_id: str, classifier: NewsletterClassifier,
    do_quality: bool = True, do_themes: bool = True, source: str = "curated",
) -> tuple[StoryPrediction, NewsletterThinkingEntry]:
    """Score one fixed golden story on quality and/or themes; derive tier from scores.

    Quality and themes are evaluated on the fixed golden story text, decoupled
    from extraction variability. predicted_tier is derived via compute_tier only
    when the scores parsed; a parse failure leaves scores/tier None (an error in
    the report, not a mis-tier). *do_quality*/*do_themes* let quality/themes modes
    skip the other LLM call entirely (skipped fields stay None/[], with raw=None
    marking "never attempted" as distinct from a parse failure).
    """
    pred = StoryPrediction(
        story_id=story.story_id,
        thread_id=thread_id,
        expected_scores=story.expected_scores,
        expected_tier=story.expected_tier,
        expected_themes=dict(story.expected_themes),
        source=source,
    )
    thinking = NewsletterThinkingEntry(story_id=story.story_id)

    real_client = classifier.cloud_llm
    capture = _RawCapture(real_client)
    # Shallow-copy the classifier so swapping the client is concurrency-safe: under
    # parallelism>1, mutating the shared classifier's cloud_llm would race between
    # tasks. The copy shares the (immutable) config dicts but owns its client ref.
    local_classifier = copy.copy(classifier)
    local_classifier.cloud_llm = capture
    start = time.monotonic()
    try:
        if do_quality:
            capture.last_raw = None
            scores, quality_cot = await local_classifier.assess_quality(story.text)
            pred.scores_raw = capture.last_raw
            pred.predicted_scores = scores
            thinking.quality_cot = quality_cot
            if scores:
                pred.predicted_tier = compute_tier(scores).value

        if do_themes:
            capture.last_raw = None
            themes, theme_cot = await local_classifier.classify_themes(story.text)
            pred.themes_raw = capture.last_raw
            pred.predicted_themes = themes
            thinking.theme_cot = theme_cot
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        pred.error = format_network_error(exc, "newsletter LLM")
    except Exception as exc:
        pred.error = str(exc)
    if pred.error:
        url = _endpoint_url(real_client)
        if url:
            pred.error += f" (endpoint: {url})"
    pred.duration_seconds = _elapsed(real_client, start)
    return pred, thinking


def load_story_labels(
    path: Path, curated_thread_ids: set[str],
) -> tuple[list[tuple[GoldenStory, str]], dict]:
    """Ground-truth stories from the story-label log written by newsletter_grade.

    Returns ((story, thread_id) pairs, stats). Only the newest label per story
    counts, and only graded ones are ground truth: a miscut story has none. A
    graded story whose thread is in the curated golden set is left to that set,
    so a thread curated after one of its stories was graded is not scored twice.
    A missing file is an empty log. stats counts {"graded", "miscut",
    "in_curated_threads"} so the CLI can say what it used.
    """
    stats = {"graded": 0, "miscut": 0, "in_curated_threads": 0}
    pairs: list[tuple[GoldenStory, str]] = []
    for label in latest_labels(read_label_log(path)).values():
        if label.status != "graded":
            stats["miscut"] += 1
        elif label.thread_id in curated_thread_ids:
            stats["in_curated_threads"] += 1
        else:
            stats["graded"] += 1
            pairs.append((label.to_golden_story(), label.thread_id))
    return pairs, stats


VALID_MODES = ("extraction", "quality", "themes", "all")


def select_stories(
    golden_set: list[GoldenNewsletter],
) -> tuple[list[tuple[GoldenStory, str]], int, int]:
    """Pick the (story, thread_id) pairs story modes evaluate + skip counts.

    A story is evaluated only when it is Phase-B labeled (``story.reviewed``)
    and not ``excluded`` — an unlabeled story has no ground truth, so scoring
    it would burn LLM calls and pollute metrics with expected=None/[] rows.

    Returns (pairs, n_excluded, n_unlabeled).
    """
    pairs: list[tuple[GoldenStory, str]] = []
    n_excluded = n_unlabeled = 0
    for nl in golden_set:
        for story in nl.stories:
            if story.excluded:
                n_excluded += 1
            elif not story.reviewed:
                n_unlabeled += 1
            else:
                pairs.append((story, nl.thread_id))
    return pairs, n_excluded, n_unlabeled


async def run_evaluation(
    golden_set: list[GoldenNewsletter],
    classifier: NewsletterClassifier,
    mode: str,
    parallelism: int = 1,
    progress: bool = False,
    extra_stories: list[tuple[GoldenStory, str]] | tuple = (),
) -> tuple[list, list[NewsletterThinkingEntry]]:
    """Replay the golden set through the classifier under *mode*.

    *extra_stories* are (story, thread_id) pairs from the story-label log
    (``load_story_labels``). They are scored in the story modes only, never as
    extraction truth: their boundaries were drawn by the extractor itself.

    Returns (rows, thinking_entries). ``rows`` mixes ExtractionPrediction (one per
    reviewed newsletter, extraction/all modes) and StoryPrediction (one per
    reviewed, non-excluded golden story, quality/themes/all modes). Concurrency is
    bounded by an asyncio.Semaphore(parallelism); the shared classifier's client is
    never mutated (evaluate_story shallow-copies it), so tasks don't race.
    With *progress*, a "[k/N] <what>" line is printed to stderr as each task
    finishes, so long runs show signs of life.
    """
    semaphore = asyncio.Semaphore(parallelism)
    do_extraction = mode in ("extraction", "all")
    do_quality = mode in ("quality", "all")
    do_themes = mode in ("themes", "all")
    ticker = {"done": 0, "total": 0}

    def _tick(label: str) -> None:
        ticker["done"] += 1
        if progress:
            print(f"  [{ticker['done']}/{ticker['total']}] {label}", file=sys.stderr)

    async def run_extraction(nl: GoldenNewsletter):
        async with semaphore:
            result = await evaluate_extraction(nl, classifier)
        _tick(f"extraction {nl.thread_id}")
        return result  # (ExtractionPrediction, NewsletterThinkingEntry)

    async def run_story(story: GoldenStory, thread_id: str, source: str = "curated"):
        async with semaphore:
            result = await evaluate_story(
                story, thread_id, classifier,
                do_quality=do_quality, do_themes=do_themes, source=source,
            )
        _tick(f"story {story.story_id}")
        return result

    extraction_tasks = []
    if do_extraction:
        extraction_tasks = [run_extraction(nl) for nl in golden_set]

    story_tasks = []
    if do_quality or do_themes:
        story_pairs_in, _n_excluded, _n_unlabeled = select_stories(golden_set)
        story_tasks = [run_story(story, tid) for story, tid in story_pairs_in]
        story_tasks += [
            run_story(story, tid, source="story_labels") for story, tid in extra_stories
        ]

    ticker["total"] = len(extraction_tasks) + len(story_tasks)

    extraction_pairs = await asyncio.gather(*extraction_tasks)
    story_pairs = await asyncio.gather(*story_tasks)

    rows = []
    thinking = []
    for pred, entry in extraction_pairs:
        rows.append(pred)
        thinking.append(entry)
    for pred, entry in story_pairs:
        rows.append(pred)
        thinking.append(entry)
    return rows, thinking


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge *override* into *base* in place (override wins on leaves)."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def format_story_label_summary(stats: dict, path) -> str:
    """One line on what the story-label log contributed to this run."""
    parts = [
        f"{plural(stats['graded'], 'graded story', 'graded stories')} "
        f"from the story-label log {path}"
    ]
    if stats["miscut"]:
        parts.append(f"{stats['miscut']} miscut left out")
    if stats["in_curated_threads"]:
        parts.append(f"{stats['in_curated_threads']} left to the curated golden set")
    return "; ".join(parts)


def build_meta(
    config: dict,
    config_path: str,
    golden_path: str,
    golden_set: list[GoldenNewsletter],
    mode: str,
    tag: str,
    parallelism: int,
    story_labels_path: str = "",
    story_label_count: int = 0,
) -> "NewsletterRunMeta":
    """Assemble the NewsletterRunMeta first-line record for a results file.

    Records prompt_hash (content-hash of the merged prompt block) + the three
    system prompts verbatim + model/temperature/max_tokens/extra_body/counts/tag/
    mode so prompt A/Bs are self-identifying and comparable.

    story_count reflects what the run actually evaluated: story modes count only
    Phase-B-labeled, non-excluded stories (the metric denominators), while an
    extraction-only run counts every golden story — evaluate_extraction matches
    predictions against the full story list, so an --include-unreviewed
    extraction run must not record 0. Stories from the story-label log
    (*story_label_count*) are scored in the story modes only and are counted
    there, not in an extraction-only run.
    """
    llm_cfg = config["newsletter"]["llm"]
    prompts = config["newsletter"]["prompts"]
    config_bytes = json.dumps(config, sort_keys=True).encode()
    if mode == "extraction":
        story_count = sum(len(nl.stories) for nl in golden_set)
        story_label_count = 0
    else:
        story_count = len(select_stories(golden_set)[0]) + story_label_count
    return NewsletterRunMeta(
        run_id=uuid.uuid4().hex,
        timestamp=datetime.now(timezone.utc).isoformat(),
        config_hash=hashlib.sha256(config_bytes).hexdigest()[:16],
        config_path=config_path,
        newsletter_model=llm_cfg["model"],
        golden_set_path=golden_path,
        golden_set_count=len(golden_set),
        story_count=story_count,
        mode=mode,
        prompt_hash=compute_prompt_hash(config),
        temperature=llm_cfg.get("temperature", 0.0),
        max_tokens=llm_cfg.get("max_tokens", 0),
        extra_body=llm_cfg.get("extra_body"),
        parallelism=parallelism,
        tag=tag,
        extraction_system_prompt=prompts["story_extraction"]["system"],
        quality_system_prompt=prompts["quality_assessment"]["system"],
        theme_system_prompt=prompts["theme_classification"]["system"],
        story_labels_path=story_labels_path if story_label_count else "",
        story_label_count=story_label_count,
    )


def build_output_path(output_dir: Path, mode: str, tag: str, run_id: str) -> Path:
    """Build the timestamped results file path: <ts>_<mode>_<tag>_<runid8>.jsonl."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parts = [timestamp, mode]
    if tag:
        parts.append(tag)
    parts.append(run_id[:8])
    return output_dir / ("_".join(parts) + ".jsonl")


def write_results(path: Path, meta: "NewsletterRunMeta", rows: list) -> None:
    """Write the run meta (first line) then each prediction row to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps(meta.to_dict()) + "\n")
        for row in rows:
            f.write(json.dumps(row.to_dict()) + "\n")


def write_thinking_sidecar(
    results_path: Path, thinking_entries: list[NewsletterThinkingEntry],
) -> Path | None:
    """Write non-empty chain-of-thought entries to the .cot.jsonl sidecar.

    Only entries with at least one non-empty cot field are written; if none
    qualify, no sidecar file is created. Returns the sidecar path when one was
    written (so the run summary can mention it), else None.
    """
    entries = [
        t for t in thinking_entries
        if t.quality_cot or t.theme_cot or t.extraction_cot
    ]
    if not entries:
        return None
    sidecar_path = results_path.with_suffix(".cot.jsonl")
    with open(sidecar_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry.to_dict()) + "\n")
    return sidecar_path


def _theme_line_anomalies(themes_raw: str) -> tuple[bool, list[str], list[str]]:
    """Classify a raw theme response for the run summary (issue #53 format).

    Routes every line through the shared ``newsletter.classify_theme_line`` so
    this diagnostic and the production parser cannot drift (Finding 2). Returns
    ``(is_garbage, dropped_name_tokens, near_miss_lines)``:

    - *is_garbage* — the response is neither NONE nor contains any recognizable
      theme grading (prose masquerading as a confident answer). An all-ABSENT
      response is a valid empty result, not garbage.
    - *dropped_name_tokens* — off-taxonomy theme names the parser silently ignored.
    - *near_miss_lines* — non-blank lines that are not a recognizable grading and
      parse_themes silently skipped (a hyphenated/spaced name or a misspelled
      grade that reads as a missed prediction). Surfacing them keeps a systematic
      parser gap from masquerading as a genuine model miss (Finding 3).
    """
    stripped = themes_raw.strip()
    if not stripped or stripped.upper() == "NONE":
        return False, [], []

    graded_any = False
    dropped: list[str] = []
    near_miss: list[str] = []
    for line in stripped.splitlines():
        kind, name, _grade = classify_theme_line(line)
        if kind == "blank":
            continue
        if kind == "anomalous":
            near_miss.append(line.strip())
            continue
        graded_any = True  # "theme" or "off_taxonomy" is a recognizable grading
        if kind == "off_taxonomy":
            dropped.append(name)
    return not graded_any, dropped, near_miss


def summarize_rows(rows: list) -> dict:
    """Count the silent-failure modes in a run's prediction rows.

    - errors: rows with .error set (network/exception failures)
    - quality_parse_failures: quality was attempted (scores_raw captured) but
      parse_quality_scores returned None
    - theme_parse_failures: theme response was neither NONE nor parseable —
      garbage output masquerading as a confident "no themes"
    - dropped_theme_tokens: {token: count} of off-taxonomy theme names parse_themes
      silently dropped from otherwise-parsed responses
    - theme_near_miss_lines: {line: count} of non-blank lines parse_themes silently
      skipped (hyphenated/spaced name, misspelled grade) in an otherwise-parsed
      response — a parser gap that would otherwise read as a model miss (Finding 3)
    """
    counts = {
        "rows": len(rows),
        "errors": 0,
        "quality_parse_failures": 0,
        "theme_parse_failures": 0,
        "dropped_theme_tokens": {},
        "theme_near_miss_lines": {},
    }
    for r in rows:
        if getattr(r, "error", None):
            counts["errors"] += 1
            continue
        scores_raw = getattr(r, "scores_raw", None)
        if scores_raw is not None and getattr(r, "predicted_scores", None) is None:
            counts["quality_parse_failures"] += 1
        themes_raw = getattr(r, "themes_raw", None)
        if themes_raw is not None:
            is_garbage, dropped, near_miss = _theme_line_anomalies(themes_raw)
            if is_garbage:
                counts["theme_parse_failures"] += 1
            else:
                # A garbage response is already counted above and its lines are
                # the prose; only tally near-misses when the response otherwise
                # parsed, so the near-miss signal isn't double-reported.
                for line in near_miss:
                    counts["theme_near_miss_lines"][line] = (
                        counts["theme_near_miss_lines"].get(line, 0) + 1
                    )
            for token in dropped:
                counts["dropped_theme_tokens"][token] = (
                    counts["dropped_theme_tokens"].get(token, 0) + 1
                )
    return counts


def format_unreviewed_note(golden_set: list[GoldenNewsletter], mode: str) -> str | None:
    """Warn when extraction metrics will score unreviewed newsletters.

    An unreviewed (harvested-but-uncurated) newsletter has an empty golden
    story list, so every correct prediction on it counts as a false positive.
    Returns None outside extraction modes or when everything is reviewed.
    """
    if mode not in ("extraction", "all"):
        return None
    n_unreviewed = sum(1 for nl in golden_set if not nl.reviewed)
    if not n_unreviewed:
        return None
    return (
        f"Note: {n_unreviewed} of {len(golden_set)} evaluated newsletters are "
        "unreviewed (no curated story list) — extraction metrics count every "
        "predicted story on them as a false positive; treat precision as "
        "smoke-test only."
    )


def format_run_summary(rows: list) -> str:
    """Multi-line run summary: row count plus every silent-failure signal.

    A catastrophic parse failure (e.g. a --prompts variant that broke the
    response format for every story) must be visible here, not just via
    --report or grepping the results JSONL.
    """
    counts = summarize_rows(rows)
    near_miss_total = sum(counts["theme_near_miss_lines"].values())
    problems = []
    if counts["errors"]:
        problems.append(plural(counts["errors"], "error"))
    if counts["quality_parse_failures"]:
        problems.append(plural(counts["quality_parse_failures"], "quality parse failure"))
    if counts["theme_parse_failures"]:
        problems.append(plural(counts["theme_parse_failures"], "theme parse failure"))
    if near_miss_total:
        problems.append(plural(near_miss_total, "theme near-miss", "theme near-misses"))
    lines = [f"  Rows: {counts['rows']} ({', '.join(problems) if problems else 'no errors'})"]
    if counts["dropped_theme_tokens"]:
        detail = ", ".join(
            f"{tok} x{n}" for tok, n in sorted(counts["dropped_theme_tokens"].items())
        )
        lines.append(f"  Unrecognized theme labels dropped by the parser: {detail}")
    if counts["theme_near_miss_lines"]:
        detail = ", ".join(
            f'"{line}" x{n}'
            for line, n in sorted(counts["theme_near_miss_lines"].items())
        )
        lines.append(f"  Theme lines the parser skipped (near-misses): {detail}")
    return "\n".join(lines)


def merge_prompts_override(config: dict, override: dict) -> None:
    """Deep-merge only [newsletter.prompts.*] from *override* onto *config*, in place.

    A --prompts alt.toml may declare other sections, but only its newsletter prompt
    blocks are applied (everything else — model, recipient — is ignored) so a prompt
    variant file can't silently change the model or other run parameters. Must run
    BEFORE compute_prompt_hash so the recorded hash reflects the merged prompts.
    """
    prompts_override = override.get("newsletter", {}).get("prompts")
    if not prompts_override:
        return
    base_prompts = config.setdefault("newsletter", {}).setdefault("prompts", {})
    _deep_merge(base_prompts, prompts_override)


def build_classifier(config: dict, no_cache: bool) -> tuple[NewsletterClassifier, object]:
    """Build the LLM client (cached unless *no_cache*) and the classifier.

    Resolves the newsletter endpoint via daemon.resolve_newsletter_llm_endpoint()
    (falls back to the cloud endpoint) and builds one LLMClient from
    [newsletter.llm], mirroring daemon.run_daemon(). Returns (classifier, llm) so
    the caller can flush()/report cache stats on the wrapping client.
    """
    llm_cfg = config["newsletter"]["llm"]
    base_url, api_key = resolve_newsletter_llm_endpoint()
    inner = LLMClient(
        base_url=base_url,
        api_key=api_key,
        model=llm_cfg["model"],
        max_tokens=llm_cfg.get("max_tokens", 1024),
        temperature=llm_cfg.get("temperature", 0),
        timeout=llm_cfg.get("timeout", 60),
        extra_body=llm_cfg.get("extra_body"),
    )
    if no_cache:
        llm = inner
    else:
        cache_path = Path(__file__).parent / "cache" / "llm_cache.jsonl"
        llm = CachedLLMClient(inner, cache_path)
    classifier = NewsletterClassifier(cloud_llm=llm, config=config)
    return classifier, llm


def _split_rows(rows: list) -> tuple[list[StoryPrediction], list[ExtractionPrediction]]:
    """Split a mixed prediction row list into (story_preds, extraction_preds)."""
    story = [r for r in rows if isinstance(r, StoryPrediction)]
    extraction = [r for r in rows if isinstance(r, ExtractionPrediction)]
    return story, extraction


def maybe_report(
    meta, rows, report_enabled: bool, compare_to: str | None,
    verbose: bool = False, match_threshold: float = 0.6,
) -> None:
    """Print a metrics report (and optional comparison) for a just-finished run.

    A no-op unless --report or --compare-to was passed. Defers to
    evals.newsletter_report; prints a hint if it isn't importable so a missing
    report module never crashes a completed run. *verbose* and *match_threshold*
    forward the standalone report's knobs so run --report doesn't require a
    second newsletter_report invocation.
    """
    if not report_enabled and not compare_to:
        return
    try:
        from evals import newsletter_report
    except ImportError:
        print("Note: evals.newsletter_report is not available; skipping --report.",
              file=sys.stderr)
        return

    story_preds, extraction_preds = _split_rows(rows)
    metrics = newsletter_report.compute_all_metrics(
        story_preds, extraction_preds, match_threshold=match_threshold
    )
    if report_enabled:
        newsletter_report.print_report(
            meta, metrics, verbose=verbose,
            story_results=story_preds, extraction_results=extraction_preds,
            match_threshold=match_threshold,
        )
    if compare_to:
        compare_path = Path(compare_to)
        if not compare_path.exists():
            print(f"Error: --compare-to file not found: {compare_path}", file=sys.stderr)
            return
        try:
            meta_b, story_b, extraction_b = newsletter_report.load_results(compare_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            # Never crash a completed (paid) run over a bad comparison file —
            # e.g. --compare-to pointed at a .cot.jsonl sidecar by mistake.
            print(f"Error: could not read --compare-to file {compare_path}: {exc}",
                  file=sys.stderr)
            return
        metrics_b = newsletter_report.compute_all_metrics(
            story_b, extraction_b, match_threshold=match_threshold
        )
        # Prior run is Run A, the just-finished run is Run B (chronological order).
        newsletter_report.print_comparison(
            meta_b, metrics_b, meta, metrics,
            verbose=verbose, story1=story_b, story2=story_preds,
        )


def _load_toml_or_exit(path: Path, description: str) -> dict:
    """tomllib.load with a one-line error + exit 1 instead of a traceback."""
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        print(f"Error: {description} not found: {path}", file=sys.stderr)
    except tomllib.TOMLDecodeError as exc:
        print(f"Error: could not parse {description} {path}: {exc}", file=sys.stderr)
    sys.exit(1)


async def main(args: argparse.Namespace) -> None:
    # Load config (tomllib + env substitution), mirroring run_eval.
    config_path = (
        Path(args.config) if args.config
        else Path(__file__).parent.parent / "config.toml"
    )
    config = substitute_env_vars(_load_toml_or_exit(config_path, "config file"))

    # --prompts deep-merges alternate [newsletter.prompts.*] BEFORE prompt_hash and
    # classifier are computed, so the run measures (and records the hash of) the
    # merged prompts. --model overrides the newsletter model.
    if args.prompts:
        override = substitute_env_vars(
            _load_toml_or_exit(Path(args.prompts), "--prompts file")
        )
        merge_prompts_override(config, override)
    if args.model:
        config["newsletter"]["llm"]["model"] = args.model

    parallelism = args.parallelism if args.parallelism is not None else 1

    # Load golden set
    golden_path = Path(args.golden_set)
    try:
        golden_set, load_stats = load_golden_set(
            golden_path, reviewed_only=not args.include_unreviewed
        )
    except FileNotFoundError:
        print(f"Error: golden set file not found: {golden_path}", file=sys.stderr)
        sys.exit(1)
    print(format_load_summary(len(golden_set), load_stats, golden_path), file=sys.stderr)
    if not golden_set:
        sys.exit(1)
    unreviewed_note = format_unreviewed_note(golden_set, args.mode)
    if unreviewed_note:
        print(unreviewed_note, file=sys.stderr)

    # Stories graded in newsletter_grade (the story-label log) join the story
    # modes. Absent flag (older callers) or a missing file means none.
    extra_stories: list = []
    labels_path = getattr(args, "story_labels", None)
    if labels_path and args.mode != "extraction":
        try:
            extra_stories, label_stats = load_story_labels(
                Path(labels_path), {nl.thread_id for nl in golden_set}
            )
        except ValueError as exc:
            print(f"Error: cannot read the story-label log: {exc}", file=sys.stderr)
            sys.exit(1)
        if any(label_stats.values()):
            print(format_story_label_summary(label_stats, labels_path), file=sys.stderr)

    classifier, llm = build_classifier(config, args.no_cache)

    # Preflight: probe the newsletter endpoint once and fail fast on an
    # unreachable / name-mismatched one (skippable via --skip-preflight).
    if not args.skip_preflight:
        result = await llm.probe()
        if not result.ok:
            base = config["newsletter"]["llm"]
            # Surface probe()'s captured diagnosis (HTTP status / exception text /
            # 404 model-mismatch hint) instead of discarding it (Finding 1). All
            # wording lives on AvailabilityResult.detail() so this and run_eval's
            # preflight can't drift (Finding 2).
            detail = result.detail()
            detail_part = f"\n{detail}" if detail else ""
            print(
                f"Error: newsletter LLM '{base['model']}' not reachable at "
                f"{describe_endpoint()}.\n"
                "Check that the endpoint is up." + detail_part,
                file=sys.stderr,
            )
            sys.exit(1)

    try:
        if args.mode != "extraction":
            _pairs, _n_excluded, n_unlabeled = select_stories(golden_set)
            if n_unlabeled:
                print(
                    f"Skipping {plural(n_unlabeled, 'story', 'stories')} never labeled in the "
                    "TUI (no expected scores/themes — label with "
                    "python -m evals.newsletter_label)",
                    file=sys.stderr,
                )
        print(f"Running newsletter evaluation (mode={args.mode}, "
              f"parallelism={parallelism})...", file=sys.stderr)
        rows, thinking_entries = await run_evaluation(
            golden_set=golden_set,
            classifier=classifier,
            mode=args.mode,
            parallelism=parallelism,
            progress=True,
            extra_stories=extra_stories,
        )

        meta = build_meta(
            config=config,
            config_path=str(config_path),
            golden_path=str(golden_path),
            golden_set=golden_set,
            mode=args.mode,
            tag=args.tag or "",
            parallelism=parallelism,
            story_labels_path=str(labels_path or ""),
            story_label_count=len(extra_stories),
        )

        output_dir = Path(args.output_dir)
        output_path = build_output_path(output_dir, args.mode, args.tag or "", meta.run_id)
        write_results(output_path, meta, rows)
        sidecar_path = write_thinking_sidecar(output_path, thinking_entries)

        maybe_report(
            meta, rows, args.report, args.compare_to,
            verbose=getattr(args, "verbose", False),
            match_threshold=getattr(args, "match_threshold", 0.6),
        )

        print(f"\nResults written to {output_path}", file=sys.stderr)
        if sidecar_path:
            print(f"Chain-of-thought written to {sidecar_path}", file=sys.stderr)
        print(format_run_summary(rows), file=sys.stderr)
        if not args.no_cache and hasattr(llm, "hits"):
            total = llm.hits + llm.misses
            if total:
                print(f"  Cache: {llm.hits}/{total} hits ({llm.hits / total:.1%})",
                      file=sys.stderr)
    finally:
        if not args.no_cache and hasattr(llm, "flush"):
            llm.flush()


def cli():
    parser = argparse.ArgumentParser(description="Run newsletter classification evaluation")
    parser.add_argument("--golden-set", default="evals/newsletter_golden_set.jsonl",
                        help="Path to newsletter golden set JSONL")
    parser.add_argument("--story-labels", default="evals/newsletter_story_labels.jsonl",
                        help="Story-label log from newsletter_grade; its graded stories are "
                             "scored in the quality/themes modes (a missing file means none)")
    parser.add_argument("--config",
                        help="Path to config.toml (default: the repo-root config.toml, "
                             "regardless of CWD)")
    parser.add_argument("--output-dir", default="evals/newsletter_results/",
                        help="Output directory for results")
    parser.add_argument("--mode", choices=VALID_MODES, default="all",
                        help="Which outputs to evaluate")
    parser.add_argument("--tag", help="Tag for the results file name")
    parser.add_argument("--no-cache", action="store_true", help="Disable LLM response cache")
    parser.add_argument("--parallelism", type=int, default=None,
                        help="Concurrent evaluations (default: 1)")
    parser.add_argument("--include-unreviewed", action="store_true",
                        help="Also evaluate newsletters not yet reviewed (default: reviewed only)")
    parser.add_argument("--prompts", metavar="TOML",
                        help="Deep-merge [newsletter.prompts.*] from this TOML over the base config")
    parser.add_argument("--model", help="Override the newsletter LLM model name")
    parser.add_argument("--report", action="store_true",
                        help="Print a metrics report for this run after it completes")
    parser.add_argument("--compare-to", metavar="PATH",
                        help="After the run, print a comparison against this prior results file")
    parser.add_argument("--verbose", action="store_true",
                        help="With --report/--compare-to: per-story diffs and extraction detail")
    parser.add_argument("--match-threshold", type=float, default=0.6,
                        help="With --report/--compare-to: SequenceMatcher ratio threshold "
                             "for extraction matching (default 0.6)")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip the endpoint reachability check before evaluating")
    args = parser.parse_args()
    if args.parallelism is not None and args.parallelism < 1:
        # Semaphore(0) would make every task block forever: the run hangs
        # silently after preflight. Reject up front.
        parser.error(f"--parallelism must be >= 1 (got {args.parallelism})")
    # httpx logs one INFO line per request; at eval volume that drowns the
    # run's own progress output.
    quiet_http_logging()
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
