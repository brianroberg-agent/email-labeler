"""Compute and display metrics from newsletter evaluation results.

Usage:
    python -m evals.newsletter_report --results evals/newsletter_results/run.jsonl
    python -m evals.newsletter_report --compare run1.jsonl run2.jsonl
    python -m evals.newsletter_report --results-dir evals/newsletter_results/
    python -m evals.newsletter_report --results run.jsonl --verbose
    python -m evals.newsletter_report --results run.jsonl --format json
"""

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

from evals import plural
from evals.newsletter_schemas import (
    ExtractionPrediction,
    NewsletterRunMeta,
    StoryPrediction,
)
from evals.report import (
    compute_confusion_matrix,
    compute_precision_recall_f1,
    format_confusion_matrix,
    format_pct,
    format_per_class_table,
    format_table_row,
)
from newsletter import _VALID_THEMES, classify_theme_line

TIERS = ["excellent", "good", "fair", "poor"]
DIMENSIONS = ["simple", "concrete", "personal", "dynamic"]

# Display order for theme tables; the theme SET is single-sourced from
# newsletter._VALID_THEMES — a theme dropped there disappears here, and any
# theme added there is appended (sorted) even before this ordering is updated.
_THEME_DISPLAY_ORDER = [
    "scripture", "christlikeness", "church", "vocation_family", "disciple_making",
]
THEMES = [t for t in _THEME_DISPLAY_ORDER if t.upper() in _VALID_THEMES] + sorted(
    t.lower() for t in _VALID_THEMES if t.lower() not in _THEME_DISPLAY_ORDER
)


def _dimension_pairs(results: list[StoryPrediction], dim: str) -> list[tuple[int, int]]:
    """(expected, predicted) pairs for a dimension over stories where both are present."""
    pairs: list[tuple[int, int]] = []
    for r in results:
        if r.expected_scores is None or r.predicted_scores is None:
            continue
        if dim not in r.expected_scores or dim not in r.predicted_scores:
            continue
        pairs.append((r.expected_scores[dim], r.predicted_scores[dim]))
    return pairs


def _dimension_mean(
    results: list[StoryPrediction],
    per_pair,
) -> dict[str, float | None]:
    """Mean of per_pair(expected, predicted) per dimension; None with no pairs."""
    out: dict[str, float | None] = {}
    for dim in DIMENSIONS:
        pairs = _dimension_pairs(results, dim)
        out[dim] = sum(per_pair(e, p) for e, p in pairs) / len(pairs) if pairs else None
    return out


def compute_dimension_mae(results: list[StoryPrediction]) -> dict[str, float]:
    """Mean absolute error per dimension over stories with both scores present."""
    return _dimension_mean(results, lambda e, p: abs(e - p))


def compute_dimension_exact_match(results: list[StoryPrediction]) -> dict[str, float]:
    """Fraction of exact-match predictions per dimension."""
    return _dimension_mean(results, lambda e, p: e == p)


def _direction(pairs: list[tuple[int, int]]) -> dict:
    """Direction of (expected, predicted) pairs on an ordered scale.

    over/under are the fractions graded higher/lower than the label and signed
    is the mean of predicted - expected. All three are None with no pairs: an
    empty cell is "no data", not a rate of zero.
    """
    n = len(pairs)
    if not n:
        return {"n": 0, "over": None, "under": None, "signed": None}
    return {
        "n": n,
        "over": sum(p > e for e, p in pairs) / n,
        "under": sum(p < e for e, p in pairs) / n,
        "signed": sum(p - e for e, p in pairs) / n,
    }


def compute_dimension_direction(results: list[StoryPrediction]) -> dict:
    """Per dimension, how often the model graded higher or lower than the label.

    Agreement metrics (MAE, exact match) cannot tell over-grading from
    under-grading, and the rubric ranks over-grading as the costliest error
    (docs/newsletter-rubric.md section 1). Keys are the dimensions plus "all",
    which pools every dimension judgment.
    """
    out = {}
    pooled: list[tuple[int, int]] = []
    for dim in DIMENSIONS:
        pairs = _dimension_pairs(results, dim)
        pooled += pairs
        out[dim] = _direction(pairs)
    out["all"] = _direction(pooled)
    return out


_THEME_RANK = {None: 0, "present": 1, "emphasized": 2}
_THEME_GRADE_NAME = {None: "absent", "present": "present", "emphasized": "emphasized"}
# predicted_for_expected confusions worth naming: a wrong EMPHASIZED earns a
# Gmail label the story did not deserve (D14); the rest are its mirror images.
_THEME_CONFUSIONS = (
    "emphasized_for_present", "emphasized_for_absent", "present_for_absent",
    "present_for_emphasized", "absent_for_emphasized", "absent_for_present",
)


def compute_theme_direction(results: list[StoryPrediction], themes: list[str]) -> dict:
    """Per theme, how often the model's grade came out higher or lower than the
    label on Absent < Present < Emphasized, and which confusions those were
    (issue #55). Only rows whose theme call succeeded count. Keys are the themes
    plus "all", which pools every theme judgment.
    """
    scored = [r for r in results if _theme_scored(r)]
    out = {}
    pooled: list[tuple[int, int]] = []
    pooled_confusions = dict.fromkeys(_THEME_CONFUSIONS, 0)
    for theme in themes:
        pairs = []
        confusions = dict.fromkeys(_THEME_CONFUSIONS, 0)
        for r in scored:
            expected = _theme_grade(r.expected_themes, theme)
            predicted = _theme_grade(r.predicted_themes, theme)
            pairs.append((_THEME_RANK[expected], _THEME_RANK[predicted]))
            if expected != predicted:
                key = f"{_THEME_GRADE_NAME[predicted]}_for_{_THEME_GRADE_NAME[expected]}"
                confusions[key] += 1
                pooled_confusions[key] += 1
        pooled += pairs
        out[theme] = {**_direction(pairs), **confusions}
    out["all"] = {**_direction(pooled), **pooled_confusions}
    return out


def format_rate_delta(a: float | None, b: float | None) -> str:
    """Delta for a lower-is-better rate (graded too high / too low), as a percent."""
    if a is None or b is None:
        return "N/A"
    d = b - a
    sign = "+" if d >= 0 else ""
    label = " (improved)" if d < 0 else (" (worse)" if d > 0 else "")
    return f"{sign}{d:.1%}{label}"


# NB: a "within-1" dimension metric was dropped in the 3-value migration (#53):
# on a 1-3 scale the only pair NOT within 1 is a Poor<->Good confusion, so it was
# ~always 1.0 and added no signal beyond MAE (now 0-2) + exact-match.


def format_metric_delta(a: float | None, b: float | None) -> str:
    """Delta for a higher-is-better metric (accuracy, F1), formatted as a percent."""
    if a is None or b is None:
        return "N/A"
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1%}"


def format_mae_delta(a: float | None, b: float | None) -> str:
    """Delta for MAE (lower-is-better).

    Reports the raw signed change (b - a) and flags a decrease as an improvement.
    """
    if a is None or b is None:
        return "N/A"
    d = b - a
    sign = "+" if d >= 0 else ""
    label = " (improved)" if d < 0 else (" (worse)" if d > 0 else "")
    return f"{sign}{d:.2f}{label}"


def compute_all_metrics(
    story_results: list[StoryPrediction],
    extraction_results: list[ExtractionPrediction] | None = None,
    match_threshold: float = 0.6,
    mode: str = "all",
) -> dict:
    """Bundle every metric section for a single run.

    mode is the run's RunMeta.mode; it tells tier metrics whether errored rows
    count as quality failures (they don't in a themes-only run).
    """
    return {
        "story_count": len(story_results),
        "tier": compute_tier_metrics(story_results, mode),
        "dimension_mae": compute_dimension_mae(story_results),
        "dimension_exact": compute_dimension_exact_match(story_results),
        # Which way the disagreements run: over-grading is the costly direction.
        "dimension_direction": compute_dimension_direction(story_results),
        "theme_direction": compute_theme_direction(story_results, THEMES),
        # Primary, production-aligned: only Emphasized earns a Gmail label (#53).
        "themes": compute_multilabel_metrics(story_results, THEMES, positive="emphasized"),
        # Secondary: whether the theme was detected at all (Present or Emphasized).
        "themes_detection": compute_multilabel_metrics(story_results, THEMES, positive="detection"),
        "theme_anomalies": theme_parse_anomalies(story_results),
        "extraction": compute_extraction_metrics(
            extraction_results or [], threshold=match_threshold
        ),
    }


def compute_extraction_metrics(
    results: list[ExtractionPrediction],
    threshold: float = 0.6,
) -> dict:
    """Extraction metrics aggregated over newsletters.

    Per newsletter: precision = matched/predicted, recall = matched/golden via
    greedy one-to-one match_stories. A newsletter with 0 predicted and 0 golden
    stories is a correct abstention (precision = recall = 1.0). Aggregate as micro
    (pooled counts) and macro (mean of per-newsletter precision/recall). All
    aggregates are None when there are no scored newsletters, so empty sections
    render as N/A rather than a fake 0.0%. Rows with error set are excluded
    from scoring but counted under "errors"/"error_threads" so a failed run is
    visible in the report (mirroring the tier section).
    """
    total_matched = total_pred = total_gold = 0
    per_nl_precision: list[float] = []
    per_nl_recall: list[float] = []
    per_newsletter: list[dict] = []
    error_threads: list[str] = []

    for r in results:
        if r.error is not None:
            error_threads.append(r.thread_id)
            continue
        matched, n_pred, n_gold = match_stories(
            r.predicted_stories, r.golden_stories, threshold=threshold
        )
        total_matched += matched
        total_pred += n_pred
        total_gold += n_gold
        # 0/0 = correct abstention, not a zero: the model was right to predict
        # nothing for a story-less newsletter.
        p = matched / n_pred if n_pred > 0 else (1.0 if n_gold == 0 else 0.0)
        rec = matched / n_gold if n_gold > 0 else (1.0 if n_pred == 0 else 0.0)
        per_nl_precision.append(p)
        per_nl_recall.append(rec)
        per_newsletter.append({
            "thread_id": r.thread_id,
            "matched": matched,
            "predicted": n_pred,
            "golden": n_gold,
            "precision": p,
            "recall": rec,
        })

    if not per_newsletter:
        return {
            "micro_precision": None,
            "micro_recall": None,
            "micro_f1": None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "count": 0,
            "errors": len(error_threads),
            "error_threads": error_threads,
            "per_newsletter": [],
        }

    micro_p = total_matched / total_pred if total_pred > 0 else 1.0
    micro_r = total_matched / total_gold if total_gold > 0 else 1.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    )
    macro_p = sum(per_nl_precision) / len(per_nl_precision)
    macro_r = sum(per_nl_recall) / len(per_nl_recall)
    macro_f1 = (
        2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0.0
    )

    return {
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "count": len(per_newsletter),
        "errors": len(error_threads),
        "error_threads": error_threads,
        "per_newsletter": per_newsletter,
    }


def _theme_scored(r: StoryPrediction) -> bool:
    """Whether a row's own theme call succeeded (independent of quality parsing).

    A row participates in theme metrics iff it has no error and the theme
    response exists (themes_raw recorded, or — for legacy rows without
    themes_raw — a non-empty parsed prediction). A quality-parse failure must
    NOT drop a story whose themes parsed fine.
    """
    if r.error is not None:
        return False
    return r.themes_raw is not None or bool(r.predicted_themes)


def _theme_grade(themes_val, theme: str) -> str | None:
    """A theme's stored grade (theme -> "present"/"emphasized"), or None if absent."""
    return (themes_val or {}).get(theme)


def _theme_grade_diffs(before, after) -> list[str]:
    """Grade-aware per-theme diff lines between two theme dicts.

    For each theme in the union of keys, a missing key reads as "absent"; when
    the grade changed, emit ``"<theme>: <before>-><after>"`` (themes sorted).
    Unchanged themes are omitted. Verbose disagreement/flip output uses this so a
    grade-only change (e.g. present -> emphasized) — invisible to a key-set
    compare but scored as an FP/FN by the Emphasized metric — is still shown.
    """
    before = before or {}
    after = after or {}
    parts: list[str] = []
    for theme in sorted(set(before) | set(after)):
        b = before.get(theme, "absent")
        a = after.get(theme, "absent")
        if b != a:
            parts.append(f"{theme}: {b}->{a}")
    return parts


# What counts as a positive theme label for the multilabel metric (issue #53):
_POSITIVE_TESTS = {
    "emphasized": lambda g: g == "emphasized",              # what earns a Gmail label
    "detection": lambda g: g in ("present", "emphasized"),  # theme present at all
}


def _positive_theme_set(themes_val, themes: list[str], is_pos) -> set[str]:
    return {t for t in themes if is_pos(_theme_grade(themes_val, t))}


def compute_multilabel_metrics(
    results: list[StoryPrediction],
    themes: list[str],
    positive: str = "emphasized",
) -> dict:
    """Multi-label theme metrics at a chosen positive threshold (issue #53).

    *positive* selects what counts as a positive label per theme:
    - ``"emphasized"`` (default, PRIMARY): only Emphasized — this is exactly what
      the daemon labels in Gmail, so it is the production-aligned headline metric.
    - ``"detection"``: Present OR Emphasized — a secondary "did we notice the
      theme at all" signal.

    Rows whose own theme call failed (see _theme_scored) are excluded; a
    quality-parse failure alone is not. Returns per-theme P/R/F1, micro-F1,
    macro-F1, and exact-set-match (fraction where the positive expected set ==
    the positive predicted set). Aggregates are None when no rows were scored, so
    empty sections render as N/A rather than a fake 0.0%.
    """
    is_pos = _POSITIVE_TESTS[positive]
    scored = [r for r in results if _theme_scored(r)]

    if not scored:
        return {
            "per_theme": {
                t: {"precision": None, "recall": None, "f1": None} for t in themes
            },
            "micro_precision": None,
            "micro_recall": None,
            "micro_f1": None,
            "macro_f1": None,
            "exact_set_match": None,
            "count": 0,
        }

    per_theme: dict[str, dict[str, float]] = {}
    total_tp = total_fp = total_fn = 0
    for theme in themes:
        tp = fp = fn = 0
        for r in scored:
            exp = is_pos(_theme_grade(r.expected_themes, theme))
            pred = is_pos(_theme_grade(r.predicted_themes, theme))
            if exp and pred:
                tp += 1
            elif pred and not exp:
                fp += 1
            elif exp and not pred:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_theme[theme] = {"precision": precision, "recall": recall, "f1": f1}
        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    )
    macro_f1 = sum(per_theme[t]["f1"] for t in themes) / len(themes) if themes else 0.0

    exact = sum(
        1 for r in scored
        if _positive_theme_set(r.expected_themes, themes, is_pos)
        == _positive_theme_set(r.predicted_themes, themes, is_pos)
    )
    exact_set_match = exact / len(scored) if scored else None

    return {
        "per_theme": per_theme,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "exact_set_match": exact_set_match,
        "count": len(scored),
    }


def _quality_attempted(r: StoryPrediction, mode: str = "all") -> bool:
    """Whether the quality call ran (or failed) for this row.

    A --mode themes run never fires the quality call: error=None AND
    scores_raw=None AND predicted_scores=None means "never attempted", which
    must be skipped — not counted as a parse failure. scores_raw captured (or
    an error, or parsed scores) means quality was genuinely attempted.

    In a themes-only run (mode="themes", from the run's RunMeta) an error is a
    *theme* call failure, so it does not count as a quality attempt either —
    only actual quality evidence (scores_raw / predicted_scores) does.
    """
    if mode == "themes":
        return r.scores_raw is not None or r.predicted_scores is not None
    return (
        r.error is not None
        or r.scores_raw is not None
        or r.predicted_scores is not None
    )


def compute_tier_metrics(results: list[StoryPrediction], mode: str = "all") -> dict:
    """Tier metrics over the 4 tier classes.

    Rows where quality was attempted but predicted_scores is None are parse/
    network failures: counted as errors and excluded from the confusion matrix
    / accuracy. Their story_ids are returned under "error_stories" so failures
    are identifiable in the report. Rows where quality was never attempted
    (e.g. a --mode themes run: error=None, scores_raw=None — or, given
    mode="themes", any errored row) are skipped entirely — neither scored nor
    errors.
    """
    attempted = [r for r in results if _quality_attempted(r, mode)]
    errors = [r for r in attempted if r.predicted_scores is None]
    valid = [r for r in attempted if r.predicted_scores is not None]

    matrix = compute_confusion_matrix(valid, "expected_tier", "predicted_tier", TIERS)
    prf = compute_precision_recall_f1(matrix, TIERS)

    scored = [r for r in valid if r.expected_tier in TIERS and r.predicted_tier in TIERS]
    correct = sum(1 for r in scored if r.expected_tier == r.predicted_tier)
    accuracy = correct / len(scored) if scored else None

    return {
        "accuracy": accuracy,
        "confusion_matrix": matrix,
        "per_class": prf,
        "count": len(scored),
        "errors": len(errors),
        "error_stories": [r.story_id for r in errors],
    }


def _normalize(story: dict) -> str:
    """Lowercase, whitespace-collapse the story's text for matching.

    Text is the ground truth curated in the labeling TUI — extraction matching
    is on the extracted spans (text) only.
    """
    raw = story.get("text") or ""
    return re.sub(r"\s+", " ", raw).strip().lower()


def match_stories_detailed(
    predicted: list[dict],
    golden: list[dict],
    threshold: float = 0.6,
) -> dict:
    """Greedy one-to-one matching with full pair/unmatched detail.

    Uses difflib.SequenceMatcher ratio on normalized (lowercased,
    whitespace-collapsed) story text. Highest-ratio candidate pairs are
    assigned first; each predicted and each golden story matches at most once.

    Returns:
        {"matched": [{"pred_index", "gold_index", "ratio"}, ...],
         "unmatched_predicted": [pred indices],
         "unmatched_golden": [gold indices]}
    """
    pred_norm = [_normalize(p) for p in predicted]
    gold_norm = [_normalize(g) for g in golden]

    # Score every predicted/golden pair, then assign greedily by descending ratio.
    pairs: list[tuple[float, int, int]] = []
    for pi, pn in enumerate(pred_norm):
        for gi, gn in enumerate(gold_norm):
            ratio = difflib.SequenceMatcher(None, pn, gn).ratio()
            if ratio >= threshold:
                pairs.append((ratio, pi, gi))

    pairs.sort(key=lambda x: x[0], reverse=True)

    used_pred: set[int] = set()
    used_gold: set[int] = set()
    matched: list[dict] = []
    for ratio, pi, gi in pairs:
        if pi in used_pred or gi in used_gold:
            continue
        used_pred.add(pi)
        used_gold.add(gi)
        matched.append({"pred_index": pi, "gold_index": gi, "ratio": ratio})

    return {
        "matched": matched,
        "unmatched_predicted": [i for i in range(len(predicted)) if i not in used_pred],
        "unmatched_golden": [i for i in range(len(golden)) if i not in used_gold],
    }


def match_stories(
    predicted: list[dict],
    golden: list[dict],
    threshold: float = 0.6,
) -> tuple[int, int, int]:
    """Greedy one-to-one match count (see match_stories_detailed).

    Returns:
        (matched, n_predicted, n_golden)
    """
    detail = match_stories_detailed(predicted, golden, threshold=threshold)
    return len(detail["matched"]), len(predicted), len(golden)


def theme_parse_anomalies(results: list[StoryPrediction]) -> list[dict]:
    """Detect theme responses that parse_themes silently normalized or skipped.

    Routes every line through the shared ``newsletter.classify_theme_line`` so
    this report diagnostic and its run-side twin
    (``newsletter_run._theme_line_anomalies``) share one source of the response
    format and cannot drift (Finding 2). At most one anomaly per story; every
    anomaly dict carries both an ``invalid_tokens`` and a ``near_miss_lines``
    list. Kinds, by precedence:

    - "empty_parse": themes_raw is non-empty and not NONE, yet no line is a
      recognizable theme grading — the model emitted unusable prose.
    - "invalid_tokens": at least one line grades an off-taxonomy theme NAME
      (e.g. FELLOWSHIP) that parse_themes silently drops.
    - "near_miss": the response otherwise parsed, but a non-blank line is not a
      recognizable grading (hyphenated/spaced name, misspelled grade) and
      parse_themes silently skipped it — a parser gap that would otherwise read
      as a genuine model miss (Finding 3).

    An all-ABSENT response is a valid empty result, not an anomaly. Rows without
    themes_raw (legacy files, error rows) are skipped.
    """
    anomalies: list[dict] = []
    for r in results:
        if r.error is not None or r.themes_raw is None:
            continue
        raw = r.themes_raw.strip()
        if not raw or raw.upper() == "NONE":
            continue
        graded_any = False
        invalid: list[str] = []
        near_miss: list[str] = []
        for line in raw.splitlines():
            kind, name, _grade = classify_theme_line(line)
            if kind == "blank":
                continue
            if kind == "anomalous":
                near_miss.append(line.strip())
                continue
            graded_any = True
            if kind == "off_taxonomy":
                invalid.append(name)
        if not graded_any:
            kind = "empty_parse"  # prose subsumes the near-miss lines
        elif invalid:
            kind = "invalid_tokens"
        elif near_miss:
            kind = "near_miss"
        else:
            continue  # fully parsed — nothing dropped or skipped
        anomalies.append({
            "story_id": r.story_id,
            "kind": kind,
            "invalid_tokens": invalid,
            "near_miss_lines": near_miss if graded_any else [],
            "themes_raw": r.themes_raw,
        })
    return anomalies


# --- Loading ---

def load_results(
    path: Path,
) -> tuple[NewsletterRunMeta, list[StoryPrediction], list[ExtractionPrediction]]:
    """Load run metadata and predictions from a newsletter results JSONL.

    Dispatches rows on the "type" discriminator.
    """
    meta = None
    story_preds: list[StoryPrediction] = []
    extraction_preds: list[ExtractionPrediction] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            t = d.get("type")
            if t == "run_meta":
                meta = NewsletterRunMeta.from_dict(d)
            elif t == "story_prediction":
                story_preds.append(StoryPrediction.from_dict(d))
            elif t == "extraction_prediction":
                extraction_preds.append(ExtractionPrediction.from_dict(d))
    if meta is None:
        raise ValueError(f"No run_meta found in {path}")
    return meta, story_preds, extraction_preds


def _text_excerpt(text: str) -> str:
    """Whitespace-collapse *text* and truncate to a short label (~60 chars)."""
    collapsed = re.sub(r"\s+", " ", text or "").strip()
    return collapsed[:57] + "..." if len(collapsed) > 60 else collapsed


def load_story_excerpts(golden_set_path: str) -> dict[str, str]:
    """Best-effort story_id -> text-excerpt map from the run's golden set file.

    Results rows carry only story_ids; the story text lives in the golden set,
    so verbose output labels a story by the first words of its text. Any failure
    (moved/deleted file, malformed line) degrades to {} so verbose output falls
    back to bare story_ids.
    """
    excerpts: dict[str, str] = {}
    try:
        with open(golden_set_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                for s in d.get("stories", []):
                    if s.get("story_id") and s.get("text"):
                        excerpts[s["story_id"]] = _text_excerpt(s["text"])
    except (OSError, json.JSONDecodeError):
        return {}
    return excerpts


# --- Formatters ---

def _format_dim_table(
    mae: dict[str, float | None],
    exact: dict[str, float | None],
) -> str:
    widths = [max(len("Dimension"), *(len(d) for d in DIMENSIONS)), 10, 12]
    lines = []
    lines.append("  " + format_table_row(["Dimension", "MAE", "Exact"], widths))
    lines.append("  " + "-" * sum(w + 2 for w in widths))
    for dim in DIMENSIONS:
        m = mae.get(dim)
        mae_s = "N/A" if m is None else f"{m:.2f}"
        lines.append("  " + format_table_row(
            [dim, mae_s, format_pct(exact.get(dim))],
            widths,
        ))
    return "\n".join(lines)


def _format_signed(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.2f}"


def _format_direction_tables(dimension: dict, theme: dict) -> str:
    """Too-high / too-low rates per dimension and per theme, with n beside each."""
    name_width = max(len("Dimension"), len("all"), *(len(t) for t in THEMES))
    widths = [name_width, 4, 10, 10, 8]
    lines = ["  " + format_table_row(["Dimension", "n", "Too high", "Too low", "Signed"], widths),
             "  " + "-" * sum(w + 2 for w in widths)]
    for dim in [*DIMENSIONS, "all"]:
        d = dimension[dim]
        lines.append("  " + format_table_row(
            [dim, str(d["n"]), format_pct(d["over"]), format_pct(d["under"]),
             _format_signed(d["signed"])], widths))
    widths = [name_width, 4, 10, 10, 18, 18]
    lines += ["", "  " + format_table_row(
        ["Theme", "n", "Too high", "Too low", "Emph. for Present", "Emph. for Absent"], widths),
        "  " + "-" * sum(w + 2 for w in widths)]
    for name in [*THEMES, "all"]:
        t = theme[name]
        lines.append("  " + format_table_row(
            [name, str(t["n"]), format_pct(t["over"]), format_pct(t["under"]),
             str(t["emphasized_for_present"]), str(t["emphasized_for_absent"])], widths))
    return "\n".join(lines)


def _format_theme_table(themes: dict) -> str:
    widths = [max(len(t) for t in THEMES), 10, 10, 10]
    lines = []
    lines.append("  " + format_table_row(["Theme", "Precision", "Recall", "F1"], widths))
    lines.append("  " + "-" * sum(w + 2 for w in widths))
    for t in THEMES:
        m = themes["per_theme"][t]
        lines.append("  " + format_table_row(
            [t, format_pct(m["precision"]), format_pct(m["recall"]), format_pct(m["f1"])],
            widths,
        ))
    return "\n".join(lines)


def print_report(
    meta: NewsletterRunMeta,
    metrics: dict,
    verbose: bool = False,
    story_results: list[StoryPrediction] | None = None,
    extraction_results: list[ExtractionPrediction] | None = None,
    match_threshold: float = 0.6,
) -> None:
    """Print a formatted single-run report to stdout.

    match_threshold must be the same threshold the metrics were computed with,
    so verbose extraction diffs agree with the headline numbers.
    """
    print(f"\n{'=' * 60}")
    print(f"Newsletter Evaluation Report: {meta.run_id[:8]}")
    print(f"{'=' * 60}")
    print(f"  Timestamp:    {meta.timestamp}")
    print(f"  Config:       {meta.config_path} ({meta.config_hash})")
    print(f"  Model:        {meta.newsletter_model}")
    print(f"  Mode:         {meta.mode}")
    print(f"  Prompt hash:  {meta.prompt_hash}")
    if meta.tag:
        print(f"  Tag:          {meta.tag}")
    print(f"  Golden set:   {meta.golden_set_path}")
    print(f"  Newsletters:  {meta.golden_set_count}")
    stories_line = f"  Stories:      {meta.story_count}"
    if meta.story_label_count:
        curated = meta.story_count - meta.story_label_count
        stories_line += (
            f" ({curated} curated, {meta.story_label_count} from the story-label log)"
        )
    print(stories_line)

    # Sections with zero underlying predictions are omitted (mirroring the tier
    # section) instead of rendering as a fake 0.0%.
    has_story_rows = bool(metrics.get("story_count", len(story_results or [])))

    tier = metrics["tier"]
    if tier["count"] or tier["errors"]:
        print("\n--- Tier Classification (4-class) ---")
        print(f"  Accuracy: {format_pct(tier['accuracy'])} "
              f"({plural(tier['count'], 'story', 'stories')}, "
              f"{plural(tier['errors'], 'error')})")
        if tier["error_stories"]:
            print(f"  Failed stories (quality parse/network): "
                  f"{', '.join(tier['error_stories'])}")
        print("\n  Confusion matrix:")
        print(format_confusion_matrix(tier["confusion_matrix"], TIERS))
        print("\n  Per-class metrics:")
        print(format_per_class_table(tier["per_class"], TIERS))

    if has_story_rows:
        print("\n--- Quality Dimensions ---")
        print(_format_dim_table(
            metrics["dimension_mae"], metrics["dimension_exact"],
        ))

        themes = metrics["themes"]
        print("\n--- Themes (Emphasized — what gets labeled) ---")
        print(f"  Micro-F1: {format_pct(themes['micro_f1'])}   "
              f"Macro-F1: {format_pct(themes['macro_f1'])}   "
              f"Exact-set match: {format_pct(themes['exact_set_match'])}")
        detection = metrics.get("themes_detection")
        if detection:
            print(f"  Detection (≥Present) Micro-F1: {format_pct(detection['micro_f1'])}")
        anomalies = metrics.get("theme_anomalies", [])
        if anomalies:
            print(f"  Parse anomalies: {plural(len(anomalies), 'story', 'stories')} "
                  f"(invalid/unparseable theme output; --verbose shows raw)")
        print()
        print(_format_theme_table(themes))

        if "dimension_direction" in metrics:
            print("\n--- Grading direction (too high is the costly error) ---")
            print(_format_direction_tables(
                metrics["dimension_direction"], metrics["theme_direction"]))

    extraction = metrics["extraction"]
    if extraction["count"] or extraction.get("errors"):
        print("\n--- Extraction ---")
        print(f"  ({plural(extraction['count'], 'newsletter')}, "
              f"{plural(extraction.get('errors', 0), 'error')}, "
              f"match threshold {match_threshold})")
        if extraction.get("error_threads"):
            print(f"  Failed newsletters (extraction call): "
                  f"{', '.join(extraction['error_threads'])}")
        print(f"  Micro:  P={format_pct(extraction['micro_precision'])} "
              f"R={format_pct(extraction['micro_recall'])} "
              f"F1={format_pct(extraction['micro_f1'])}")
        print(f"  Macro:  P={format_pct(extraction['macro_precision'])} "
              f"R={format_pct(extraction['macro_recall'])} "
              f"F1={format_pct(extraction['macro_f1'])}")

    if verbose:
        _print_verbose_single(
            story_results or [],
            extraction_results or [],
            match_threshold=match_threshold,
            excerpts=load_story_excerpts(meta.golden_set_path),
            anomalies=metrics.get("theme_anomalies", []),
        )

    print()


def _story_label(story: dict) -> str:
    """Human-readable story identifier: a text excerpt (first words)."""
    return _text_excerpt(story.get("text") or "")


def _print_verbose_single(
    story_results: list[StoryPrediction],
    extraction_results: list[ExtractionPrediction],
    match_threshold: float = 0.6,
    excerpts: dict[str, str] | None = None,
    anomalies: list[dict] | None = None,
) -> None:
    excerpts = excerpts or {}

    def label(story_id: str) -> str:
        excerpt = excerpts.get(story_id)
        return f"{story_id} ({excerpt})" if excerpt else story_id

    print("\n--- Story Disagreements ---")
    # A row is compared field-by-field on whatever DID parse: tiers only when
    # quality parsed, themes only when the theme call succeeded — so a quality
    # parse failure can't hide a theme disagreement.
    disagreements = []
    for r in story_results:
        if r.error is not None:
            continue
        tier_diff = r.predicted_scores is not None and r.expected_tier != r.predicted_tier
        theme_diff = _theme_scored(r) and (
            (r.expected_themes or {}) != (r.predicted_themes or {})
        )
        if tier_diff or theme_diff:
            disagreements.append((r, tier_diff, theme_diff))
    if not disagreements:
        print("  None!")
    for r, tier_diff, theme_diff in disagreements:
        parts = [f"  {label(r.story_id)}:"]
        if tier_diff:
            parts.append(f"tier={r.expected_tier}->{r.predicted_tier}")
        if theme_diff:
            parts.append("themes: " + ", ".join(
                _theme_grade_diffs(r.expected_themes, r.predicted_themes)
            ))
        print(" ".join(parts))

    # "Never attempted" rows (e.g. --mode themes: no error, no scores_raw) are
    # not failures — only attempted-but-unparsed or errored rows are listed.
    failures = [
        r for r in story_results
        if _quality_attempted(r) and r.predicted_scores is None
    ]
    if failures:
        print("\n--- Parse/Network Failures ---")
        for r in failures:
            print(f"  {label(r.story_id)}:")
            if r.error is not None:
                print(f"    error: {r.error}")
            elif r.scores_raw is not None:
                raw = "\n".join(f"    | {ln}" for ln in r.scores_raw.splitlines())
                print("    quality response failed to parse; raw:")
                print(raw)
            else:
                print("    quality response missing (no raw captured)")

    if anomalies:
        print("\n--- Theme Parse Anomalies ---")
        for a in anomalies:
            if a["kind"] == "empty_parse":
                what = "unparseable output (parsed to [])"
            else:
                parts = []
                if a["invalid_tokens"]:
                    parts.append(f"invalid tokens dropped: {', '.join(a['invalid_tokens'])}")
                if a.get("near_miss_lines"):
                    parts.append(
                        f"near-miss lines skipped: {', '.join(a['near_miss_lines'])}"
                    )
                what = "; ".join(parts)
            print(f"  {label(a['story_id'])}: {what}")
            raw = "\n".join(f"    | {ln}" for ln in a["themes_raw"].splitlines())
            print(raw)

    if extraction_results:
        print(f"\n--- Extraction Diffs (per newsletter, threshold {match_threshold}) ---")
        for r in extraction_results:
            if r.error is not None:
                print(f"  {r.thread_id}: ERROR {r.error}")
                continue
            detail = match_stories_detailed(
                r.predicted_stories, r.golden_stories, threshold=match_threshold
            )
            matched = detail["matched"]
            n_pred, n_gold = len(r.predicted_stories), len(r.golden_stories)
            print(f"  {r.thread_id}: matched={len(matched)} "
                  f"predicted={n_pred} golden={n_gold}")
            if len(matched) == n_pred == n_gold:
                continue  # perfect match: counts line is enough
            for pair in matched:
                p = r.predicted_stories[pair["pred_index"]]
                g = r.golden_stories[pair["gold_index"]]
                print(f"    matched ({pair['ratio']:.2f}): "
                      f"'{_story_label(p)}' ~ '{_story_label(g)}'")
            for i in detail["unmatched_predicted"]:
                print(f"    unmatched predicted: '{_story_label(r.predicted_stories[i])}'")
            for i in detail["unmatched_golden"]:
                print(f"    unmatched golden:    '{_story_label(r.golden_stories[i])}'")


def print_comparison(
    meta1: NewsletterRunMeta, metrics1: dict,
    meta2: NewsletterRunMeta, metrics2: dict,
    verbose: bool = False,
    story1: list[StoryPrediction] | None = None,
    story2: list[StoryPrediction] | None = None,
) -> None:
    """Print a side-by-side comparison of two newsletter runs.

    Sections where a run has zero scored items render as N/A (not 0.0%), so a
    mode-mismatched comparison can't fake a -100% regression.
    """
    print(f"\n{'=' * 60}")
    print("Newsletter Comparison Report")
    print(f"{'=' * 60}")
    print(f"  Run A: {meta1.run_id[:8]}  mode={meta1.mode}  model={meta1.newsletter_model}  "
          f"prompt_hash={meta1.prompt_hash or 'n/a'}  tag={meta1.tag or '-'}")
    print(f"  Run B: {meta2.run_id[:8]}  mode={meta2.mode}  model={meta2.newsletter_model}  "
          f"prompt_hash={meta2.prompt_hash or 'n/a'}  tag={meta2.tag or '-'}")
    if meta1.mode != meta2.mode:
        print(f"\n  WARNING: run modes differ ({meta1.mode} vs {meta2.mode}); "
              "sections absent from one run show N/A and deltas are not comparable.")

    widths = [22, 12, 12, 16]

    def header(title: str) -> None:
        print(f"\n--- {title} ---")
        print("  " + format_table_row(["Metric", "Run A", "Run B", "Delta"], widths))
        print("  " + "-" * sum(w + 2 for w in widths))

    def pct_row(label: str, a: float | None, b: float | None) -> None:
        print("  " + format_table_row(
            [label, format_pct(a), format_pct(b), format_metric_delta(a, b)], widths
        ))

    # Tier — per-class F1 is only meaningful when the run scored tier rows.
    ta, tb = metrics1["tier"], metrics2["tier"]
    header("Tier")
    pct_row("Accuracy", ta["accuracy"], tb["accuracy"])
    for cls in TIERS:
        a_f1 = ta["per_class"][cls]["f1"] if ta["count"] else None
        b_f1 = tb["per_class"][cls]["f1"] if tb["count"] else None
        pct_row(f"  {cls} F1", a_f1, b_f1)

    # Dimensions (MAE, lower is better)
    header("Dimension MAE (lower=better)")
    for dim in DIMENSIONS:
        a = metrics1["dimension_mae"].get(dim)
        b = metrics2["dimension_mae"].get(dim)
        a_s = "N/A" if a is None else f"{a:.2f}"
        b_s = "N/A" if b is None else f"{b:.2f}"
        print("  " + format_table_row([dim, a_s, b_s, format_mae_delta(a, b)], widths))

    # Direction: is the model grading too high (the costly error) less often?
    da, db = metrics1.get("dimension_direction"), metrics2.get("dimension_direction")
    ta_dir, tb_dir = metrics1.get("theme_direction"), metrics2.get("theme_direction")
    if da and db and ta_dir and tb_dir:
        for title, key in (("Graded too high (lower=better)", "over"),
                           ("Graded too low (lower=better)", "under")):
            header(title)
            for label, a_cell, b_cell in (
                [(dim, da[dim], db[dim]) for dim in DIMENSIONS]
                + [("all dimensions", da["all"], db["all"]),
                   ("all themes", ta_dir["all"], tb_dir["all"])]
            ):
                a, b = a_cell[key], b_cell[key]
                print("  " + format_table_row(
                    [label, format_pct(a), format_pct(b), format_rate_delta(a, b)], widths))

    # Themes — primary Emphasized rows, then the secondary detection (>=Present)
    # rows so a detection-level regression between runs is visible here too
    # (print_report shows the detection line for each single run).
    tha, thb = metrics1["themes"], metrics2["themes"]
    dta, dtb = metrics1["themes_detection"], metrics2["themes_detection"]
    header("Themes")
    pct_row("Micro-F1", tha["micro_f1"], thb["micro_f1"])
    pct_row("Macro-F1", tha["macro_f1"], thb["macro_f1"])
    pct_row("Detection Micro-F1", dta["micro_f1"], dtb["micro_f1"])
    pct_row("Detection Macro-F1", dta["macro_f1"], dtb["macro_f1"])

    # Extraction
    ea, eb = metrics1["extraction"], metrics2["extraction"]
    header("Extraction")
    pct_row("Micro-F1", ea["micro_f1"], eb["micro_f1"])
    pct_row("Macro-F1", ea["macro_f1"], eb["macro_f1"])

    if verbose and story1 is not None and story2 is not None:
        _print_verbose_compare(story1, story2)

    print()


def _print_verbose_compare(
    story1: list[StoryPrediction],
    story2: list[StoryPrediction],
) -> None:
    print("\n--- Per-story Flips (A -> B) ---")
    r2_by_id = {r.story_id: r for r in story2}
    flips = []
    for r1 in story1:
        r2 = r2_by_id.get(r1.story_id)
        if r2 is None:
            continue
        parts = []
        # Tier flips are only meaningful when both runs' quality calls parsed;
        # theme flips only need both rows' theme side attempted (a --mode
        # themes run never has predicted_scores but can still flip themes).
        if r1.predicted_scores is not None and r2.predicted_scores is not None:
            if r1.predicted_tier != r2.predicted_tier:
                a_right = r1.predicted_tier == r1.expected_tier
                b_right = r2.predicted_tier == r2.expected_tier
                flag = ""
                if a_right and not b_right:
                    flag = " [regression]"
                elif b_right and not a_right:
                    flag = " [improvement]"
                parts.append(f"tier: {r1.predicted_tier}->{r2.predicted_tier} "
                             f"(expected {r1.expected_tier}){flag}")
        if _theme_scored(r1) and _theme_scored(r2):
            if (r1.predicted_themes or {}) != (r2.predicted_themes or {}):
                parts.append("themes: " + ", ".join(
                    _theme_grade_diffs(r1.predicted_themes, r2.predicted_themes)
                ))
        if parts:
            flips.append((r1.story_id, parts))
    if not flips:
        print("  None!")
    for sid, parts in flips:
        print(f"  {sid}: {', '.join(parts)}")


def report_as_json(meta: NewsletterRunMeta, metrics: dict) -> None:
    print(json.dumps({"meta": meta.to_dict(), "metrics": metrics}, indent=2))


def comparison_as_json(
    meta1: NewsletterRunMeta, metrics1: dict,
    meta2: NewsletterRunMeta, metrics2: dict,
) -> None:
    print(json.dumps({
        "run_a": {"meta": meta1.to_dict(), "metrics": metrics1},
        "run_b": {"meta": meta2.to_dict(), "metrics": metrics2},
    }, indent=2))


def build_trend_rows(
    results_dir: Path, match_threshold: float = 0.6,
) -> tuple[list[dict], list[str]]:
    """Summary row per results file, sorted chronologically by run timestamp.

    Returns (rows, errors): unreadable files land in errors as one-line
    strings instead of aborting the whole trend.
    """
    files = sorted(results_dir.glob("*.jsonl"))
    files = [f for f in files if not f.name.endswith(".cot.jsonl")]

    rows: list[dict] = []
    errors: list[str] = []
    for f in files:
        try:
            meta, story_preds, extraction_preds = load_results(f)
            metrics = compute_all_metrics(story_preds, extraction_preds,
                                          match_threshold, mode=meta.mode)
            rows.append({
                "file": f.name,
                "run_id": meta.run_id,
                "timestamp": meta.timestamp,
                "mode": meta.mode,
                "tag": meta.tag,
                "prompt_hash": meta.prompt_hash,
                "tier_accuracy": metrics["tier"]["accuracy"],
                "theme_micro_f1": metrics["themes"]["micro_f1"],
                "mae_simple": metrics["dimension_mae"].get("simple"),
                "extraction_micro_f1": metrics["extraction"]["micro_f1"],
            })
        except Exception as exc:
            errors.append(f"Error reading {f.name}: {exc}")

    rows.sort(key=lambda r: r["timestamp"])
    return rows, errors


def print_trend(results_dir: Path, match_threshold: float = 0.6) -> None:
    """Print a trend table across all result files in a directory."""
    rows, errors = build_trend_rows(results_dir, match_threshold)
    for err in errors:
        print(f"  {err}", file=sys.stderr)
    if not rows:
        print(f"No result files found in {results_dir}", file=sys.stderr)
        return

    print(f"\n{'=' * 60}")
    print(f"Newsletter Trend Report ({len(rows)} runs)")
    print(f"{'=' * 60}")

    widths = [10, 16, 10, 12, 10, 10, 10, 12, 10]
    print("  " + format_table_row(
        ["Run ID", "Timestamp", "Mode", "Tag", "Prompt", "TierAcc", "ThemeF1",
         "MAE(simple)", "ExtractF1"],
        widths,
    ))
    print("  " + "-" * sum(w + 2 for w in widths))

    for r in rows:
        mae_simple = r["mae_simple"]
        mae_s = "N/A" if mae_simple is None else f"{mae_simple:.2f}"
        print("  " + format_table_row([
            r["run_id"][:8],
            r["timestamp"][:16],
            r["mode"],
            r["tag"] or "-",
            r["prompt_hash"][:8],
            format_pct(r["tier_accuracy"]),
            format_pct(r["theme_micro_f1"]),
            mae_s,
            format_pct(r["extraction_micro_f1"]),
        ], widths))

    print()


def _load_results_or_exit(
    path_str: str,
) -> tuple[NewsletterRunMeta, list[StoryPrediction], list[ExtractionPrediction]]:
    """load_results with a friendly one-line error + exit 1 instead of a traceback."""
    path = Path(path_str)
    try:
        return load_results(path)
    except FileNotFoundError:
        print(f"Error: results file not found: {path}", file=sys.stderr)
    # OSError covers IsADirectoryError, PermissionError, etc.; JSONDecodeError
    # is a ValueError subclass but is listed for clarity.
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: could not read results file {path}: {exc}", file=sys.stderr)
        if path.name.endswith(".cot.jsonl"):
            print(
                f"Hint: {path.name} looks like a chain-of-thought sidecar — "
                f"pass the main results file ({path.name[:-len('.cot.jsonl')]}.jsonl).",
                file=sys.stderr,
            )
    sys.exit(1)


def cli():
    parser = argparse.ArgumentParser(description="Generate newsletter evaluation reports")
    parser.add_argument("--results", help="Path to results JSONL file")
    parser.add_argument("--compare", nargs="+", metavar="PATH",
                        help="Compare two result files (.cot.jsonl sidecars matched "
                             "by a glob are ignored)")
    parser.add_argument("--results-dir", help="Directory of results for trend view")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-story flips and per-newsletter extraction diffs")
    parser.add_argument("--format", choices=["table", "json"], default="table",
                        help="Output format")
    parser.add_argument("--match-threshold", type=float, default=0.6,
                        help="SequenceMatcher ratio threshold for extraction matching (default 0.6)")
    args = parser.parse_args()

    if not any([args.results, args.compare, args.results_dir]):
        parser.error("Provide --results, --compare, or --results-dir")

    if args.results:
        meta, story_preds, extraction_preds = _load_results_or_exit(args.results)
        metrics = compute_all_metrics(story_preds, extraction_preds,
                                      args.match_threshold, mode=meta.mode)
        if args.format == "json":
            report_as_json(meta, metrics)
        else:
            print_report(meta, metrics, verbose=args.verbose,
                         story_results=story_preds, extraction_results=extraction_preds,
                         match_threshold=args.match_threshold)

    elif args.compare:
        # A glob like *baseline*.jsonl also matches the run's .cot.jsonl
        # sidecar; drop sidecars so the documented glob workflow works.
        sidecars = [p for p in args.compare if p.endswith(".cot.jsonl")]
        compare_paths = [p for p in args.compare if not p.endswith(".cot.jsonl")]
        if sidecars:
            print(
                "Note: ignoring chain-of-thought sidecar file(s): "
                + ", ".join(sidecars),
                file=sys.stderr,
            )
        if len(compare_paths) != 2:
            parser.error(
                f"--compare needs exactly two results files, got "
                f"{len(compare_paths)} after ignoring .cot.jsonl sidecars: "
                f"{', '.join(compare_paths) or '(none)'}"
            )
        meta1, s1, e1 = _load_results_or_exit(compare_paths[0])
        meta2, s2, e2 = _load_results_or_exit(compare_paths[1])
        m1 = compute_all_metrics(s1, e1, args.match_threshold, mode=meta1.mode)
        m2 = compute_all_metrics(s2, e2, args.match_threshold, mode=meta2.mode)
        if args.format == "json":
            comparison_as_json(meta1, m1, meta2, m2)
        else:
            print_comparison(meta1, m1, meta2, m2, verbose=args.verbose,
                             story1=s1, story2=s2)

    elif args.results_dir:
        if args.verbose:
            print("Note: --verbose has no effect with --results-dir (trend view).",
                  file=sys.stderr)
        if args.format == "json":
            rows, errors = build_trend_rows(Path(args.results_dir), args.match_threshold)
            for err in errors:
                print(f"  {err}", file=sys.stderr)
            print(json.dumps(rows, indent=2))
        else:
            print_trend(Path(args.results_dir), args.match_threshold)


if __name__ == "__main__":
    cli()
