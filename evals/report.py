"""Compute and display metrics from evaluation results.

Usage:
    python -m evals.report --results evals/results/run.jsonl
    python -m evals.report --compare run1.jsonl run2.jsonl
    python -m evals.report --results-dir evals/results/
    python -m evals.report --results run.jsonl --verbose
    python -m evals.report --results run.jsonl --format json
"""

import argparse
import json
import sys
from pathlib import Path

from evals.schemas import GoldenThread, PredictionResult, RunMeta

SENDER_TYPES = ["person", "service"]
LABEL_CLASSES = ["needs_response", "fyi", "low_priority"]


def load_golden_context(meta: RunMeta) -> dict[str, GoldenThread]:
    """Try to load golden set from meta's recorded path. Returns empty dict on failure."""
    path = Path(meta.golden_set_path)
    if not path.exists():
        return {}
    try:
        context: dict[str, GoldenThread] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                gt = GoldenThread.from_dict(json.loads(line))
                context[gt.thread_id] = gt
        return context
    except Exception:
        return {}


def format_thread_label(thread_id: str, context: dict[str, GoldenThread], max_subject: int = 60) -> str:
    """Format a thread ID with subject/sender context if available."""
    gt = context.get(thread_id)
    if gt is None:
        return thread_id
    subject = gt.subject[:max_subject] + "..." if len(gt.subject) > max_subject else gt.subject
    sender = gt.senders[0] if gt.senders else "unknown"
    return f"{thread_id} ({sender}: {subject})"


def load_results(path: Path) -> tuple[RunMeta, list[PredictionResult]]:
    """Load run metadata and prediction results from JSONL."""
    meta = None
    predictions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("type") == "run_meta":
                meta = RunMeta.from_dict(d)
            elif d.get("type") == "prediction":
                predictions.append(PredictionResult.from_dict(d))
    if meta is None:
        raise ValueError(f"No run_meta found in {path}")
    return meta, predictions


def compute_confusion_matrix(
    results: list[PredictionResult],
    expected_field: str,
    predicted_field: str,
    classes: list[str],
) -> dict[str, dict[str, int]]:
    """Compute a confusion matrix.

    Returns:
        Nested dict: matrix[expected][predicted] = count
    """
    matrix: dict[str, dict[str, int]] = {c: {c2: 0 for c2 in classes} for c in classes}
    for r in results:
        expected = getattr(r, expected_field)
        predicted = getattr(r, predicted_field)
        if expected in classes and predicted in classes:
            matrix[expected][predicted] += 1
    return matrix


def compute_precision_recall_f1(
    matrix: dict[str, dict[str, int]],
    classes: list[str],
) -> dict[str, dict[str, float]]:
    """Compute per-class precision, recall, and F1 from confusion matrix.

    Returns:
        Dict mapping class name to {"precision", "recall", "f1"}.
    """
    metrics: dict[str, dict[str, float]] = {}
    for cls in classes:
        tp = matrix[cls][cls]
        fp = sum(matrix[other][cls] for other in classes if other != cls)
        fn = sum(matrix[cls][other] for other in classes if other != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[cls] = {"precision": precision, "recall": recall, "f1": f1}
    return metrics


def compute_accuracy(results: list[PredictionResult], correct_field: str) -> float | None:
    """Compute accuracy for a correctness field."""
    applicable = [r for r in results if getattr(r, correct_field) is not None]
    if not applicable:
        return None
    correct = sum(1 for r in applicable if getattr(r, correct_field) is True)
    return correct / len(applicable)


def compute_assistant_metrics(valid: list[PredictionResult]) -> dict:
    """Binary metrics for the assistant field (issue #78).

    Scored population: threads whose expected label is ``needs_response`` AND
    which carry a non-null ``expected_assistant`` annotation. ``count`` is that
    n, and every reported figure is over it.

    Within that population a thread counts as a positive prediction only when
    ``predicted_assistant`` is True; a thread with no prediction counts as a
    negative, which is what the absence of the marker label means in practice
    (the owner sees the mail as his own).

    A positive prediction on any thread OUTSIDE the scored population — wrong
    label, or annotation not yet made — is reported separately as
    ``out_of_scope_positives`` rather than folded into precision, so an
    unfinished annotation pass cannot flatter or punish the score.

    ``predictions`` counts non-null ``predicted_assistant`` values WITHIN the
    scored population, because those are the only ones the figures are computed
    over. While it is 0, precision/recall/f1 are None: there is nothing to
    score, and a 0.0 would read as a measured failure.

    Each figure is also None when its own denominator is zero — precision when
    ``tp + fp`` is 0 (nothing was predicted positive), recall when ``tp + fn``
    is 0 (nothing was annotated positive), F1 when ``precision + recall`` is 0.
    A run that correctly predicts an all-negative scored set therefore reads
    N/A rather than 0.0%.
    """
    needs_response = [r for r in valid if r.expected_label == "needs_response"]
    scored = [r for r in needs_response if r.expected_assistant is not None]
    scored_ids = {id(r) for r in scored}
    predictions = [r for r in scored if r.predicted_assistant is not None]
    out_of_scope_positives = sum(
        1 for r in valid if r.predicted_assistant is True and id(r) not in scored_ids
    )

    tp = sum(1 for r in scored if r.expected_assistant is True and r.predicted_assistant is True)
    fp = sum(1 for r in scored if r.expected_assistant is False and r.predicted_assistant is True)
    fn = sum(1 for r in scored if r.expected_assistant is True and r.predicted_assistant is not True)

    metrics: dict = {
        "needs_response_threads": len(needs_response),
        "annotated": len(scored),
        "count": len(scored),
        "predictions": len(predictions),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "out_of_scope_positives": out_of_scope_positives,
        "precision": None,
        "recall": None,
        "f1": None,
    }
    if predictions:
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = None
        metrics.update({"precision": precision, "recall": recall, "f1": f1})
    return metrics


def compute_metrics(results: list[PredictionResult]) -> dict:
    """Compute all metrics from prediction results.

    Returns a dict with stage1, stage2, combined, and summary sections.
    """
    valid = [r for r in results if r.error is None]
    errors = [r for r in results if r.error is not None]

    metrics: dict = {
        "total": len(results),
        "errors": len(errors),
        "valid": len(valid),
    }

    # Stage 1 metrics (sender type)
    st_results = [r for r in valid if r.sender_type_correct is not None]
    if st_results:
        st_accuracy = compute_accuracy(st_results, "sender_type_correct")
        st_matrix = compute_confusion_matrix(st_results, "expected_sender_type", "predicted_sender_type",
                                             SENDER_TYPES)
        st_prf = compute_precision_recall_f1(st_matrix, SENDER_TYPES)
        violations = sum(1 for r in st_results if r.privacy_violation)

        metrics["stage1"] = {
            "accuracy": st_accuracy,
            "confusion_matrix": st_matrix,
            "per_class": st_prf,
            "privacy_violations": violations,
            "privacy_violation_rate": violations / len(st_results) if st_results else 0.0,
            "count": len(st_results),
        }

    # Stage 2 metrics (label)
    lb_results = [r for r in valid if r.label_correct is not None]
    if lb_results:
        lb_accuracy = compute_accuracy(lb_results, "label_correct")
        lb_matrix = compute_confusion_matrix(lb_results, "expected_label", "predicted_label", LABEL_CLASSES)
        lb_prf = compute_precision_recall_f1(lb_matrix, LABEL_CLASSES)

        metrics["stage2"] = {
            "accuracy": lb_accuracy,
            "confusion_matrix": lb_matrix,
            "per_class": lb_prf,
            "count": len(lb_results),
        }

    # Assistant field (issue #78): scored separately from the three-way label.
    if valid:
        metrics["assistant"] = compute_assistant_metrics(valid)

    # Combined (both stages correct)
    both_results = [r for r in valid
                    if r.sender_type_correct is not None and r.label_correct is not None]
    if both_results:
        both_correct = sum(1 for r in both_results
                          if r.sender_type_correct and r.label_correct)
        metrics["combined"] = {
            "accuracy": both_correct / len(both_results),
            "count": len(both_results),
        }

    return metrics


# --- Formatters ---

def format_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1%}"


def format_table_row(cells: list[str], widths: list[int]) -> str:
    return "  ".join(cell.ljust(w) for cell, w in zip(cells, widths))


def format_confusion_matrix(matrix: dict[str, dict[str, int]], classes: list[str]) -> str:
    """Format confusion matrix as ASCII table."""
    col_w = max(len(c) for c in classes) + 2
    header_w = col_w

    lines = []
    lines.append(f"  {'Predicted:':>{header_w}}  " + "  ".join(c.rjust(col_w) for c in classes))
    lines.append("  " + "-" * (header_w + 2 + (col_w + 2) * len(classes)))

    for expected in classes:
        row = f"  {expected.rjust(header_w)}  "
        row += "  ".join(str(matrix[expected][predicted]).rjust(col_w) for predicted in classes)
        lines.append(row)

    return "\n".join(lines)


def format_per_class_table(prf: dict[str, dict[str, float]], classes: list[str]) -> str:
    """Format per-class precision/recall/F1 as ASCII table."""
    widths = [max(len(c) for c in classes), 10, 10, 10]
    lines = []
    lines.append("  " + format_table_row(["Class", "Precision", "Recall", "F1"], widths))
    lines.append("  " + "-" * sum(w + 2 for w in widths))
    for cls in classes:
        m = prf[cls]
        lines.append("  " + format_table_row(
            [cls, format_pct(m["precision"]), format_pct(m["recall"]), format_pct(m["f1"])],
            widths,
        ))
    return "\n".join(lines)


def _out_of_scope_line(a: dict) -> str:
    return (
        f"  Predicted assistant outside the scored set: {a['out_of_scope_positives']} "
        f"(counted separately, not in the figures above)"
    )


def format_assistant_section(a: dict) -> list[str]:
    """Lines for the assistant-field block of a report.

    With no predictions inside the scored set the block states annotation
    progress instead — how much of the needs_response population has been
    annotated — because that is the only thing the run can actually say about
    the field; any positives predicted outside that set are still listed.
    An undefined figure prints ``N/A`` (see ``compute_assistant_metrics``).
    """
    if not a["predictions"]:
        lines = [
            f"  assistant: no predictions in this run ({a['annotated']} of "
            f"{a['needs_response_threads']} needs_response threads annotated)"
        ]
        if a["out_of_scope_positives"]:
            lines.append(_out_of_scope_line(a))
        return lines
    n = a["count"]
    lines = [
        f"  Scored: {n} needs_response threads carrying an expected_assistant annotation",
        f"  Precision: {format_pct(a['precision'])} (n={n})",
        f"  Recall:    {format_pct(a['recall'])} (n={n})",
        f"  F1:        {format_pct(a['f1'])} (n={n})",
        f"  Counts:    tp={a['true_positives']}  fp={a['false_positives']}  "
        f"fn={a['false_negatives']} (n={n})",
    ]
    if a["out_of_scope_positives"]:
        lines.append(_out_of_scope_line(a))
    return lines


def print_report(meta: RunMeta, metrics: dict, verbose: bool = False,
                 results: list[PredictionResult] | None = None,
                 golden_context: dict[str, GoldenThread] | None = None) -> None:
    """Print formatted report to stdout."""
    print(f"\n{'=' * 60}")
    print(f"Evaluation Report: {meta.run_id[:8]}")
    print(f"{'=' * 60}")
    print(f"  Timestamp:   {meta.timestamp}")
    print(f"  Config:      {meta.config_path} ({meta.config_hash})")
    print(f"  Cloud model: {meta.cloud_model}")
    print(f"  Local model: {meta.local_model}")
    print(f"  Stages:      {meta.stages}")
    print(f"  Threads:     {metrics['valid']}/{metrics['total']} ({metrics['errors']} errors)")
    if meta.tag:
        print(f"  Tag:         {meta.tag}")

    if "stage1" in metrics:
        s1 = metrics["stage1"]
        print("\n--- Stage 1: Sender Classification ---")
        print(f"  Accuracy: {format_pct(s1['accuracy'])} ({s1['count']} threads)")
        pv, pvr = s1["privacy_violations"], format_pct(s1["privacy_violation_rate"])
        print(f"  Privacy violations: {pv} ({pvr})")
        print("\n  Confusion matrix:")
        print(format_confusion_matrix(s1["confusion_matrix"], SENDER_TYPES))
        print("\n  Per-class metrics:")
        print(format_per_class_table(s1["per_class"], SENDER_TYPES))

    if "stage2" in metrics:
        s2 = metrics["stage2"]
        print("\n--- Stage 2: Label Classification ---")
        print(f"  Accuracy: {format_pct(s2['accuracy'])} ({s2['count']} threads)")
        print("\n  Confusion matrix:")
        print(format_confusion_matrix(s2["confusion_matrix"], LABEL_CLASSES))
        print("\n  Per-class metrics:")
        print(format_per_class_table(s2["per_class"], LABEL_CLASSES))

    if "assistant" in metrics:
        print("\n--- Assistant Field ---")
        for line in format_assistant_section(metrics["assistant"]):
            print(line)

    if "combined" in metrics:
        c = metrics["combined"]
        print("\n--- Combined (End-to-End) ---")
        print(f"  Accuracy: {format_pct(c['accuracy'])} ({c['count']} threads)")

    if verbose and results:
        print("\n--- Disagreements ---")
        ctx = golden_context or {}
        disagreements = [r for r in results if r.error is None
                         and (r.sender_type_correct is False or r.label_correct is False)]
        if not disagreements:
            print("  None!")
        for r in disagreements:
            parts = [f"  {format_thread_label(r.thread_id, ctx)}:"]
            if r.sender_type_correct is False:
                parts.append(f"sender={r.expected_sender_type}->{r.predicted_sender_type}")
            if r.label_correct is False:
                parts.append(f"label={r.expected_label}->{r.predicted_label}")
            if r.privacy_violation:
                parts.append("[PRIVACY VIOLATION]")
            print(" ".join(parts))

    print()


def print_comparison(
    meta1: RunMeta, metrics1: dict,
    meta2: RunMeta, metrics2: dict,
    verbose: bool = False,
    results1: list[PredictionResult] | None = None,
    results2: list[PredictionResult] | None = None,
    golden_context: dict[str, GoldenThread] | None = None,
) -> None:
    """Print side-by-side comparison of two runs."""
    print(f"\n{'=' * 60}")
    print("Comparison Report")
    print(f"{'=' * 60}")
    print(f"  Run A: {meta1.run_id[:8]} ({meta1.tag or meta1.config_hash})")
    print(f"  Run B: {meta2.run_id[:8]} ({meta2.tag or meta2.config_hash})")

    def delta(a: float | None, b: float | None) -> str:
        if a is None or b is None:
            return "N/A"
        d = b - a
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.1%}"

    widths = [20, 12, 12, 12]

    if "stage1" in metrics1 and "stage1" in metrics2:
        s1a, s1b = metrics1["stage1"], metrics2["stage1"]
        print("\n--- Stage 1: Sender Classification ---")
        print("  " + format_table_row(["Metric", "Run A", "Run B", "Delta"], widths))
        print("  " + "-" * sum(w + 2 for w in widths))
        print("  " + format_table_row([
            "Accuracy",
            format_pct(s1a["accuracy"]), format_pct(s1b["accuracy"]),
            delta(s1a["accuracy"], s1b["accuracy"]),
        ], widths))
        print("  " + format_table_row([
            "Privacy violations",
            str(s1a["privacy_violations"]), str(s1b["privacy_violations"]),
            str(s1b["privacy_violations"] - s1a["privacy_violations"]),
        ], widths))

    if "stage2" in metrics1 and "stage2" in metrics2:
        s2a, s2b = metrics1["stage2"], metrics2["stage2"]
        print("\n--- Stage 2: Label Classification ---")
        print("  " + format_table_row(["Metric", "Run A", "Run B", "Delta"], widths))
        print("  " + "-" * sum(w + 2 for w in widths))
        print("  " + format_table_row([
            "Accuracy",
            format_pct(s2a["accuracy"]), format_pct(s2b["accuracy"]),
            delta(s2a["accuracy"], s2b["accuracy"]),
        ], widths))
        for cls in LABEL_CLASSES:
            if cls in s2a["per_class"] and cls in s2b["per_class"]:
                print("  " + format_table_row([
                    f"  {cls} F1",
                    format_pct(s2a["per_class"][cls]["f1"]),
                    format_pct(s2b["per_class"][cls]["f1"]),
                    delta(s2a["per_class"][cls]["f1"], s2b["per_class"][cls]["f1"]),
                ], widths))

    aa, ab = metrics1.get("assistant"), metrics2.get("assistant")
    if aa and ab and aa["predictions"] and ab["predictions"]:
        print("\n--- Assistant Field ---")
        print("  " + format_table_row(["Metric", "Run A", "Run B", "Delta"], widths))
        print("  " + "-" * sum(w + 2 for w in widths))
        for name, key in [("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")]:
            print("  " + format_table_row([
                f"{name} (n={aa['count']}/{ab['count']})",
                format_pct(aa[key]), format_pct(ab[key]), delta(aa[key], ab[key]),
            ], widths))

    if "combined" in metrics1 and "combined" in metrics2:
        ca, cb = metrics1["combined"], metrics2["combined"]
        print("\n--- Combined ---")
        print("  " + format_table_row(["Metric", "Run A", "Run B", "Delta"], widths))
        print("  " + "-" * sum(w + 2 for w in widths))
        print("  " + format_table_row([
            "E2E Accuracy",
            format_pct(ca["accuracy"]), format_pct(cb["accuracy"]),
            delta(ca["accuracy"], cb["accuracy"]),
        ], widths))

    if verbose and results1 and results2:
        compare_sender = meta1.stages != "stage2_only" and meta2.stages != "stage2_only"
        compare_label = meta1.stages != "stage1_only" and meta2.stages != "stage1_only"
        ctx = golden_context or {}

        regressions: list[tuple[str, list[str]]] = []   # A correct, B wrong
        improvements: list[tuple[str, list[str]]] = []  # A wrong, B correct
        other: list[tuple[str, list[str]]] = []         # both wrong, different

        r2_by_id = {r.thread_id: r for r in results2}
        for r1 in results1:
            r2 = r2_by_id.get(r1.thread_id)
            if r2 is None:
                continue
            if r1.error or r2.error:
                continue
            parts = []
            has_regression = False
            has_improvement = False
            if compare_sender and r1.predicted_sender_type != r2.predicted_sender_type:
                a_right = r1.predicted_sender_type == r1.expected_sender_type
                b_right = r2.predicted_sender_type == r2.expected_sender_type
                parts.append(f"sender: {r1.predicted_sender_type}->{r2.predicted_sender_type}"
                             f" (expected {r1.expected_sender_type})")
                has_regression = has_regression or (a_right and not b_right)
                has_improvement = has_improvement or (b_right and not a_right)
            if compare_label and r1.predicted_label != r2.predicted_label:
                a_right = r1.predicted_label == r1.expected_label
                b_right = r2.predicted_label == r2.expected_label
                parts.append(f"label: {r1.predicted_label}->{r2.predicted_label}"
                             f" (expected {r1.expected_label})")
                has_regression = has_regression or (a_right and not b_right)
                has_improvement = has_improvement or (b_right and not a_right)
            if not parts:
                continue
            entry = (r1.thread_id, parts)
            if has_regression and not has_improvement:
                regressions.append(entry)
            elif has_improvement and not has_regression:
                improvements.append(entry)
            else:
                other.append(entry)

        print("\n--- Prediction Differences (A -> B) ---")
        if not compare_sender:
            print("  (sender differences omitted — not all runs include stage 1)")
        if not compare_label:
            print("  (label differences omitted — not all runs include stage 2)")
        if not regressions and not improvements and not other:
            print("  None!")
        for section_label, entries in [
            ("Regressions (A correct, B wrong)", regressions),
            ("Improvements (A wrong, B correct)", improvements),
            ("Other changes (both wrong)", other),
        ]:
            if not entries:
                continue
            print(f"\n  {section_label}:")
            for tid, parts in entries:
                print(f"    {format_thread_label(tid, ctx)}: {', '.join(parts)}")

    print()


def print_trend(results_dir: Path) -> None:
    """Print trend view across all result files in a directory."""
    files = sorted(results_dir.glob("*.jsonl"))
    if not files:
        print(f"No result files found in {results_dir}", file=sys.stderr)
        return

    print(f"\n{'=' * 60}")
    print(f"Trend Report ({len(files)} runs)")
    print(f"{'=' * 60}")

    widths = [12, 10, 20, 12, 12, 12, 12, 8]
    print("  " + format_table_row(
        ["Run ID", "Stages", "Tag/Config", "Stage 1", "Stage 2", "Combined", "Assistant",
         "Errors"],
        widths,
    ))
    print("  " + "-" * sum(w + 2 for w in widths))

    for f in files:
        try:
            meta, predictions = load_results(f)
            metrics = compute_metrics(predictions)

            s1_acc = format_pct(metrics.get("stage1", {}).get("accuracy"))
            s2_acc = format_pct(metrics.get("stage2", {}).get("accuracy"))
            comb = format_pct(metrics.get("combined", {}).get("accuracy"))
            # Assistant F1 is None until a run actually predicts the field, so
            # an unscored run reads N/A rather than 0.0%.
            asst = format_pct(metrics.get("assistant", {}).get("f1"))

            print("  " + format_table_row([
                meta.run_id[:8],
                meta.stages,
                meta.tag or meta.config_hash[:12],
                s1_acc,
                s2_acc,
                comb,
                asst,
                str(metrics["errors"]),
            ], widths))
        except Exception as exc:
            print(f"  Error reading {f.name}: {exc}", file=sys.stderr)

    print()


def report_as_json(meta: RunMeta, metrics: dict) -> None:
    """Output report as JSON."""
    output = {
        "meta": meta.to_dict(),
        "metrics": {},
    }
    # Convert metrics, replacing non-serializable values
    for key, value in metrics.items():
        if isinstance(value, dict):
            output["metrics"][key] = value
        else:
            output["metrics"][key] = value
    print(json.dumps(output, indent=2))


def cli():
    parser = argparse.ArgumentParser(description="Generate evaluation reports")
    parser.add_argument("--results", help="Path to results JSONL file")
    parser.add_argument("--compare", nargs=2, metavar="PATH", help="Compare two result files")
    parser.add_argument("--results-dir", help="Directory of results for trend view")
    parser.add_argument("--verbose", action="store_true", help="Show per-thread disagreements")
    parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format")
    args = parser.parse_args()

    if not any([args.results, args.compare, args.results_dir]):
        parser.error("Provide --results, --compare, or --results-dir")

    if args.results:
        meta, predictions = load_results(Path(args.results))
        metrics = compute_metrics(predictions)
        if args.format == "json":
            report_as_json(meta, metrics)
        else:
            ctx = load_golden_context(meta) if args.verbose else {}
            print_report(meta, metrics, verbose=args.verbose, results=predictions,
                         golden_context=ctx)

    elif args.compare:
        meta1, pred1 = load_results(Path(args.compare[0]))
        meta2, pred2 = load_results(Path(args.compare[1]))
        metrics1 = compute_metrics(pred1)
        metrics2 = compute_metrics(pred2)
        ctx = load_golden_context(meta1) if args.verbose else {}
        print_comparison(meta1, metrics1, meta2, metrics2,
                         verbose=args.verbose, results1=pred1, results2=pred2,
                         golden_context=ctx)

    elif args.results_dir:
        print_trend(Path(args.results_dir))


if __name__ == "__main__":
    cli()
