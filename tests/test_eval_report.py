"""Tests for evals.report — metric computation with hand-crafted result lists."""

from evals.report import (
    LABEL_CLASSES,
    SENDER_TYPES,
    compute_accuracy,
    compute_confusion_matrix,
    compute_metrics,
    compute_precision_recall_f1,
    format_thread_label,
    print_comparison,
    print_report,
)
from evals.schemas import GoldenThread, PredictionResult, RunMeta


def _make_result(
    expected_st: str, predicted_st: str | None,
    expected_lb: str, predicted_lb: str | None,
    error: str | None = None,
) -> PredictionResult:
    """Helper to build a PredictionResult with correctness flags set."""
    st_correct = None
    lb_correct = None
    privacy_violation = False

    if predicted_st is not None and error is None:
        st_correct = expected_st == predicted_st
        privacy_violation = expected_st == "person" and predicted_st == "service"
    if predicted_lb is not None and error is None:
        lb_correct = expected_lb == predicted_lb

    return PredictionResult(
        thread_id=f"t_{expected_st}_{expected_lb}",
        expected_sender_type=expected_st,
        expected_label=expected_lb,
        predicted_sender_type=predicted_st,
        predicted_label=predicted_lb,
        sender_type_correct=st_correct,
        label_correct=lb_correct,
        privacy_violation=privacy_violation,
        error=error,
    )


class TestComputeAccuracy:
    def test_all_correct(self):
        results = [
            _make_result("person", "person", "fyi", "fyi"),
            _make_result("service", "service", "low_priority", "low_priority"),
        ]
        assert compute_accuracy(results, "sender_type_correct") == 1.0
        assert compute_accuracy(results, "label_correct") == 1.0

    def test_half_correct(self):
        results = [
            _make_result("person", "person", "fyi", "fyi"),
            _make_result("person", "service", "fyi", "low_priority"),
        ]
        assert compute_accuracy(results, "sender_type_correct") == 0.5
        assert compute_accuracy(results, "label_correct") == 0.5

    def test_none_correct(self):
        results = [
            _make_result("person", "service", "fyi", "low_priority"),
        ]
        assert compute_accuracy(results, "sender_type_correct") == 0.0

    def test_skips_none_values(self):
        """Results with None correctness fields should be excluded."""
        results = [
            _make_result("person", "person", "fyi", "fyi"),
            _make_result("service", None, "low_priority", None, error="timeout"),
        ]
        # Error result has sender_type_correct=None, should be skipped
        assert compute_accuracy(results, "sender_type_correct") == 1.0

    def test_empty_list(self):
        assert compute_accuracy([], "sender_type_correct") is None


class TestConfusionMatrix:
    def test_perfect_sender_classification(self):
        results = [
            _make_result("person", "person", "fyi", "fyi"),
            _make_result("person", "person", "needs_response", "needs_response"),
            _make_result("service", "service", "low_priority", "low_priority"),
        ]
        matrix = compute_confusion_matrix(results, "expected_sender_type", "predicted_sender_type",
                                          SENDER_TYPES)
        assert matrix["person"]["person"] == 2
        assert matrix["person"]["service"] == 0
        assert matrix["service"]["service"] == 1
        assert matrix["service"]["person"] == 0

    def test_sender_misclassifications(self):
        results = [
            _make_result("person", "person", "fyi", "fyi"),      # TP for person
            _make_result("person", "service", "fyi", "fyi"),     # FN for person (privacy violation)
            _make_result("service", "service", "low_priority", "low_priority"),  # TN for person
            _make_result("service", "person", "low_priority", "low_priority"),   # FP for person
        ]
        matrix = compute_confusion_matrix(results, "expected_sender_type", "predicted_sender_type",
                                          SENDER_TYPES)
        assert matrix["person"]["person"] == 1   # TP
        assert matrix["person"]["service"] == 1  # FN (privacy violation!)
        assert matrix["service"]["service"] == 1 # TN
        assert matrix["service"]["person"] == 1  # FP

    def test_label_confusion_matrix(self):
        results = [
            _make_result("service", "service", "needs_response", "needs_response"),
            _make_result("service", "service", "fyi", "fyi"),
            _make_result("service", "service", "low_priority", "fyi"),  # Misclassified
            _make_result("service", "service", "low_priority", "low_priority"),
        ]
        matrix = compute_confusion_matrix(results, "expected_label", "predicted_label", LABEL_CLASSES)
        assert matrix["needs_response"]["needs_response"] == 1
        assert matrix["fyi"]["fyi"] == 1
        assert matrix["low_priority"]["fyi"] == 1  # Misclassification
        assert matrix["low_priority"]["low_priority"] == 1


class TestPrecisionRecallF1:
    def test_perfect_binary(self):
        """Perfect classification should give 1.0 for all metrics."""
        matrix = {
            "person": {"person": 5, "service": 0},
            "service": {"person": 0, "service": 5},
        }
        prf = compute_precision_recall_f1(matrix, SENDER_TYPES)
        assert prf["person"]["precision"] == 1.0
        assert prf["person"]["recall"] == 1.0
        assert prf["person"]["f1"] == 1.0
        assert prf["service"]["precision"] == 1.0
        assert prf["service"]["recall"] == 1.0
        assert prf["service"]["f1"] == 1.0

    def test_known_values(self):
        """Hand-computed precision/recall/F1.

        Person: TP=3, FP=1, FN=2  -> P=3/4=0.75, R=3/5=0.60, F1=2*0.75*0.60/1.35=0.6667
        Service: TP=4, FP=2, FN=1 -> P=4/6=0.6667, R=4/5=0.80, F1=2*0.6667*0.80/1.4667=0.7273
        """
        matrix = {
            "person":  {"person": 3, "service": 2},  # 3 TP, 2 FN
            "service": {"person": 1, "service": 4},   # 4 TP, 1 FN; person FP=1
        }
        prf = compute_precision_recall_f1(matrix, SENDER_TYPES)

        assert abs(prf["person"]["precision"] - 0.75) < 0.001
        assert abs(prf["person"]["recall"] - 0.60) < 0.001
        assert abs(prf["person"]["f1"] - 2 * 0.75 * 0.60 / (0.75 + 0.60)) < 0.001

        assert abs(prf["service"]["precision"] - 4 / 6) < 0.001
        assert abs(prf["service"]["recall"] - 0.80) < 0.001

    def test_zero_support_class(self):
        """Class with no true positives or predictions should get 0.0."""
        matrix = {
            "needs_response": {"needs_response": 0, "fyi": 0, "low_priority": 0},
            "fyi": {"needs_response": 0, "fyi": 5, "low_priority": 0},
            "low_priority": {"needs_response": 0, "fyi": 0, "low_priority": 3},
        }
        prf = compute_precision_recall_f1(matrix, LABEL_CLASSES)
        assert prf["needs_response"]["precision"] == 0.0
        assert prf["needs_response"]["recall"] == 0.0
        assert prf["needs_response"]["f1"] == 0.0
        assert prf["fyi"]["precision"] == 1.0
        assert prf["fyi"]["recall"] == 1.0


class TestComputeMetrics:
    def test_full_pipeline_metrics(self):
        """Test complete metrics computation with a mix of correct/incorrect results."""
        results = [
            # Correct: person + needs_response
            _make_result("person", "person", "needs_response", "needs_response"),
            # Correct: service + low_priority
            _make_result("service", "service", "low_priority", "low_priority"),
            # Wrong sender type (privacy violation), right label
            _make_result("person", "service", "fyi", "fyi"),
            # Right sender type, wrong label
            _make_result("service", "service", "needs_response", "low_priority"),
        ]
        metrics = compute_metrics(results)

        assert metrics["total"] == 4
        assert metrics["errors"] == 0
        assert metrics["valid"] == 4

        # Stage 1: 3/4 correct (one person->service)
        assert metrics["stage1"]["count"] == 4
        assert metrics["stage1"]["accuracy"] == 0.75
        assert metrics["stage1"]["privacy_violations"] == 1
        assert metrics["stage1"]["privacy_violation_rate"] == 0.25

        # Stage 2: 3/4 correct (one needs_response->low_priority)
        assert metrics["stage2"]["count"] == 4
        assert metrics["stage2"]["accuracy"] == 0.75

        # Combined: only 2/4 have both stages correct
        assert metrics["combined"]["count"] == 4
        assert metrics["combined"]["accuracy"] == 0.5

    def test_with_errors(self):
        results = [
            _make_result("person", "person", "fyi", "fyi"),
            _make_result("service", None, "low_priority", None, error="timeout"),
        ]
        metrics = compute_metrics(results)
        assert metrics["total"] == 2
        assert metrics["errors"] == 1
        assert metrics["valid"] == 1

    def test_stage1_only(self):
        """Results with only sender_type predictions (stage1_only mode)."""
        results = [
            PredictionResult(
                thread_id="t1", expected_sender_type="person", expected_label="fyi",
                predicted_sender_type="person", sender_type_correct=True,
            ),
            PredictionResult(
                thread_id="t2", expected_sender_type="service", expected_label="low_priority",
                predicted_sender_type="service", sender_type_correct=True,
            ),
        ]
        metrics = compute_metrics(results)
        assert "stage1" in metrics
        assert metrics["stage1"]["accuracy"] == 1.0
        assert "stage2" not in metrics  # No label predictions
        assert "combined" not in metrics

    def test_empty_results(self):
        metrics = compute_metrics([])
        assert metrics["total"] == 0
        assert metrics["errors"] == 0
        assert "stage1" not in metrics
        assert "stage2" not in metrics
        assert "combined" not in metrics

    def test_privacy_violation_tracking(self):
        """Privacy violations should be tracked independently of accuracy."""
        results = [
            # Person correctly classified as person
            _make_result("person", "person", "fyi", "fyi"),
            # Person misclassified as service (PRIVACY VIOLATION)
            _make_result("person", "service", "needs_response", "needs_response"),
            # Service classified as person (harmless, just inefficient)
            _make_result("service", "person", "low_priority", "low_priority"),
        ]
        metrics = compute_metrics(results)
        assert metrics["stage1"]["privacy_violations"] == 1
        assert abs(metrics["stage1"]["privacy_violation_rate"] - 1 / 3) < 0.001


def _make_meta(**overrides) -> RunMeta:
    defaults = {
        "run_id": "abc12345-6789",
        "timestamp": "2025-01-01T00:00:00",
        "config_hash": "deadbeef",
        "config_path": "config.toml",
        "cloud_model": "test-cloud",
        "local_model": "test-local",
        "golden_set_path": "golden.jsonl",
        "golden_set_count": 10,
    }
    defaults.update(overrides)
    return RunMeta(**defaults)


class TestPrintReportVerbose:
    """Verify --verbose shows disagreements in single-run report."""

    def test_verbose_shows_disagreement_section(self, capsys):
        results = [
            _make_result("person", "service", "fyi", "low_priority"),
            _make_result("service", "service", "low_priority", "low_priority"),
        ]
        meta = _make_meta()
        metrics = compute_metrics(results)
        print_report(meta, metrics, verbose=True, results=results)
        out = capsys.readouterr().out
        assert "--- Disagreements ---" in out
        assert "t_person_fyi" in out
        assert "sender=person->service" in out
        assert "label=fyi->low_priority" in out

    def test_verbose_shows_none_when_all_correct(self, capsys):
        results = [
            _make_result("person", "person", "fyi", "fyi"),
            _make_result("service", "service", "low_priority", "low_priority"),
        ]
        meta = _make_meta()
        metrics = compute_metrics(results)
        print_report(meta, metrics, verbose=True, results=results)
        out = capsys.readouterr().out
        assert "--- Disagreements ---" in out
        assert "None!" in out

    def test_no_verbose_omits_disagreements(self, capsys):
        results = [
            _make_result("person", "service", "fyi", "low_priority"),
        ]
        meta = _make_meta()
        metrics = compute_metrics(results)
        print_report(meta, metrics, verbose=False, results=results)
        out = capsys.readouterr().out
        assert "Disagreements" not in out

    def test_verbose_flags_privacy_violation(self, capsys):
        results = [
            _make_result("person", "service", "fyi", "fyi"),
        ]
        meta = _make_meta()
        metrics = compute_metrics(results)
        print_report(meta, metrics, verbose=True, results=results)
        out = capsys.readouterr().out
        assert "[PRIVACY VIOLATION]" in out


class TestPrintComparisonVerbose:
    """Verify --verbose shows prediction differences in comparison report."""

    def test_regression_a_correct_b_wrong(self, capsys):
        """A predicted correctly, B didn't → appears under Regressions."""
        r1 = [PredictionResult(
            thread_id="t1", expected_sender_type="person", expected_label="fyi",
            predicted_sender_type="person", predicted_label="fyi",
            sender_type_correct=True, label_correct=True,
        )]
        r2 = [PredictionResult(
            thread_id="t1", expected_sender_type="person", expected_label="fyi",
            predicted_sender_type="service", predicted_label="fyi",
            sender_type_correct=False, label_correct=True,
        )]
        meta = _make_meta()
        m1, m2 = compute_metrics(r1), compute_metrics(r2)
        print_comparison(meta, m1, meta, m2, verbose=True, results1=r1, results2=r2)
        out = capsys.readouterr().out
        assert "Regressions (A correct, B wrong)" in out
        assert "sender: person->service" in out
        assert "Improvements" not in out

    def test_improvement_a_wrong_b_correct(self, capsys):
        """A predicted wrong, B got it right → appears under Improvements."""
        r1 = [PredictionResult(
            thread_id="t1", expected_sender_type="service", expected_label="fyi",
            predicted_sender_type="service", predicted_label="low_priority",
            sender_type_correct=True, label_correct=False,
        )]
        r2 = [PredictionResult(
            thread_id="t1", expected_sender_type="service", expected_label="fyi",
            predicted_sender_type="service", predicted_label="fyi",
            sender_type_correct=True, label_correct=True,
        )]
        meta = _make_meta()
        m1, m2 = compute_metrics(r1), compute_metrics(r2)
        print_comparison(meta, m1, meta, m2, verbose=True, results1=r1, results2=r2)
        out = capsys.readouterr().out
        assert "Improvements (A wrong, B correct)" in out
        assert "label: low_priority->fyi" in out
        assert "Regressions" not in out

    def test_other_changes_both_wrong(self, capsys):
        """Both wrong with different predictions → appears under Other changes."""
        r1 = [PredictionResult(
            thread_id="t1", expected_sender_type="service", expected_label="fyi",
            predicted_sender_type="service", predicted_label="needs_response",
            sender_type_correct=True, label_correct=False,
        )]
        r2 = [PredictionResult(
            thread_id="t1", expected_sender_type="service", expected_label="fyi",
            predicted_sender_type="service", predicted_label="low_priority",
            sender_type_correct=True, label_correct=False,
        )]
        meta = _make_meta()
        m1, m2 = compute_metrics(r1), compute_metrics(r2)
        print_comparison(meta, m1, meta, m2, verbose=True, results1=r1, results2=r2)
        out = capsys.readouterr().out
        assert "Other changes (both wrong)" in out
        assert "label: needs_response->low_priority" in out
        assert "Regressions" not in out
        assert "Improvements" not in out

    def test_mixed_thread_regression_and_improvement(self, capsys):
        """Thread with regression on one field and improvement on another → Other."""
        r1 = [PredictionResult(
            thread_id="t1", expected_sender_type="person", expected_label="fyi",
            predicted_sender_type="person", predicted_label="low_priority",
            sender_type_correct=True, label_correct=False,
        )]
        r2 = [PredictionResult(
            thread_id="t1", expected_sender_type="person", expected_label="fyi",
            predicted_sender_type="service", predicted_label="fyi",
            sender_type_correct=False, label_correct=True,
        )]
        meta = _make_meta()
        m1, m2 = compute_metrics(r1), compute_metrics(r2)
        print_comparison(meta, m1, meta, m2, verbose=True, results1=r1, results2=r2)
        out = capsys.readouterr().out
        # Mixed: regression on sender, improvement on label → Other
        assert "Other changes (both wrong)" in out
        assert "sender: person->service" in out
        assert "label: low_priority->fyi" in out
        assert "Regressions (A correct, B wrong)" not in out
        assert "Improvements (A wrong, B correct)" not in out

    def test_verbose_shows_none_when_identical(self, capsys):
        r = [PredictionResult(
            thread_id="t1", expected_sender_type="person", expected_label="fyi",
            predicted_sender_type="person", predicted_label="fyi",
            sender_type_correct=True, label_correct=True,
        )]
        meta = _make_meta()
        m = compute_metrics(r)
        print_comparison(meta, m, meta, m, verbose=True, results1=r, results2=r)
        out = capsys.readouterr().out
        assert "--- Prediction Differences (A -> B) ---" in out
        assert "None!" in out

    def test_no_verbose_omits_diffs(self, capsys):
        r1 = [PredictionResult(
            thread_id="t1", expected_sender_type="person", expected_label="fyi",
            predicted_sender_type="person", predicted_label="fyi",
            sender_type_correct=True, label_correct=True,
        )]
        r2 = [PredictionResult(
            thread_id="t1", expected_sender_type="person", expected_label="fyi",
            predicted_sender_type="service", predicted_label="low_priority",
            sender_type_correct=False, label_correct=False,
        )]
        meta = _make_meta()
        m1, m2 = compute_metrics(r1), compute_metrics(r2)
        print_comparison(meta, m1, meta, m2, verbose=False, results1=r1, results2=r2)
        out = capsys.readouterr().out
        assert "Prediction Differences" not in out

    def test_verbose_skips_error_results(self, capsys):
        r1 = [PredictionResult(
            thread_id="t1", expected_sender_type="person", expected_label="fyi",
            predicted_sender_type="person", predicted_label="fyi",
            sender_type_correct=True, label_correct=True,
        )]
        r2 = [PredictionResult(
            thread_id="t1", expected_sender_type="person", expected_label="fyi",
            error="timeout",
        )]
        meta = _make_meta()
        m1, m2 = compute_metrics(r1), compute_metrics(r2)
        print_comparison(meta, m1, meta, m2, verbose=True, results1=r1, results2=r2)
        out = capsys.readouterr().out
        assert "None!" in out

    def test_verbose_omits_sender_diffs_when_stage1_missing(self, capsys):
        """When one run is stage2_only, sender diffs should be omitted with a warning."""
        r1 = [PredictionResult(
            thread_id="t1", expected_sender_type="person", expected_label="fyi",
            predicted_sender_type="person", predicted_label="fyi",
            sender_type_correct=True, label_correct=True,
        )]
        r2 = [PredictionResult(
            thread_id="t1", expected_sender_type="person", expected_label="fyi",
            predicted_sender_type=None, predicted_label="low_priority",
            label_correct=False,
        )]
        meta_full = _make_meta(stages="full")
        meta_s2 = _make_meta(run_id="def67890", stages="stage2_only")
        m1, m2 = compute_metrics(r1), compute_metrics(r2)
        print_comparison(meta_full, m1, meta_s2, m2, verbose=True, results1=r1, results2=r2)
        out = capsys.readouterr().out
        assert "sender differences omitted" in out
        assert "sender:" not in out
        # Label diffs should still appear under Regressions (A had fyi correct, B got it wrong)
        assert "Regressions" in out
        assert "label: fyi->low_priority" in out

    def test_verbose_omits_label_diffs_when_stage2_missing(self, capsys):
        """When one run is stage1_only, label diffs should be omitted with a warning."""
        r1 = [PredictionResult(
            thread_id="t1", expected_sender_type="person", expected_label="fyi",
            predicted_sender_type="person", predicted_label="fyi",
            sender_type_correct=True, label_correct=True,
        )]
        r2 = [PredictionResult(
            thread_id="t1", expected_sender_type="person", expected_label="fyi",
            predicted_sender_type="service", predicted_label=None,
            sender_type_correct=False,
        )]
        meta_full = _make_meta(stages="full")
        meta_s1 = _make_meta(run_id="def67890", stages="stage1_only")
        m1, m2 = compute_metrics(r1), compute_metrics(r2)
        print_comparison(meta_full, m1, meta_s1, m2, verbose=True, results1=r1, results2=r2)
        out = capsys.readouterr().out
        assert "label differences omitted" in out
        assert "label:" not in out
        # Sender diffs should still appear
        assert "sender: person->service" in out

    def test_verbose_omits_both_when_stages_incompatible(self, capsys):
        """stage1_only vs stage2_only: both warnings shown, no diffs."""
        r1 = [PredictionResult(
            thread_id="t1", expected_sender_type="person", expected_label="fyi",
            predicted_sender_type="person", sender_type_correct=True,
        )]
        r2 = [PredictionResult(
            thread_id="t1", expected_sender_type="person", expected_label="fyi",
            predicted_label="low_priority", label_correct=False,
        )]
        meta_s1 = _make_meta(stages="stage1_only")
        meta_s2 = _make_meta(run_id="def67890", stages="stage2_only")
        m1, m2 = compute_metrics(r1), compute_metrics(r2)
        print_comparison(meta_s1, m1, meta_s2, m2, verbose=True, results1=r1, results2=r2)
        out = capsys.readouterr().out
        assert "sender differences omitted" in out
        assert "label differences omitted" in out
        assert "None!" in out

    def test_verbose_no_warning_when_both_full(self, capsys):
        """Two full runs should show no omission warnings."""
        r = [PredictionResult(
            thread_id="t1", expected_sender_type="person", expected_label="fyi",
            predicted_sender_type="person", predicted_label="fyi",
            sender_type_correct=True, label_correct=True,
        )]
        meta = _make_meta(stages="full")
        m = compute_metrics(r)
        print_comparison(meta, m, meta, m, verbose=True, results1=r, results2=r)
        out = capsys.readouterr().out
        assert "omitted" not in out


def _make_golden(thread_id: str, subject: str, senders: list[str]) -> GoldenThread:
    return GoldenThread(
        thread_id=thread_id,
        messages=[],
        senders=senders,
        subject=subject,
        snippet="",
        expected_sender_type="person",
        expected_label="fyi",
    )


class TestFormatThreadLabel:
    def test_with_context(self):
        ctx = {"t1": _make_golden("t1", "Meeting tomorrow", ["alice@example.com"])}
        assert format_thread_label("t1", ctx) == "t1 (alice@example.com: Meeting tomorrow)"

    def test_without_context(self):
        assert format_thread_label("t1", {}) == "t1"

    def test_long_subject_truncated(self):
        ctx = {"t1": _make_golden("t1", "A" * 100, ["bob@test.com"])}
        result = format_thread_label("t1", ctx, max_subject=20)
        assert result == "t1 (bob@test.com: " + "A" * 20 + "...)"

    def test_short_subject_not_truncated(self):
        ctx = {"t1": _make_golden("t1", "Short", ["x@y.com"])}
        result = format_thread_label("t1", ctx, max_subject=20)
        assert "..." not in result

    def test_no_senders_shows_unknown(self):
        ctx = {"t1": _make_golden("t1", "No sender", [])}
        assert "unknown" in format_thread_label("t1", ctx)


class TestVerboseGoldenContext:
    """Verify that golden set context appears in verbose output."""

    def test_report_disagreements_show_subject(self, capsys):
        results = [
            _make_result("person", "service", "fyi", "low_priority"),
        ]
        ctx = {"t_person_fyi": _make_golden("t_person_fyi", "Lunch plans", ["friend@mail.com"])}
        meta = _make_meta()
        metrics = compute_metrics(results)
        print_report(meta, metrics, verbose=True, results=results, golden_context=ctx)
        out = capsys.readouterr().out
        assert "Lunch plans" in out
        assert "friend@mail.com" in out

    def test_report_without_context_shows_plain_id(self, capsys):
        results = [
            _make_result("person", "service", "fyi", "low_priority"),
        ]
        meta = _make_meta()
        metrics = compute_metrics(results)
        print_report(meta, metrics, verbose=True, results=results, golden_context={})
        out = capsys.readouterr().out
        assert "t_person_fyi:" in out
        # No parenthetical context
        assert "t_person_fyi (" not in out

    def test_comparison_diffs_show_subject(self, capsys):
        r1 = [PredictionResult(
            thread_id="t1", expected_sender_type="person", expected_label="fyi",
            predicted_sender_type="person", predicted_label="fyi",
            sender_type_correct=True, label_correct=True,
        )]
        r2 = [PredictionResult(
            thread_id="t1", expected_sender_type="person", expected_label="fyi",
            predicted_sender_type="service", predicted_label="fyi",
            sender_type_correct=False, label_correct=True,
        )]
        ctx = {"t1": _make_golden("t1", "Project update", ["coworker@corp.com"])}
        meta = _make_meta()
        m1, m2 = compute_metrics(r1), compute_metrics(r2)
        print_comparison(meta, m1, meta, m2, verbose=True, results1=r1, results2=r2,
                         golden_context=ctx)
        out = capsys.readouterr().out
        assert "Project update" in out
        assert "coworker@corp.com" in out


# ---------------------------------------------------------------------------
# Assistant field metrics (issue #78 groundwork)
# ---------------------------------------------------------------------------

def _asst_result(
    expected_lb: str,
    expected_assistant: bool | None,
    predicted_assistant: bool | None = None,
    thread_id: str = "t",
) -> PredictionResult:
    return PredictionResult(
        thread_id=thread_id,
        expected_sender_type="person",
        expected_label=expected_lb,
        predicted_sender_type="person",
        predicted_label=expected_lb,
        sender_type_correct=True,
        label_correct=True,
        expected_assistant=expected_assistant,
        predicted_assistant=predicted_assistant,
    )


class TestAssistantMetrics:
    def test_scored_population_is_annotated_needs_response_threads_only(self):
        results = [
            _asst_result("needs_response", True, True, "a"),
            _asst_result("needs_response", None, None, "b"),   # not annotated
            _asst_result("fyi", None, None, "c"),              # not needs_response
        ]
        a = compute_metrics(results)["assistant"]
        assert a["needs_response_threads"] == 2
        assert a["annotated"] == 1
        assert a["count"] == 1

    def test_binary_precision_recall_f1(self):
        # 2 true positives, 1 false positive, 2 false negatives.
        results = [
            _asst_result("needs_response", True, True, "tp1"),
            _asst_result("needs_response", True, True, "tp2"),
            _asst_result("needs_response", False, True, "fp1"),
            _asst_result("needs_response", True, False, "fn1"),
            _asst_result("needs_response", True, False, "fn2"),
            _asst_result("needs_response", False, False, "tn1"),
        ]
        a = compute_metrics(results)["assistant"]
        assert a["true_positives"] == 2
        assert a["false_positives"] == 1
        assert a["false_negatives"] == 2
        assert a["precision"] == 2 / 3
        assert a["recall"] == 0.5
        assert abs(a["f1"] - 2 * (2 / 3) * 0.5 / ((2 / 3) + 0.5)) < 1e-9
        assert a["count"] == 6

    def test_prediction_outside_the_scored_set_is_a_false_positive_on_its_own_line(self):
        results = [
            _asst_result("needs_response", True, True, "in"),
            _asst_result("fyi", None, True, "out_label"),          # wrong label
            _asst_result("needs_response", None, True, "out_annot"),  # unannotated
            _asst_result("fyi", None, False, "negative"),          # not a positive
        ]
        a = compute_metrics(results)["assistant"]
        assert a["out_of_scope_positives"] == 2
        # ...and it does not contaminate the scored precision.
        assert a["true_positives"] == 1
        assert a["false_positives"] == 0
        assert a["precision"] == 1.0

    def test_no_predictions_reports_annotation_progress_instead_of_metrics(self):
        results = [
            _asst_result("needs_response", True, None, "a"),
            _asst_result("needs_response", None, None, "b"),
            _asst_result("needs_response", False, None, "c"),
            _asst_result("fyi", None, None, "d"),
        ]
        a = compute_metrics(results)["assistant"]
        assert a["predictions"] == 0
        assert a["precision"] is None
        assert a["recall"] is None
        assert a["f1"] is None
        assert a["annotated"] == 2
        assert a["needs_response_threads"] == 3

    def test_error_results_are_excluded(self):
        results = [
            _asst_result("needs_response", True, True, "ok"),
            PredictionResult(
                thread_id="boom", expected_sender_type="person",
                expected_label="needs_response", expected_assistant=True,
                error="timeout",
            ),
        ]
        a = compute_metrics(results)["assistant"]
        assert a["annotated"] == 1
        assert a["needs_response_threads"] == 1

    def test_empty_results_have_no_assistant_section(self):
        assert "assistant" not in compute_metrics([])


class TestAssistantJson:
    def test_json_output_carries_the_assistant_section(self, capsys):
        from evals.report import report_as_json

        metrics = compute_metrics([_asst_result("needs_response", True, None, "a")])
        report_as_json(_make_meta(), metrics)
        payload = __import__("json").loads(capsys.readouterr().out)
        assert payload["metrics"]["assistant"]["annotated"] == 1
        assert payload["metrics"]["assistant"]["predictions"] == 0


class TestPrintAssistantSection:
    def test_no_predictions_prints_annotation_progress(self, capsys):
        results = [
            _asst_result("needs_response", True, None, "a"),
            _asst_result("needs_response", False, None, "b"),
            _asst_result("needs_response", None, None, "c"),
            _asst_result("fyi", None, None, "d"),
        ]
        print_report(_make_meta(), compute_metrics(results))
        out = capsys.readouterr().out
        assert (
            "assistant: no predictions in this run "
            "(2 of 3 needs_response threads annotated)"
        ) in out
        assert "Precision" not in out.split("--- Assistant Field ---")[1]

    def test_metrics_printed_with_n_on_every_figure(self, capsys):
        results = [
            _asst_result("needs_response", True, True, "tp1"),
            _asst_result("needs_response", True, True, "tp2"),
            _asst_result("needs_response", False, True, "fp1"),
            _asst_result("needs_response", True, False, "fn1"),
        ]
        print_report(_make_meta(), compute_metrics(results))
        section = capsys.readouterr().out.split("--- Assistant Field ---")[1]
        assert "Precision: 66.7% (n=4)" in section
        assert "Recall:    66.7% (n=4)" in section
        assert "F1:        66.7% (n=4)" in section

    def test_out_of_scope_positives_get_their_own_line(self, capsys):
        results = [
            _asst_result("needs_response", True, True, "in"),
            _asst_result("fyi", None, True, "out"),
        ]
        print_report(_make_meta(), compute_metrics(results))
        section = capsys.readouterr().out.split("--- Assistant Field ---")[1]
        assert "outside the scored set: 1" in section


def _write_run(path, results, **meta_kw):
    """Write a results JSONL (run_meta line + prediction lines) at *path*."""
    import json

    meta = _make_meta(**meta_kw)
    with open(path, "w") as f:
        f.write(json.dumps(meta.to_dict()) + "\n")
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")


class TestTrendAssistantColumn:
    def test_trend_has_an_assistant_column(self, tmp_path, capsys):
        from evals.report import print_trend

        _write_run(tmp_path / "a.jsonl", [
            _asst_result("needs_response", True, True, "tp"),
            _asst_result("needs_response", False, True, "fp"),
        ], run_id="run-a", tag="with-preds")
        print_trend(tmp_path)
        out = capsys.readouterr().out
        assert "Assistant" in out
        assert "66.7%" in out  # F1 for 1 tp / 1 fp / 0 fn: P=50%, R=100%

    def test_trend_shows_na_when_a_run_has_no_assistant_predictions(self, tmp_path, capsys):
        from evals.report import print_trend

        _write_run(tmp_path / "a.jsonl", [
            _asst_result("needs_response", True, None, "unpredicted"),
        ], run_id="run-a", tag="no-preds")
        print_trend(tmp_path)
        out = capsys.readouterr().out
        assert "Assistant" in out
        assert "N/A" in out


class TestCompareAssistant:
    def _metrics(self, *results):
        return compute_metrics(list(results))

    def test_comparison_shows_assistant_precision_recall_f1_with_delta(self, capsys):
        a = self._metrics(
            _asst_result("needs_response", True, True, "tp"),
            _asst_result("needs_response", False, True, "fp"),
        )
        b = self._metrics(
            _asst_result("needs_response", True, True, "tp"),
            _asst_result("needs_response", False, False, "tn"),
        )
        print_comparison(_make_meta(), a, _make_meta(), b)
        out = capsys.readouterr().out
        assert "--- Assistant Field ---" in out
        section = out.split("--- Assistant Field ---")[1]
        assert "Precision" in section
        assert "Recall" in section
        assert "F1" in section
        assert "+50.0%" in section  # precision 50% -> 100%

    def test_comparison_omits_the_block_when_a_run_has_no_predictions(self, capsys):
        a = self._metrics(_asst_result("needs_response", True, True, "tp"))
        b = self._metrics(_asst_result("needs_response", True, None, "none"))
        print_comparison(_make_meta(), a, _make_meta(), b)
        out = capsys.readouterr().out
        assert "--- Assistant Field ---" not in out
