"""Tests for evals.newsletter_report — newsletter eval metrics on hand-built fixtures."""

import json
import sys

import pytest

from evals.newsletter_report import (
    THEMES,
    TIERS,
    _format_dim_table,
    build_trend_rows,
    cli,
    comparison_as_json,
    compute_all_metrics,
    compute_dimension_direction,
    compute_dimension_exact_match,
    compute_dimension_mae,
    compute_extraction_metrics,
    compute_multilabel_metrics,
    compute_theme_direction,
    compute_tier_metrics,
    format_mae_delta,
    format_metric_delta,
    format_rate_delta,
    load_story_excerpts,
    match_stories,
    match_stories_detailed,
    print_comparison,
    print_report,
    print_trend,
    report_as_json,
    theme_parse_anomalies,
)
from evals.newsletter_schemas import (
    ExtractionPrediction,
    NewsletterRunMeta,
    StoryPrediction,
)


def _write_results(path, meta, rows=()):
    with open(path, "w") as f:
        f.write(json.dumps(meta.to_dict()) + "\n")
        for row in rows:
            f.write(json.dumps(row.to_dict()) + "\n")


def _meta(mode="all", **kwargs) -> NewsletterRunMeta:
    defaults = dict(
        run_id="abcdef0123456789",
        timestamp="2026-07-03T00:00:00+00:00",
        config_hash="cfg",
        config_path="config.toml",
        newsletter_model="test-model",
        golden_set_path="golden.jsonl",
        golden_set_count=1,
        story_count=1,
        mode=mode,
    )
    defaults.update(kwargs)
    return NewsletterRunMeta(**defaults)


def _pred(
    story_id="s",
    expected_scores=None,
    expected_tier=None,
    expected_themes=None,
    predicted_scores=None,
    predicted_tier=None,
    predicted_themes=None,
    error=None,
) -> StoryPrediction:
    return StoryPrediction(
        story_id=story_id,
        thread_id="t",
        expected_scores=expected_scores,
        expected_tier=expected_tier,
        expected_themes=expected_themes or {},
        predicted_scores=predicted_scores,
        predicted_tier=predicted_tier,
        predicted_themes=predicted_themes or {},
        error=error,
    )


class TestMatchStories:
    def test_exact_match(self):
        predicted = [{"text": "the quick brown fox"}]
        golden = [{"story_id": "t:0", "text": "the quick brown fox"}]
        matched, n_pred, n_gold = match_stories(predicted, golden)
        assert matched == 1
        assert n_pred == 1
        assert n_gold == 1

    def test_fuzzy_match_above_threshold(self):
        # Same story with a couple words changed — ratio stays well above 0.6.
        predicted = [{"text": "the quick brown fox jumped over the lazy dog"}]
        golden = [{"story_id": "t:0",
                   "text": "the quick brown fox jumped over the lazy cat"}]
        matched, _, _ = match_stories(predicted, golden)
        assert matched == 1

    def test_below_threshold_no_match(self):
        predicted = [{"text": "completely unrelated content here"}]
        golden = [{"story_id": "t:0",
                   "text": "an entirely different subject about gardening"}]
        matched, n_pred, n_gold = match_stories(predicted, golden)
        assert matched == 0
        assert n_pred == 1
        assert n_gold == 1

    def test_one_to_one_greedy(self):
        # One predicted story is similar to two golden stories; it may match only one.
        predicted = [{"text": "alpha beta gamma delta"}]
        golden = [
            {"story_id": "t:0", "text": "alpha beta gamma delta"},
            {"story_id": "t:1", "text": "alpha beta gamma delta epsilon"},
        ]
        matched, n_pred, n_gold = match_stories(predicted, golden)
        assert matched == 1  # cannot match both golden stories
        assert n_pred == 1
        assert n_gold == 2

    def test_extra_predicted_drops_precision(self):
        # 2 predicted, 1 golden -> 1 match -> precision 1/2, recall 1/1.
        predicted = [
            {"text": "the quick brown fox"},
            {"text": "spurious hallucinated story"},
        ]
        golden = [{"story_id": "t:0", "text": "the quick brown fox"}]
        matched, n_pred, n_gold = match_stories(predicted, golden)
        assert matched == 1
        assert n_pred == 2
        assert n_gold == 1

    def test_missing_predicted_drops_recall(self):
        # 1 predicted, 2 golden -> 1 match -> precision 1/1, recall 1/2.
        predicted = [{"text": "the quick brown fox"}]
        golden = [
            {"story_id": "t:0", "text": "the quick brown fox"},
            {"story_id": "t:1", "text": "a wholly separate tale of woe"},
        ]
        matched, n_pred, n_gold = match_stories(predicted, golden)
        assert matched == 1
        assert n_pred == 1
        assert n_gold == 2


class TestTierMetrics:
    def test_confusion_and_prf(self):
        results = [
            _pred(predicted_scores={"simple": 3}, expected_tier="excellent",
                  predicted_tier="excellent"),
            _pred(predicted_scores={"simple": 2}, expected_tier="excellent",
                  predicted_tier="good"),  # off-diagonal
            _pred(predicted_scores={"simple": 3}, expected_tier="good",
                  predicted_tier="good"),
            _pred(predicted_scores={"simple": 1}, expected_tier="poor",
                  predicted_tier="poor"),
        ]
        m = compute_tier_metrics(results)
        assert m["errors"] == 0
        assert m["count"] == 4
        assert abs(m["accuracy"] - 0.75) < 1e-9
        cm = m["confusion_matrix"]
        assert cm["excellent"]["excellent"] == 1
        assert cm["excellent"]["good"] == 1
        assert cm["good"]["good"] == 1
        assert cm["poor"]["poor"] == 1
        # excellent recall = 1 TP / (1 TP + 1 FN) = 0.5
        assert abs(m["per_class"]["excellent"]["recall"] - 0.5) < 1e-9
        # good precision = 1 TP / (1 TP + 1 FP) = 0.5
        assert abs(m["per_class"]["good"]["precision"] - 0.5) < 1e-9

    def test_parse_failure_counted_as_error_excluded_from_matrix(self):
        # parse failure: quality was attempted (scores_raw captured) but
        # predicted_scores is None -> error, excluded from matrix.
        # A stray predicted_tier is present but must be ignored because scores
        # failed to parse (guards exclusion-by-None-scores, not by None-tier).
        failed = _pred(predicted_scores=None, expected_tier="good", predicted_tier="good")
        failed.scores_raw = "garbage the parser rejected"
        results = [
            _pred(predicted_scores={"simple": 3}, expected_tier="excellent",
                  predicted_tier="excellent"),
            failed,
        ]
        m = compute_tier_metrics(results)
        assert m["errors"] == 1
        assert m["count"] == 1
        assert m["accuracy"] == 1.0
        # The good-row must NOT appear in the matrix despite predicted_tier="good"
        assert m["confusion_matrix"]["good"]["good"] == 0
        assert sum(
            m["confusion_matrix"][e][p] for e in TIERS for p in TIERS
        ) == 1




class TestDimensionMetrics:
    def test_mae_and_exact(self):
        # 1-3 (Poor/OK/Good) scores; within-1 was dropped (#53) as degenerate.
        results = [
            _pred(
                expected_scores={"simple": 3, "concrete": 2, "personal": 3, "dynamic": 1},
                predicted_scores={"simple": 3, "concrete": 1, "personal": 1, "dynamic": 1},
            ),
            _pred(
                expected_scores={"simple": 1, "concrete": 2, "personal": 2, "dynamic": 2},
                predicted_scores={"simple": 3, "concrete": 2, "personal": 2, "dynamic": 2},
            ),
        ]
        mae = compute_dimension_mae(results)
        # simple: |3-3|=0, |1-3|=2 -> mean 1.0
        assert abs(mae["simple"] - 1.0) < 1e-9
        # concrete: |2-1|=1, |2-2|=0 -> mean 0.5
        assert abs(mae["concrete"] - 0.5) < 1e-9
        # personal: |3-1|=2, |2-2|=0 -> mean 1.0
        assert abs(mae["personal"] - 1.0) < 1e-9
        # dynamic: 0, 0 -> 0.0
        assert abs(mae["dynamic"] - 0.0) < 1e-9

        exact = compute_dimension_exact_match(results)
        # simple: exact on row1 only -> 1/2
        assert abs(exact["simple"] - 0.5) < 1e-9
        # dynamic: both exact -> 1.0
        assert abs(exact["dynamic"] - 1.0) < 1e-9
        # concrete: row1 wrong, row2 exact -> 1/2
        assert abs(exact["concrete"] - 0.5) < 1e-9

    def test_only_over_stories_with_both_scores(self):
        results = [
            _pred(
                expected_scores={"simple": 2, "concrete": 2, "personal": 2, "dynamic": 2},
                predicted_scores={"simple": 2, "concrete": 2, "personal": 2, "dynamic": 2},
            ),
            # parse failure -> excluded entirely
            _pred(expected_scores={"simple": 1}, predicted_scores=None),
            # no expected -> excluded
            _pred(expected_scores=None, predicted_scores={"simple": 1}),
        ]
        mae = compute_dimension_mae(results)
        assert mae["simple"] == 0.0
        exact = compute_dimension_exact_match(results)
        assert exact["simple"] == 1.0




class TestMultilabelThemeMetrics:
    # The default (PRIMARY) metric is positive=emphasized — what earns a Gmail
    # label (issue #53 / A′). "detection" (>=Present) is the secondary metric.
    def test_per_theme_micro_macro_and_exact_set(self):
        # Story 1: expected {scripture, church} emphasized, predicted {scripture}
        #   scripture: TP; church: FN
        # Story 2: expected {christlikeness}, predicted {christlikeness, church}
        #   christlikeness: TP; church: FP
        results = [
            _pred(expected_themes={"scripture": "emphasized", "church": "emphasized"},
                  predicted_themes={"scripture": "emphasized"},
                  predicted_scores={"simple": 3}),
            _pred(expected_themes={"christlikeness": "emphasized"},
                  predicted_themes={"christlikeness": "emphasized", "church": "emphasized"},
                  predicted_scores={"simple": 3}),
        ]
        m = compute_multilabel_metrics(results, THEMES)

        assert m["per_theme"]["scripture"]["f1"] == 1.0
        assert m["per_theme"]["church"]["precision"] == 0.0
        assert m["per_theme"]["church"]["recall"] == 0.0
        assert m["per_theme"]["christlikeness"]["f1"] == 1.0
        # Micro: TP=2, FP=1, FN=1 -> F1=2/3
        assert abs(m["micro_f1"] - (2 / 3)) < 1e-9
        # Macro F1: (1+1+0+0+0)/5 = 0.4
        assert abs(m["macro_f1"] - 0.4) < 1e-9
        assert m["exact_set_match"] == 0.0

    def test_exact_set_match_positive(self):
        results = [
            _pred(expected_themes={"scripture": "emphasized"},
                  predicted_themes={"scripture": "emphasized"},
                  predicted_scores={"simple": 3}),
            _pred(expected_themes={"church": "emphasized", "scripture": "emphasized"},
                  predicted_themes={"scripture": "emphasized", "church": "emphasized"},
                  predicted_scores={"simple": 3}),
        ]
        m = compute_multilabel_metrics(results, THEMES)
        assert m["exact_set_match"] == 1.0

    def test_emphasized_vs_detection_positive(self):
        # Golden PRESENT, predicted EMPHASIZED: an over-grade.
        results = [
            _pred(expected_themes={"scripture": "present"},
                  predicted_themes={"scripture": "emphasized"},
                  predicted_scores={"simple": 3}),
        ]
        # Emphasized (primary): expected is not emphasized -> a false positive.
        emph = compute_multilabel_metrics(results, THEMES, positive="emphasized")
        assert emph["per_theme"]["scripture"]["precision"] == 0.0
        # Detection (secondary): both are >=Present -> a true positive.
        det = compute_multilabel_metrics(results, THEMES, positive="detection")
        assert det["per_theme"]["scripture"]["f1"] == 1.0




class TestExtractionMetrics:
    def test_micro_and_macro(self):
        # Newsletter A: 2 predicted, 2 golden, both match -> P=1, R=1
        # Newsletter B: 2 predicted, 1 golden, 1 match -> P=1/2, R=1
        nl_a = ExtractionPrediction(
            thread_id="A",
            golden_stories=[
                {"story_id": "A:0", "text": "alpha story about faith"},
                {"story_id": "A:1", "text": "beta story about hope"},
            ],
            predicted_stories=[
                {"text": "alpha story about faith"},
                {"text": "beta story about hope"},
            ],
        )
        nl_b = ExtractionPrediction(
            thread_id="B",
            golden_stories=[
                {"story_id": "B:0", "text": "gamma story of grace"},
            ],
            predicted_stories=[
                {"text": "gamma story of grace"},
                {"text": "spurious hallucination unrelated"},
            ],
        )
        m = compute_extraction_metrics([nl_a, nl_b])

        # Micro: matched=3, predicted=4, golden=3 -> P=3/4, R=3/3=1
        assert abs(m["micro_precision"] - 0.75) < 1e-9
        assert abs(m["micro_recall"] - 1.0) < 1e-9
        micro_f1 = 2 * 0.75 * 1.0 / (0.75 + 1.0)
        assert abs(m["micro_f1"] - micro_f1) < 1e-9

        # Macro precision: mean(1.0, 0.5) = 0.75; macro recall mean(1,1)=1
        assert abs(m["macro_precision"] - 0.75) < 1e-9
        assert abs(m["macro_recall"] - 1.0) < 1e-9




class TestMatchStoriesUsesText:
    def test_vague_title_verbatim_text_still_matches(self):
        # The mangle scenario: the model invents a vague title but the extracted
        # text is a verbatim span of the golden story. Matching must be on TEXT,
        # not title wording.
        golden_text = ("Priya's junior year started with a broken ankle and a "
                       "cancelled semester abroad, and she joined our Thursday "
                       "study out of boredom and stayed out of conviction.")
        predicted = [{"text": golden_text}]
        golden = [{"story_id": "g:2",
                   "text": golden_text}]
        matched, _, _ = match_stories(predicted, golden)
        assert matched == 1

    def test_copied_title_garbled_text_does_not_match(self):
        # Converse: a plausible copied title must not mask a garbled text span.
        predicted = [{
                      "text": "completely unrelated words about the annual budget"}]
        golden = [{"story_id": "g:0",
                   "text": "Marcus came to our Tuesday night dinner in January "
                           "as a self-described skeptic and left praying aloud."}]
        matched, _, _ = match_stories(predicted, golden)
        assert matched == 0


class TestMatchStoriesDetailed:
    def test_reports_pairs_and_unmatched(self):
        predicted = [
            {"text": "the quick brown fox jumped over the dog"},
            {"text": "a spurious donation appeal paragraph"},
        ]
        golden = [
            {"story_id": "g:0",
             "text": "the quick brown fox jumped over the cat"},
            {"story_id": "g:1",
             "text": "an entirely different account of the retreat weekend"},
        ]
        detail = match_stories_detailed(predicted, golden)
        assert len(detail["matched"]) == 1
        pair = detail["matched"][0]
        assert pair["pred_index"] == 0
        assert pair["gold_index"] == 0
        assert 0.6 <= pair["ratio"] <= 1.0
        assert detail["unmatched_predicted"] == [1]
        assert detail["unmatched_golden"] == [1]

    def test_respects_threshold(self):
        predicted = [{"text": "the quick brown fox"}]
        golden = [{"story_id": "g:0", "text": "an unrelated gardening story"}]
        strict = match_stories_detailed(predicted, golden, threshold=0.9)
        assert strict["matched"] == []
        loose = match_stories_detailed(predicted, golden, threshold=0.0)
        assert len(loose["matched"]) == 1


class TestExtractionAbstention:
    def test_correct_no_stories_scores_perfect(self):
        # A newsletter with 0 predicted and 0 golden stories is a CORRECT
        # abstention -> per-newsletter precision/recall 1.0, not 0.0.
        nl_empty = ExtractionPrediction(thread_id="C", golden_stories=[],
                                        predicted_stories=[])
        nl_good = ExtractionPrediction(
            thread_id="A",
            golden_stories=[{"story_id": "A:0", "text": "alpha faith"}],
            predicted_stories=[{"text": "alpha faith"}],
        )
        m = compute_extraction_metrics([nl_good, nl_empty])
        assert m["macro_precision"] == 1.0
        assert m["macro_recall"] == 1.0
        empty_row = next(r for r in m["per_newsletter"] if r["thread_id"] == "C")
        assert empty_row["precision"] == 1.0
        assert empty_row["recall"] == 1.0

    def test_empty_results_yield_none_not_zero(self):
        m = compute_extraction_metrics([])
        assert m["count"] == 0
        for key in ("micro_precision", "micro_recall", "micro_f1",
                    "macro_precision", "macro_recall", "macro_f1"):
            assert m[key] is None

    def test_fully_errored_run_surfaces_errors_in_report(self, capsys):
        # Every extraction call failed: the metrics must count the errors and
        # the report must still render an Extraction section naming them,
        # mirroring the tier section — not silently vanish.
        errored = ExtractionPrediction(thread_id="A", golden_stories=[],
                                       predicted_stories=[],
                                       error="connection refused")
        m = compute_extraction_metrics([errored])
        assert m["count"] == 0
        assert m["errors"] == 1
        assert m["error_threads"] == ["A"]
        metrics = compute_all_metrics([], [errored])
        print_report(_meta(mode="extraction"), metrics,
                     extraction_results=[errored])
        out = capsys.readouterr().out
        assert "--- Extraction ---" in out
        assert "1 error" in out
        assert "A" in out.split("--- Extraction ---")[1]


class TestThemeMetricsGating:
    def test_quality_parse_failure_does_not_drop_theme_row(self):
        # Story with malformed quality response (predicted_scores None) but a
        # perfectly parsed theme prediction must still count in theme metrics.
        results = [
            _pred(story_id="ok", expected_themes={"church": "emphasized"},
                  predicted_themes={"church": "emphasized"},
                  predicted_scores={"simple": 3}),
        ]
        failed = _pred(story_id="priya", expected_themes={"christlikeness": "emphasized"},
                       predicted_themes={}, predicted_scores=None)
        failed.themes_raw = "NONE"
        results.append(failed)
        m = compute_multilabel_metrics(results, THEMES)
        assert m["count"] == 2
        # christlikeness expected but not predicted -> FN -> recall 0
        assert m["per_theme"]["christlikeness"]["recall"] == 0.0

    def test_error_rows_still_excluded(self):
        errored = _pred(story_id="down", expected_themes={"church": "emphasized"},
                        error="connection refused")
        m = compute_multilabel_metrics([errored], THEMES)
        assert m["count"] == 0

    def test_empty_scored_yields_none_not_zero(self):
        m = compute_multilabel_metrics([], THEMES)
        assert m["count"] == 0
        for key in ("micro_precision", "micro_recall", "micro_f1", "macro_f1"):
            assert m[key] is None
        assert m["exact_set_match"] is None


class TestTierErrorStories:
    def test_error_story_ids_listed(self):
        failed = _pred(story_id="t:2", predicted_scores=None, expected_tier="good")
        failed.scores_raw = "unparseable quality response"
        results = [
            _pred(story_id="fine", predicted_scores={"simple": 3},
                  expected_tier="good", predicted_tier="good"),
            failed,
        ]
        m = compute_tier_metrics(results)
        assert m["errors"] == 1
        assert m["error_stories"] == ["t:2"]


class TestTierQualityNotAttempted:
    def test_themes_mode_rows_are_skipped_not_errors(self):
        # A --mode themes run never attempts the quality call: error=None AND
        # scores_raw=None. Those rows are "quality not attempted", not parse
        # failures — the tier section must not render as "0 stories, N errors".
        row = _pred(story_id="t:0", expected_tier="good",
                    expected_themes={"church": "emphasized"}, predicted_themes={"church": "emphasized"})
        row.themes_raw = "CHURCH"
        m = compute_tier_metrics([row])
        assert m["errors"] == 0
        assert m["error_stories"] == []
        assert m["count"] == 0

    def test_network_error_rows_still_count_as_errors(self):
        # error rows have scores_raw=None too, but must stay errors.
        m = compute_tier_metrics([
            _pred(story_id="down", expected_tier="good", error="connection refused"),
        ])
        assert m["errors"] == 1
        assert m["error_stories"] == ["down"]

    def test_themes_mode_error_row_does_not_resurrect_tier_section(
        self, tmp_path, monkeypatch, capsys,
    ):
        # In a --mode themes run the quality call is never in play, so a theme
        # call's network error must not render a misleading tier section
        # ("0 stories, 1 error"). The run's mode comes from RunMeta.
        errored = _pred(story_id="down", expected_themes={"church": "emphasized"},
                        error="connection refused")
        path = tmp_path / "themes_run.jsonl"
        _write_results(path, _meta(mode="themes"), [errored])
        monkeypatch.setattr(sys, "argv", [
            "newsletter_report", "--results", str(path),
        ])
        cli()
        out = capsys.readouterr().out
        assert "Tier Classification" not in out

    def test_verbose_failures_section_excludes_not_attempted(self, capsys):
        row = _pred(story_id="t:0", expected_themes={"church": "emphasized"},
                    predicted_themes={"church": "emphasized"})
        row.themes_raw = "CHURCH"
        metrics = compute_all_metrics([row])
        print_report(_meta(mode="themes"), metrics, verbose=True, story_results=[row])
        out = capsys.readouterr().out
        assert "Parse/Network Failures" not in out


class TestThemeParseAnomalies:
    def test_invalid_token_and_empty_parse_detected(self):
        # Graded NAME: GRADE format (issue #53).
        marcus = _pred(story_id="g:0", predicted_themes={"church": "emphasized"},
                       predicted_scores={"simple": 3})
        marcus.themes_raw = "FELLOWSHIP: PRESENT\nCHURCH: EMPHASIZED"
        alina = _pred(story_id="g:1", predicted_themes={},
                      predicted_scores={"simple": 3})
        alina.themes_raw = "The themes of hope and belonging shine through this piece."
        clean_none = _pred(story_id="g:2", predicted_themes={},
                           predicted_scores={"simple": 3})
        clean_none.themes_raw = "NONE"
        clean = _pred(story_id="g:3", predicted_themes={"scripture": "present"},
                      predicted_scores={"simple": 3})
        clean.themes_raw = "SCRIPTURE: PRESENT"
        all_absent = _pred(story_id="g:4", predicted_themes={},
                           predicted_scores={"simple": 3})
        all_absent.themes_raw = (
            "SCRIPTURE: ABSENT\nCHRISTLIKENESS: ABSENT\nCHURCH: ABSENT\n"
            "VOCATION_FAMILY: ABSENT\nDISCIPLE_MAKING: ABSENT"
        )
        anomalies = theme_parse_anomalies([marcus, alina, clean_none, clean, all_absent])
        by_id = {a["story_id"]: a for a in anomalies}
        # g:2 (NONE), g:3 (valid), g:4 (all-ABSENT = valid empty) are NOT anomalies.
        assert set(by_id) == {"g:0", "g:1"}
        assert by_id["g:0"]["kind"] == "invalid_tokens"
        assert by_id["g:0"]["invalid_tokens"] == ["FELLOWSHIP"]
        assert by_id["g:1"]["kind"] == "empty_parse"
        assert by_id["g:1"]["themes_raw"].startswith("The themes")

    def test_rows_without_raw_are_skipped(self):
        # Legacy rows / error rows carry no themes_raw -> no anomaly.
        anomalies = theme_parse_anomalies([_pred(story_id="x", predicted_themes={})])
        assert anomalies == []

    def test_near_miss_line_flagged(self):
        # A near-miss line (misspelled grade) parse_themes silently drops must
        # surface even though the response otherwise parsed — else a systematic
        # parser gap reads as a genuine model miss (Finding 3).
        row = _pred(story_id="n:0", predicted_themes={"scripture": "present"},
                    predicted_scores={"simple": 3})
        row.themes_raw = "SCRIPTURE: PRESENT\nCHURCH: EMPHASISED"
        anomalies = theme_parse_anomalies([row])
        assert len(anomalies) == 1
        assert anomalies[0]["story_id"] == "n:0"
        assert anomalies[0]["kind"] == "near_miss"
        assert "CHURCH: EMPHASISED" in anomalies[0]["near_miss_lines"]

    def test_bulleted_valid_response_is_not_flagged(self):
        # Coherence with Finding 1: once parse_themes tolerates bullets, a bulleted
        # VALID response must not be an anomaly.
        row = _pred(story_id="b:0",
                    predicted_themes={"scripture": "present", "church": "emphasized"},
                    predicted_scores={"simple": 3})
        row.themes_raw = "- SCRIPTURE: PRESENT\n- CHURCH: EMPHASIZED"
        assert theme_parse_anomalies([row]) == []


class TestThemeListSingleSource:
    def test_report_themes_match_newsletter_valid_themes(self):
        # The report's theme list is derived from the pipeline's canonical set
        # so the two can never drift.
        from newsletter import _VALID_THEMES
        assert set(THEMES) == {t.lower() for t in _VALID_THEMES}
        assert len(THEMES) == len(set(THEMES))


class TestComputeAllMetricsBundleKeys:
    def test_story_count_and_anomalies_present(self):
        m = compute_all_metrics([], [])
        assert m["story_count"] == 0
        assert m["theme_anomalies"] == []


class TestPrintReportSections:
    def _extraction_only_metrics(self):
        extraction = [
            ExtractionPrediction(
                thread_id="A",
                golden_stories=[{"story_id": "A:0", "text": "alpha faith"}],
                predicted_stories=[{"text": "alpha faith"}],
            ),
        ]
        return compute_all_metrics([], extraction), extraction

    def test_extraction_only_run_omits_quality_and_theme_sections(self, capsys):
        metrics, _ = self._extraction_only_metrics()
        print_report(_meta(mode="extraction"), metrics)
        out = capsys.readouterr().out
        assert "Themes (multi-label)" not in out
        assert "Quality Dimensions" not in out
        assert "Extraction" in out

    def test_header_shows_golden_set_path(self, capsys):
        metrics, _ = self._extraction_only_metrics()
        print_report(_meta(golden_set_path="/data/golden.jsonl"), metrics)
        out = capsys.readouterr().out
        assert "/data/golden.jsonl" in out

    def test_extraction_section_shows_count_and_threshold(self, capsys):
        metrics, extraction = self._extraction_only_metrics()
        print_report(_meta(), metrics, extraction_results=extraction,
                     match_threshold=0.4)
        out = capsys.readouterr().out
        assert "1 newsletter" in out
        assert "0.4" in out

    def test_single_error_pluralized_and_named(self, capsys):
        failed = _pred(story_id="t:2", predicted_scores=None, expected_tier="good")
        failed.scores_raw = "unparseable quality response"
        story_results = [
            _pred(story_id="ok", predicted_scores={"simple": 3},
                  expected_tier="good", predicted_tier="good"),
            failed,
        ]
        metrics = compute_all_metrics(story_results)
        print_report(_meta(), metrics, story_results=story_results)
        out = capsys.readouterr().out
        assert "1 error)" in out
        assert "1 errors" not in out
        assert "t:2" in out  # the failing story is named

    def test_story_count_pluralizes_as_stories(self, capsys):
        story_results = [
            _pred(story_id=f"s{i}", predicted_scores={"simple": 3},
                  expected_tier="good", predicted_tier="good")
            for i in range(2)
        ]
        metrics = compute_all_metrics(story_results)
        print_report(_meta(), metrics, story_results=story_results)
        out = capsys.readouterr().out
        assert "2 stories" in out
        assert "storys" not in out

    def test_theme_anomaly_count_shown(self, capsys):
        row = _pred(story_id="g:0", predicted_themes={"church": "emphasized"},
                    predicted_scores={"simple": 3})
        row.themes_raw = "FELLOWSHIP\nCHURCH"
        metrics = compute_all_metrics([row])
        print_report(_meta(), metrics, story_results=[row])
        out = capsys.readouterr().out
        assert "anomal" in out.lower()


class TestPrintReportVerbose:
    def _mangled_extraction(self):
        return ExtractionPrediction(
            thread_id="thread-g",
            golden_stories=[
                {"story_id": "g:0",
                 "text": "Marcus came to dinner as a skeptic and left praying aloud."},
                {"story_id": "g:1",
                 "text": "Nine freshmen showed up to the cookout with no idea."},
            ],
            predicted_stories=[
                {"text": "Would you consider a year-end gift to keep this going?"},
            ],
        )

    def test_verbose_lists_unmatched_stories_by_text(self, capsys):
        # Stories are identified by a text excerpt (titles were removed), so
        # unmatched predicted/golden stories are named by the first words of
        # their text.
        extraction = [self._mangled_extraction()]
        metrics = compute_all_metrics([], extraction)
        print_report(_meta(mode="extraction"), metrics, verbose=True,
                     extraction_results=extraction)
        out = capsys.readouterr().out
        assert "Would you consider a year-end gift" in out  # unmatched predicted
        assert "Marcus came to dinner" in out  # unmatched golden
        assert "Nine freshmen showed up" in out

    def test_verbose_lists_matched_pairs_with_ratio_on_mismatch(self, capsys):
        extraction = [
            ExtractionPrediction(
                thread_id="t",
                golden_stories=[
                    {"story_id": "t:0",
                     "text": "the quick brown fox jumped over the lazy dog"},
                    {"story_id": "t:1",
                     "text": "a wholly different story about the retreat"},
                ],
                predicted_stories=[
                    {"text": "the quick brown fox jumped over the lazy cat"},
                ],
            ),
        ]
        metrics = compute_all_metrics([], extraction)
        print_report(_meta(mode="extraction"), metrics, verbose=True,
                     extraction_results=extraction)
        out = capsys.readouterr().out
        assert "the quick brown fox" in out  # the matched pair is shown by text
        assert "0.9" in out  # its ratio

    def test_verbose_diffs_respect_match_threshold(self, capsys):
        # At a permissive threshold everything matches; the diff lines must
        # agree with the headline metrics computed at the same threshold.
        extraction = [self._mangled_extraction()]
        metrics = compute_all_metrics([], extraction, match_threshold=0.0)
        print_report(_meta(mode="extraction"), metrics, verbose=True,
                     extraction_results=extraction, match_threshold=0.0)
        out = capsys.readouterr().out
        assert "matched=1" in out

    def test_verbose_lists_parse_failures_with_raw(self, capsys):
        failed = _pred(story_id="g:2", predicted_scores=None, expected_tier="good")
        failed.scores_raw = "SIMPLE: 4\nCONCRETE: 4\nPERSONAL: 5"
        metrics = compute_all_metrics([failed])
        print_report(_meta(), metrics, verbose=True, story_results=[failed])
        out = capsys.readouterr().out
        assert "g:2" in out
        assert "SIMPLE: 4" in out

    def test_verbose_disagreements_include_quality_parse_failure_with_theme_diff(
        self, capsys,
    ):
        # Quality parse failed but themes parsed fine and disagree -> the story
        # must still appear in Story Disagreements.
        failed = _pred(story_id="g:2", predicted_scores=None,
                       expected_themes={"christlikeness": "emphasized"}, predicted_themes={})
        failed.themes_raw = "NONE"
        metrics = compute_all_metrics([failed])
        print_report(_meta(), metrics, verbose=True, story_results=[failed])
        out = capsys.readouterr().out
        section = out.split("Story Disagreements")[1]
        assert "g:2" in section
        assert "christlikeness" in section

    def test_verbose_disagreements_show_grade_only_theme_mismatch(self, capsys):
        # Same theme KEY, different grade (present vs emphasized): an FP/FN in the
        # primary Emphasized metric. Key-set equality hides it, so the story must
        # still be listed under Story Disagreements with grade-aware detail.
        row = _pred(story_id="g:5", predicted_scores={"simple": 3},
                    expected_tier="good", predicted_tier="good",
                    expected_themes={"scripture": "present"},
                    predicted_themes={"scripture": "emphasized"})
        metrics = compute_all_metrics([row])
        print_report(_meta(), metrics, verbose=True, story_results=[row])
        out = capsys.readouterr().out
        section = out.split("Story Disagreements")[1]
        assert "g:5" in section
        assert "scripture" in section
        assert "present" in section
        assert "emphasized" in section

    def test_verbose_shows_theme_anomaly_raw(self, capsys):
        row = _pred(story_id="g:1", predicted_themes={},
                    predicted_scores={"simple": 3})
        row.themes_raw = "The themes of hope and belonging shine through."
        metrics = compute_all_metrics([row])
        print_report(_meta(), metrics, verbose=True, story_results=[row])
        out = capsys.readouterr().out
        assert "hope and belonging" in out

    def test_verbose_disagreements_show_text_excerpt_from_golden_set(self, tmp_path, capsys):
        golden = tmp_path / "golden.jsonl"
        golden.write_text(json.dumps({
            "type": "golden_newsletter", "thread_id": "t", "message_id": "m",
            "sender": "s", "subject": "subj", "body": "b",
            "stories": [{"story_id": "t:1",
                         "text": "Alina welcomed every freshman at the table."}],
        }) + "\n")
        row = _pred(story_id="t:1", predicted_scores={"simple": 3},
                    expected_tier="good", predicted_tier="fair")
        metrics = compute_all_metrics([row])
        print_report(_meta(golden_set_path=str(golden)), metrics, verbose=True,
                     story_results=[row])
        out = capsys.readouterr().out
        assert "Alina welcomed every freshman" in out


class TestLoadStoryExcerpts:
    def test_reads_text_excerpts_by_story_id(self, tmp_path):
        golden = tmp_path / "golden.jsonl"
        golden.write_text(json.dumps({
            "type": "golden_newsletter", "thread_id": "t", "message_id": "m",
            "sender": "s", "subject": "subj", "body": "b",
            "stories": [{"story_id": "t:0", "text": "Marcus came to dinner"}],
        }) + "\n")
        assert load_story_excerpts(str(golden)) == {"t:0": "Marcus came to dinner"}

    def test_missing_file_returns_empty(self):
        assert load_story_excerpts("/nope/does-not-exist.jsonl") == {}


class TestDimTableAlignment:
    def test_data_rows_align_with_header(self):
        mae = {d: 0.5 for d in ["simple", "concrete", "personal", "dynamic"]}
        exact = {d: 0.5 for d in mae}
        lines = _format_dim_table(mae, exact).splitlines()
        header, first_row = lines[0], lines[2]
        # The MAE column must start at the same offset in header and data rows.
        assert first_row.index("0.50") == header.index("MAE")
        assert "Within-1" not in header


class TestPrintComparisonModes:
    def _story_metrics(self):
        results = [
            _pred(story_id="s", predicted_scores={"simple": 3},
                  expected_tier="good", predicted_tier="good",
                  expected_themes={"church": "emphasized"}, predicted_themes={"church": "emphasized"}),
        ]
        return compute_all_metrics(results)

    def _extraction_metrics(self):
        extraction = [
            ExtractionPrediction(
                thread_id="A",
                golden_stories=[{"story_id": "A:0", "text": "alpha faith"}],
                predicted_stories=[{"text": "alpha faith"}],
            ),
        ]
        return compute_all_metrics([], extraction)

    def test_header_shows_each_runs_mode(self, capsys):
        m = self._story_metrics()
        print_comparison(_meta(mode="quality"), m, _meta(mode="quality"), m)
        out = capsys.readouterr().out
        header = out.split("--- Tier ---")[0]
        assert "mode=quality" in header

    def test_mode_mismatch_warns(self, capsys):
        print_comparison(
            _meta(mode="quality"), self._story_metrics(),
            _meta(mode="extraction"), self._extraction_metrics(),
        )
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "quality" in out and "extraction" in out

    def test_same_mode_does_not_warn(self, capsys):
        m = self._story_metrics()
        print_comparison(_meta(mode="quality"), m, _meta(mode="quality"), m)
        assert "WARNING" not in capsys.readouterr().out

    def test_empty_sections_render_na_not_minus_100pct(self, capsys):
        # Quality-mode run A vs extraction-mode run B: B has no tier/theme data,
        # so per-class F1 and theme F1 must be N/A, never a -100.0% "regression".
        print_comparison(
            _meta(mode="quality"), self._story_metrics(),
            _meta(mode="extraction"), self._extraction_metrics(),
        )
        out = capsys.readouterr().out
        assert "-100.0%" not in out
        # e.g. the "good F1" row: Run A 100%, Run B N/A, delta N/A
        good_row = next(ln for ln in out.splitlines() if "good F1" in ln)
        assert "N/A" in good_row

    def test_themes_section_shows_detection_f1_rows(self, capsys):
        # print_report shows a detection (>=Present) line for each single run;
        # compare mode must too, or a detection-level regression between runs is
        # invisible. Build a run where detection F1 (100%) differs from the
        # Emphasized F1 (0%) so the detection rows are distinguishable.
        row = _pred(story_id="s", predicted_scores={"simple": 3},
                    expected_themes={"scripture": "present"},
                    predicted_themes={"scripture": "present"})
        m = compute_all_metrics([row])
        print_comparison(_meta(mode="all"), m, _meta(mode="all"), m)
        out = capsys.readouterr().out
        themes_section = out.split("--- Themes ---")[1].split("--- Extraction ---")[0]
        assert "Detection" in themes_section
        # Detection micro-F1 = 100% (scripture Present in both), while the
        # Emphasized micro-F1 above it is 0%.
        assert "100.0%" in themes_section


class TestVerboseCompareThemesMode:
    def test_theme_flip_shown_when_quality_never_ran(self, capsys):
        # --mode themes rows never have predicted_scores; a theme flip between
        # runs must still be listed (not "None!") under Per-story Flips.
        def row(themes):  # themes: {name: grade}
            r = _pred(story_id="s", expected_themes={"church": "emphasized"},
                      predicted_themes=themes)
            r.themes_raw = "\n".join(f"{t.upper()}: EMPHASIZED" for t in themes) or "NONE"
            return r

        story1 = [row({"church": "emphasized"})]
        story2 = [row({"scripture": "emphasized"})]
        m1 = compute_all_metrics(story1)
        m2 = compute_all_metrics(story2)
        print_comparison(_meta(mode="themes"), m1, _meta(mode="themes"), m2,
                         verbose=True, story1=story1, story2=story2)
        out = capsys.readouterr().out
        flips = out.split("Per-story Flips")[1]
        assert "None!" not in flips
        assert "church" in flips and "scripture" in flips

    def test_grade_only_theme_flip_shown(self, capsys):
        # Same theme KEY, grade changed present->emphasized between runs. Key-set
        # equality hides the flip even though it moves the Emphasized F1 shown in
        # the table; it must be listed with grade-aware detail (not "None!").
        def row(grade):
            r = _pred(story_id="s", expected_themes={"scripture": "present"},
                      predicted_themes={"scripture": grade})
            r.themes_raw = f"SCRIPTURE: {grade.upper()}"
            return r

        story1 = [row("present")]
        story2 = [row("emphasized")]
        m1 = compute_all_metrics(story1)
        m2 = compute_all_metrics(story2)
        print_comparison(_meta(mode="themes"), m1, _meta(mode="themes"), m2,
                         verbose=True, story1=story1, story2=story2)
        out = capsys.readouterr().out
        flips = out.split("Per-story Flips")[1]
        assert "None!" not in flips
        assert "scripture" in flips
        assert "present" in flips and "emphasized" in flips


class TestTrendRows:
    def _write_two_runs(self, tmp_path):
        story = _pred(story_id="s", predicted_scores={"simple": 3},
                      expected_tier="good", predicted_tier="good",
                      predicted_themes={"church": "emphasized"}, expected_themes={"church": "emphasized"})
        # File names sort in the OPPOSITE order of their timestamps, so
        # chronological ordering must come from run_meta.timestamp.
        _write_results(
            tmp_path / "a_newest.jsonl",
            _meta(run_id="b" * 16, timestamp="2026-07-02T00:00:00+00:00",
                  prompt_hash="hash-newer", tag="newer"),
            [story],
        )
        _write_results(
            tmp_path / "z_oldest.jsonl",
            _meta(run_id="a" * 16, timestamp="2026-07-01T00:00:00+00:00",
                  prompt_hash="hash-older", tag="older"),
            [story],
        )

    def test_rows_sorted_chronologically_with_timestamp_and_prompt_hash(self, tmp_path):
        self._write_two_runs(tmp_path)
        rows, errors = build_trend_rows(tmp_path)
        assert errors == []
        assert [r["timestamp"] for r in rows] == [
            "2026-07-01T00:00:00+00:00", "2026-07-02T00:00:00+00:00",
        ]
        assert rows[0]["prompt_hash"] == "hash-older"
        assert rows[0]["tier_accuracy"] == 1.0

    def test_print_trend_has_timestamp_column_in_order(self, tmp_path, capsys):
        self._write_two_runs(tmp_path)
        print_trend(tmp_path)
        out = capsys.readouterr().out
        assert "Timestamp" in out
        assert out.index("older") < out.index("newer")


class TestJsonOutput:
    def _story_and_extraction(self):
        story = _pred(story_id="s", predicted_scores={"simple": 3},
                      expected_tier="good", predicted_tier="good",
                      expected_themes={"church": "emphasized"},
                      predicted_themes={"church": "emphasized"})
        extraction = ExtractionPrediction(
            thread_id="A",
            golden_stories=[{"story_id": "A:0", "text": "alpha faith"}],
            predicted_stories=[{"text": "alpha faith"}],
        )
        return [story], [extraction]

    def test_report_as_json_emits_valid_json_with_expected_keys(self, capsys):
        stories, extraction = self._story_and_extraction()
        metrics = compute_all_metrics(stories, extraction)
        report_as_json(_meta(), metrics)
        data = json.loads(capsys.readouterr().out)
        assert set(data) == {"meta", "metrics"}
        assert data["meta"]["run_id"] == "abcdef0123456789"
        for key in ("story_count", "tier", "dimension_mae", "dimension_exact",
                    "themes", "themes_detection", "theme_anomalies", "extraction"):
            assert key in data["metrics"]
        assert "dimension_within1" not in data["metrics"]  # dropped (#53)
        assert data["metrics"]["tier"]["accuracy"] == 1.0
        assert data["metrics"]["extraction"]["micro_f1"] == 1.0

    def test_comparison_as_json_emits_valid_json_with_expected_keys(self, capsys):
        stories, extraction = self._story_and_extraction()
        metrics = compute_all_metrics(stories, extraction)
        comparison_as_json(_meta(run_id="a" * 16), metrics,
                           _meta(run_id="b" * 16), metrics)
        data = json.loads(capsys.readouterr().out)
        assert set(data) == {"run_a", "run_b"}
        assert data["run_a"]["meta"]["run_id"] == "a" * 16
        assert data["run_b"]["meta"]["run_id"] == "b" * 16
        assert data["run_b"]["metrics"]["themes"]["micro_f1"] == 1.0

    def test_cli_single_run_format_json_emits_json(self, tmp_path, monkeypatch, capsys):
        stories, _ = self._story_and_extraction()
        path = tmp_path / "run.jsonl"
        _write_results(path, _meta(), stories)
        monkeypatch.setattr(sys, "argv", [
            "newsletter_report", "--results", str(path), "--format", "json",
        ])
        cli()
        data = json.loads(capsys.readouterr().out)
        assert data["meta"]["run_id"] == "abcdef0123456789"
        assert data["metrics"]["tier"]["accuracy"] == 1.0

    def test_cli_trend_format_json_rows_have_expected_keys(
        self, tmp_path, monkeypatch, capsys,
    ):
        stories, _ = self._story_and_extraction()
        _write_results(tmp_path / "run.jsonl", _meta(), stories)
        monkeypatch.setattr(sys, "argv", [
            "newsletter_report", "--results-dir", str(tmp_path), "--format", "json",
        ])
        cli()
        data = json.loads(capsys.readouterr().out)
        assert isinstance(data, list) and len(data) == 1
        for key in ("file", "run_id", "timestamp", "mode", "tag", "prompt_hash",
                    "tier_accuracy", "theme_micro_f1", "mae_simple",
                    "extraction_micro_f1"):
            assert key in data[0]


class TestCliJsonAndErrors:
    def _write_run(self, path, **meta_kwargs):
        story = _pred(story_id="s", predicted_scores={"simple": 3},
                      expected_tier="good", predicted_tier="good",
                      predicted_themes={"church": "emphasized"}, expected_themes={"church": "emphasized"})
        _write_results(path, _meta(**meta_kwargs), [story])

    def test_compare_format_json_emits_json(self, tmp_path, monkeypatch, capsys):
        a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
        self._write_run(a, run_id="a" * 16)
        self._write_run(b, run_id="b" * 16)
        monkeypatch.setattr(sys, "argv", [
            "newsletter_report", "--compare", str(a), str(b), "--format", "json",
        ])
        cli()
        data = json.loads(capsys.readouterr().out)
        assert data["run_a"]["meta"]["run_id"] == "a" * 16
        assert data["run_b"]["metrics"]["tier"]["accuracy"] == 1.0

    def test_results_dir_format_json_emits_json(self, tmp_path, monkeypatch, capsys):
        self._write_run(tmp_path / "r.jsonl")
        monkeypatch.setattr(sys, "argv", [
            "newsletter_report", "--results-dir", str(tmp_path), "--format", "json",
        ])
        cli()
        data = json.loads(capsys.readouterr().out)
        assert isinstance(data, list)
        assert data[0]["run_id"] == "abcdef0123456789"

    def test_results_dir_verbose_warns_it_is_unsupported(self, tmp_path, monkeypatch, capsys):
        self._write_run(tmp_path / "r.jsonl")
        monkeypatch.setattr(sys, "argv", [
            "newsletter_report", "--results-dir", str(tmp_path), "--verbose",
        ])
        cli()
        assert "--verbose" in capsys.readouterr().err

    def test_missing_results_file_exits_cleanly(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", [
            "newsletter_report", "--results", "/nope/missing.jsonl",
        ])
        with pytest.raises(SystemExit) as excinfo:
            cli()
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "missing.jsonl" in err
        assert "Traceback" not in err

    def test_results_on_directory_exits_cleanly(self, tmp_path, monkeypatch, capsys):
        # --results pointed at a directory raises IsADirectoryError from open();
        # the friendly-errors path must catch it (OSError), not traceback.
        monkeypatch.setattr(sys, "argv", [
            "newsletter_report", "--results", str(tmp_path),
        ])
        with pytest.raises(SystemExit) as excinfo:
            cli()
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert str(tmp_path) in err
        assert "Traceback" not in err

    def test_compare_ignores_cot_sidecars_from_globs(self, tmp_path, monkeypatch, capsys):
        # The documented workflow globs *<tag>*.jsonl, which also matches the
        # .cot.jsonl sidecar every thinking run writes. --compare must ignore
        # sidecars and compare the two real results files.
        a, b = tmp_path / "20260701_all_baseline_aaaa.jsonl", tmp_path / "20260702_all_variant_bbbb.jsonl"
        self._write_run(a, run_id="a" * 16)
        self._write_run(b, run_id="b" * 16)
        a_cot = tmp_path / "20260701_all_baseline_aaaa.cot.jsonl"
        b_cot = tmp_path / "20260702_all_variant_bbbb.cot.jsonl"
        a_cot.write_text(json.dumps({"type": "thinking", "story_id": "s"}) + "\n")
        b_cot.write_text(json.dumps({"type": "thinking", "story_id": "s"}) + "\n")
        # Shell-glob expansion order: cot file sorts before its results file.
        monkeypatch.setattr(sys, "argv", [
            "newsletter_report", "--compare",
            str(a_cot), str(a), str(b_cot), str(b),
        ])
        cli()
        captured = capsys.readouterr()
        assert "Newsletter Comparison Report" in captured.out
        assert "aaaaaaaa" in captured.out
        assert "bbbbbbbb" in captured.out
        assert "sidecar" in captured.err  # tells the user what was skipped

    def test_compare_errors_when_not_two_results_files_remain(
        self, tmp_path, monkeypatch, capsys,
    ):
        a = tmp_path / "run_a.jsonl"
        self._write_run(a, run_id="a" * 16)
        cot = tmp_path / "run_a.cot.jsonl"
        cot.write_text(json.dumps({"type": "thinking", "story_id": "s"}) + "\n")
        monkeypatch.setattr(sys, "argv", [
            "newsletter_report", "--compare", str(a), str(cot),
        ])
        with pytest.raises(SystemExit) as excinfo:
            cli()
        assert excinfo.value.code != 0
        assert "Traceback" not in capsys.readouterr().err

    def test_results_on_cot_sidecar_hints_at_main_file(self, tmp_path, monkeypatch, capsys):
        cot = tmp_path / "run.cot.jsonl"
        cot.write_text(json.dumps({"type": "thinking", "story_id": "s"}) + "\n")
        monkeypatch.setattr(sys, "argv", ["newsletter_report", "--results", str(cot)])
        with pytest.raises(SystemExit) as excinfo:
            cli()
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "chain-of-thought sidecar" in err
        assert "Traceback" not in err

    def test_meta_less_file_exits_cleanly(self, tmp_path, monkeypatch, capsys):
        bad = tmp_path / "no_meta.jsonl"
        bad.write_text(json.dumps({"type": "story_prediction", "story_id": "s",
                                   "thread_id": "t"}) + "\n")
        monkeypatch.setattr(sys, "argv", ["newsletter_report", "--results", str(bad)])
        with pytest.raises(SystemExit) as excinfo:
            cli()
        assert excinfo.value.code == 1
        assert "no_meta.jsonl" in capsys.readouterr().err


class TestComparisonDeltaSign:
    def test_f1_improvement_positive(self):
        # Run B has higher tier accuracy than Run A -> positive delta.
        assert format_metric_delta(0.5, 0.75).startswith("+")
        assert format_metric_delta(0.75, 0.5).startswith("-")

    def test_mae_decrease_is_improvement(self):
        # MAE dropped from 1.0 to 0.5 -> improvement -> flagged as improvement (+).
        improved = format_mae_delta(1.0, 0.5)
        assert "improve" in improved.lower() or improved.strip().startswith("-")
        # The raw delta must reflect the decrease (b - a = -0.5).
        assert "-0.5" in improved
        # MAE increase is a regression.
        worsened = format_mae_delta(0.5, 1.0)
        assert "+0.5" in worsened
        assert "improve" not in worsened.lower()

    def test_compute_all_metrics_bundles_sections(self):
        results = [
            _pred(
                expected_scores={"simple": 3, "concrete": 3, "personal": 3, "dynamic": 3},
                predicted_scores={"simple": 3, "concrete": 3, "personal": 3, "dynamic": 3},
                expected_tier="excellent", predicted_tier="excellent",
                expected_themes={"scripture": "emphasized"},
                predicted_themes={"scripture": "emphasized"},
            ),
        ]
        extraction = [
            ExtractionPrediction(
                thread_id="A",
                golden_stories=[{"story_id": "A:0", "text": "alpha faith"}],
                predicted_stories=[{"text": "alpha faith"}],
            ),
        ]
        m = compute_all_metrics(results, extraction)
        assert m["tier"]["accuracy"] == 1.0
        assert m["themes"]["micro_f1"] == 1.0
        assert m["dimension_mae"]["simple"] == 0.0
        assert m["extraction"]["micro_f1"] == 1.0


class TestStoryLabelCountInHeader:
    def test_header_splits_the_story_count_by_source(self, capsys):
        metrics = compute_all_metrics([_pred()], [])
        print_report(_meta(mode="quality", story_count=7, story_label_count=5), metrics,
                     story_results=[_pred()])
        assert "Stories:      7 (2 curated, 5 from the story-label log)" in capsys.readouterr().out

    def test_header_is_unchanged_without_label_log_stories(self, capsys):
        metrics = compute_all_metrics([_pred()], [])
        print_report(_meta(mode="quality", story_count=7), metrics, story_results=[_pred()])
        out = capsys.readouterr().out
        assert "Stories:      7\n" in out
        assert "story-label log" not in out

    def test_meta_written_before_the_field_existed_loads_with_zero(self):
        old = _meta().to_dict()
        del old["story_label_count"]
        del old["story_labels_path"]
        restored = NewsletterRunMeta.from_dict(old)
        assert (restored.story_label_count, restored.story_labels_path) == (0, "")


def _scores(simple, concrete, personal, dynamic):
    return {"simple": simple, "concrete": concrete, "personal": personal, "dynamic": dynamic}


def _themed(story_id, expected, predicted):
    """A row whose theme reply is the well-formed one that parses to *predicted*."""
    row = _pred(story_id=story_id, expected_themes=expected, predicted_themes=predicted)
    row.themes_raw = "\n".join(
        f"{theme.upper()}: {predicted.get(theme, 'absent').upper()}" for theme in THEMES
    )
    return row


class TestDimensionDirection:
    """Over-grading is the rubric's costliest error (docs/newsletter-rubric.md §1);
    agreement metrics alone cannot tell it from under-grading."""

    def _rows(self):
        return [
            _pred("a", expected_scores=_scores(2, 2, 2, 2), predicted_scores=_scores(3, 2, 1, 2)),
            _pred("b", expected_scores=_scores(1, 2, 3, 2), predicted_scores=_scores(3, 2, 3, 2)),
            _pred("c", expected_scores=_scores(2, 2, 2, 2), predicted_scores=None),  # parse failure
        ]

    def test_too_high_is_measured_over_the_labels_that_could_be_graded_higher(self):
        simple = compute_dimension_direction(self._rows())["simple"]
        assert (simple["n"], simple["over_count"], simple["over_of"]) == (2, 2, 2)
        assert simple["over"] == 1.0
        assert simple["signed"] == 1.5

    def test_too_low_is_measured_over_the_labels_that_could_be_graded_lower(self):
        direction = compute_dimension_direction(self._rows())
        # simple's labels are OK and Poor: only the OK can be graded lower.
        assert (direction["simple"]["under_count"], direction["simple"]["under_of"]) == (0, 1)
        personal = direction["personal"]
        assert (personal["under_count"], personal["under_of"], personal["under"]) == (1, 2, 0.5)
        # personal's labels are OK and Good: a Good cannot be over-graded.
        assert (personal["over_count"], personal["over_of"]) == (0, 1)

    def test_a_good_label_is_never_counted_as_a_chance_to_over_grade(self):
        rows = [_pred("a", expected_scores=_scores(3, 3, 3, 3), predicted_scores=_scores(3, 3, 3, 3))]
        simple = compute_dimension_direction(rows)["simple"]
        assert (simple["over_of"], simple["over"]) == (0, None)
        assert (simple["under_of"], simple["under"]) == (1, 0.0)

    def test_a_wrong_good_is_counted_apart_from_other_over_grades(self):
        # The rubric's worst error is GOOD for a story that has not earned it;
        # OK for a Poor story is its cheapest. One pooled rate hides the difference.
        rows = [
            _pred("a", expected_scores=_scores(1, 2, 2, 2), predicted_scores=_scores(2, 3, 2, 2)),
        ]
        direction = compute_dimension_direction(rows)
        assert direction["simple"]["over_count"] == 1
        assert direction["simple"]["wrong_good"] == 0
        assert direction["concrete"]["wrong_good"] == 1
        assert direction["all"]["wrong_good"] == 1
        assert direction["all"]["wrong_good_rate"] == pytest.approx(1 / 4)

    def test_all_pools_every_dimension_judgment(self):
        direction = compute_dimension_direction(self._rows())["all"]
        assert direction["n"] == 8
        assert (direction["over_count"], direction["over_of"]) == (2, 7)
        assert (direction["under_count"], direction["under_of"]) == (1, 7)
        assert direction["signed"] == pytest.approx((1 + 2 - 1) / 8)

    def test_no_scored_stories_is_not_a_rate_of_zero(self):
        direction = compute_dimension_direction([])
        assert direction["simple"]["n"] == 0
        assert direction["simple"]["over"] is None and direction["simple"]["under"] is None
        assert direction["simple"]["signed"] is None
        assert direction["all"]["wrong_good_rate"] is None


class TestThemeDirection:
    def _rows(self):
        return [
            _themed("a", {"scripture": "present"}, {"scripture": "emphasized"}),
            _themed("b", {}, {"scripture": "emphasized", "church": "present"}),
            _themed("c", {"scripture": "emphasized", "church": "emphasized"},
                    {"scripture": "present"}),
            _pred("d", expected_themes={"scripture": "present"}),  # theme call never ran
        ]

    def test_rates_are_over_the_labels_that_could_move_that_way(self):
        direction = compute_theme_direction(self._rows(), THEMES)
        scripture = direction["scripture"]
        assert scripture["n"] == 3
        assert (scripture["over_count"], scripture["over_of"], scripture["over"]) == (2, 2, 1.0)
        assert (scripture["under_count"], scripture["under_of"], scripture["under"]) == (1, 2, 0.5)
        church = direction["church"]
        assert (church["over_count"], church["over_of"]) == (1, 2)
        assert (church["under_count"], church["under_of"]) == (1, 1)

    def test_missing_every_labeled_theme_reads_as_always_too_low(self):
        # Ten stories, two labeled scripture=emphasized, the model finds no
        # theme anywhere. Measured over all cells that would read as 4%.
        rows = [_themed(f"s{i}", {"scripture": "emphasized"} if i < 2 else {}, {})
                for i in range(10)]
        direction = compute_theme_direction(rows, THEMES)
        assert direction["scripture"]["under"] == 1.0
        assert (direction["all"]["under_count"], direction["all"]["under_of"]) == (2, 2)
        assert direction["all"]["under"] == 1.0

    def test_wrong_emphasized_grades_come_with_how_many_stories_could_have_one(self):
        # EMPHASIZED is what earns a Gmail label (D14), so a wrong one is split
        # by the true grade, each against the number of stories so labeled.
        scripture = compute_theme_direction(self._rows(), THEMES)["scripture"]
        assert (scripture["emphasized_for_present"], scripture["present_labeled"]) == (1, 1)
        assert (scripture["emphasized_for_absent"], scripture["absent_labeled"]) == (1, 1)
        church = compute_theme_direction(self._rows(), THEMES)["church"]
        assert (church["emphasized_for_absent"], church["absent_labeled"]) == (0, 2)

    def test_all_pools_every_theme_judgment(self):
        direction = compute_theme_direction(self._rows(), THEMES)["all"]
        assert direction["n"] == 3 * len(THEMES)
        assert (direction["over_count"], direction["over_of"]) == (3, 13)
        assert (direction["under_count"], direction["under_of"]) == (2, 3)

    def test_no_scored_stories_is_not_a_rate_of_zero(self):
        direction = compute_theme_direction([], THEMES)
        assert direction["scripture"]["n"] == 0
        assert direction["scripture"]["over"] is None
        assert direction["all"]["under"] is None


class TestDirectionInReports:
    def _rows(self):
        rows = [
            _pred("a", expected_scores=_scores(2, 2, 2, 2), predicted_scores=_scores(3, 2, 1, 2),
                  expected_themes={"scripture": "present"},
                  predicted_themes={"scripture": "emphasized"}),
            _pred("b", expected_scores=_scores(1, 2, 3, 2), predicted_scores=_scores(3, 2, 3, 2)),
        ]
        for row in rows:
            row.themes_raw = "SCRIPTURE: ABSENT"
        return rows

    def test_metrics_bundle_carries_both_directions(self):
        metrics = compute_all_metrics(self._rows(), [])
        assert metrics["dimension_direction"]["simple"]["over"] == 1.0
        assert metrics["theme_direction"]["scripture"]["emphasized_for_present"] == 1

    def test_single_run_report_shows_counts_with_their_denominators(self, capsys):
        rows = self._rows()
        print_report(_meta(mode="all", story_count=2), compute_all_metrics(rows, []),
                     story_results=rows)
        out = capsys.readouterr().out
        direction = out[out.index("--- Grading direction"):]
        assert "Too high" in direction and "Too low" in direction and "Wrong GOOD" in direction
        simple = next(line for line in direction.splitlines()
                      if line.strip().startswith("simple"))
        assert "2/2 (100.0%)" in simple   # too high
        assert "0/1 (0.0%)" in simple     # too low: only one label could be graded lower
        scripture = next(line for line in direction.splitlines()
                         if line.strip().startswith("scripture"))
        assert "1/1" in scripture         # Emphasized given to the one Present-labeled story

    def test_single_run_report_prints_na_when_nothing_was_scored(self, capsys):
        rows = [_pred("a", expected_scores=_scores(2, 2, 2, 2), predicted_scores=None)]
        print_report(_meta(mode="quality"), compute_all_metrics(rows, []), story_results=rows)
        out = capsys.readouterr().out
        direction = out[out.index("--- Grading direction"):]
        simple = next(line for line in direction.splitlines()
                      if line.strip().startswith("simple"))
        assert "N/A" in simple and "%" not in simple

    def _after(self):
        after = [
            _pred("a", expected_scores=_scores(2, 2, 2, 2), predicted_scores=_scores(2, 2, 1, 2),
                  expected_themes={"scripture": "present"},
                  predicted_themes={"scripture": "present"}),
            _pred("b", expected_scores=_scores(1, 2, 3, 2), predicted_scores=_scores(2, 2, 3, 2)),
        ]
        for row in after:
            row.themes_raw = "SCRIPTURE: PRESENT"
        return after

    def test_comparison_shows_whether_over_grading_fell(self, capsys):
        before, after = self._rows(), self._after()
        print_comparison(
            _meta(tag="before", newsletter_model="model-a"), compute_all_metrics(before, []),
            _meta(tag="after", newsletter_model="model-b"), compute_all_metrics(after, []),
            story1=before, story2=after,
        )
        out = capsys.readouterr().out
        too_high = out[out.index("Graded too high (lower=better)"):]
        simple = next(line for line in too_high.splitlines() if line.strip().startswith("simple"))
        assert "100.0%" in simple and "50.0%" in simple and "(improved)" in simple
        assert "wrong GOOD" in too_high
        assert "Emph. for Present" in too_high
        assert "Graded too low (lower=better)" in out
        assert "model=model-a" in out and "model=model-b" in out
        assert "WARNING: the two runs scored different stories" not in out

    def test_comparison_warns_when_the_runs_scored_different_stories(self, capsys):
        # newsletter_run reads the story-label log by default, so a run taken
        # after more labeling scores more stories than the one before it.
        before = self._rows()
        after = self._after() + [
            _pred("new", expected_scores=_scores(2, 2, 2, 2), predicted_scores=_scores(2, 2, 2, 2)),
        ]
        print_comparison(
            _meta(tag="before", story_count=2), compute_all_metrics(before, []),
            _meta(tag="after", story_count=3, story_label_count=1), compute_all_metrics(after, []),
            story1=before, story2=after,
        )
        out = capsys.readouterr().out
        assert "WARNING: the two runs scored different stories" in out
        assert "2 in common" in out
        assert "stories=2" in out and "stories=3 (1 from the story-label log)" in out


class TestThemeDirectionSkipsUnreadableReplies:
    def test_a_reply_the_parser_could_not_read_is_not_counted_as_too_low(self):
        # "DISCIPLE-MAKING: EMPHASIZED" is a near-miss line parse_themes skips.
        # That is a parser gap, reported under Parse anomalies, not the model
        # grading the theme Absent.
        unreadable = _pred("a", expected_themes={"disciple_making": "emphasized"})
        unreadable.themes_raw = "DISCIPLE-MAKING: EMPHASIZED"
        read = _themed("b", {"disciple_making": "emphasized"}, {"disciple_making": "emphasized"})
        read.themes_raw = "DISCIPLE_MAKING: EMPHASIZED"
        direction = compute_theme_direction([unreadable, read], THEMES)
        assert direction["disciple_making"]["n"] == 1
        assert direction["disciple_making"]["under_count"] == 0
        assert direction["all"]["unreadable"] == 1

    def test_the_report_says_how_many_replies_were_left_out(self, capsys):
        unreadable = _pred("a", expected_themes={"disciple_making": "emphasized"})
        unreadable.themes_raw = "DISCIPLE-MAKING: EMPHASIZED"
        print_report(_meta(mode="themes"), compute_all_metrics([unreadable], [], mode="themes"),
                     story_results=[unreadable])
        out = capsys.readouterr().out
        assert "1 story left out of the theme rows" in out


class TestFormatRateDelta:
    def test_a_lower_rate_is_an_improvement(self):
        assert format_rate_delta(0.5, 0.25) == "-25.0% (improved)"

    def test_a_higher_rate_is_worse(self):
        assert format_rate_delta(0.25, 0.5) == "+25.0% (worse)"

    def test_no_change_and_missing_values(self):
        assert format_rate_delta(0.25, 0.25) == "+0.0%"
        assert format_rate_delta(None, 0.25) == "N/A"


class TestExcerptsForLabelLogStories:
    def _label_line(self, key, text):
        return json.dumps({
            "type": "story_label", "story_key": key, "thread_id": "t_900020", "text": text,
            "status": "graded",
            "scores": {"simple": 3, "concrete": 3, "personal": 3, "dynamic": 3},
        }) + "\n"

    def test_label_log_stories_are_excerpted_by_their_key(self, tmp_path):
        labels = tmp_path / "labels.jsonl"
        labels.write_text(self._label_line("0123456789abcdef", "Priya led the Tuesday study."))
        excerpts = load_story_excerpts("/nope/absent.jsonl", str(labels))
        assert excerpts == {"0123456789abcdef": "Priya led the Tuesday study."}

    def test_verbose_report_names_a_label_log_story_by_its_text(self, tmp_path, capsys):
        labels = tmp_path / "labels.jsonl"
        labels.write_text(self._label_line("0123456789abcdef", "Priya led the Tuesday study."))
        row = _pred(story_id="0123456789abcdef", predicted_scores={"simple": 3},
                    expected_tier="good", predicted_tier="fair")
        print_report(_meta(golden_set_path="/nope/absent.jsonl", story_labels_path=str(labels),
                           story_label_count=1),
                     compute_all_metrics([row]), verbose=True, story_results=[row])
        assert "Priya led the Tuesday study" in capsys.readouterr().out

    def test_an_unreadable_label_log_degrades_to_curated_excerpts_only(self, tmp_path):
        golden = tmp_path / "golden.jsonl"
        golden.write_text(json.dumps({
            "type": "golden_newsletter", "thread_id": "t", "message_id": "m",
            "sender": "s", "subject": "subj", "body": "b",
            "stories": [{"story_id": "t:0", "text": "Marcus came to dinner"}],
        }) + "\n")
        labels = tmp_path / "labels.jsonl"
        labels.write_text("{not json}\n")
        assert load_story_excerpts(str(golden), str(labels)) == {"t:0": "Marcus came to dinner"}
