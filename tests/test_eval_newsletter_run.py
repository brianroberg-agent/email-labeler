"""Tests for the newsletter eval runner."""

import argparse
import asyncio
import copy
import json
import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from evals import newsletter_run
from evals.llm_cache import CachedLLMClient
from evals.newsletter_grade import append_label, story_key
from evals.newsletter_run import (
    build_meta,
    build_output_path,
    compute_prompt_hash,
    evaluate_extraction,
    evaluate_story,
    load_golden_set,
    load_story_labels,
    merge_prompts_override,
    run_evaluation,
    write_results,
    write_thinking_sidecar,
)
from evals.newsletter_schemas import GoldenNewsletter, GoldenStory, StoryLabel
from newsletter import NewsletterClassifier


class FakeLLM:
    """Fake inner LLMClient: returns canned (response, thinking) per prompt.

    Keyed by a substring match on the user_content so a single instance can
    answer extraction / quality / theme calls differently. Records every call.
    """

    def __init__(self, responses):
        # responses: list of (needle, response, thinking); first needle-in-user wins
        self.responses = responses
        self.model = "fake-model"
        self.temperature = 0.0
        self.max_tokens = 100
        self.extra_body = {}
        self.calls = []

    async def complete(self, system_prompt, user_content, include_thinking=False):
        self.calls.append((system_prompt, user_content))
        for needle, response, thinking in self.responses:
            if needle in user_content:
                return (response, thinking) if include_thinking else response
        return ("", "") if include_thinking else ""

    async def is_available(self, timeout=None):
        return True

    async def probe(self, timeout=None):
        from llm_client import AvailabilityResult
        return AvailabilityResult(ok=await self.is_available(timeout))


def _classifier(llm):
    config = _prompts_config()
    return NewsletterClassifier(cloud_llm=llm, config=config)


def _run(coro):
    return asyncio.run(coro)


def _newsletter(thread_id, stories=None, **kw):
    base = dict(
        thread_id=thread_id, message_id=f"m-{thread_id}", sender="s@dm.org",
        subject="subj", body="body text", stories=stories or [],
    )
    base.update(kw)
    return GoldenNewsletter(**base)


def _story(story_id, **kw):
    # Tests model Phase-B-labeled stories by default; pass reviewed=False for
    # a Phase-A-only (confirmed but never labeled) story.
    base = dict(story_id=story_id, text="x", reviewed=True)
    base.update(kw)
    return GoldenStory(**base)


def _write_golden(path, newsletters):
    path.write_text("".join(json.dumps(n.to_dict()) + "\n" for n in newsletters))


def _prompts_config():
    return {
        "newsletter": {
            "prompts": {
                "story_extraction": {"system": "extract sys", "user_template": "EXTRACT {body}"},
                "quality_assessment": {"system": "quality sys", "user_template": "QUALITY {text}"},
                "theme_classification": {"system": "theme sys", "user_template": "THEME {text}"},
            }
        }
    }


class TestComputePromptHash:
    def test_stable_for_identical_config(self):
        a = compute_prompt_hash(_prompts_config())
        b = compute_prompt_hash(_prompts_config())
        assert a == b
        assert len(a) == 16

    def test_changes_when_a_prompt_string_changes(self):
        base = _prompts_config()
        mutated = copy.deepcopy(base)
        mutated["newsletter"]["prompts"]["quality_assessment"]["system"] = "different"
        assert compute_prompt_hash(base) != compute_prompt_hash(mutated)


class TestMergePromptsOverride:
    def test_deep_merges_and_changes_prompt_hash(self):
        base = _prompts_config()
        before = compute_prompt_hash(base)
        override = {
            "newsletter": {
                "prompts": {"quality_assessment": {"system": "NEW quality sys"}}
            }
        }
        merge_prompts_override(base, override)
        # overridden key changed
        assert base["newsletter"]["prompts"]["quality_assessment"]["system"] == "NEW quality sys"
        # sibling keys in the same block preserved (deep merge, not replace)
        qa = base["newsletter"]["prompts"]["quality_assessment"]
        assert qa["user_template"] == "QUALITY {text}"
        # other prompt blocks untouched
        assert base["newsletter"]["prompts"]["story_extraction"]["system"] == "extract sys"
        # and the hash moved
        assert compute_prompt_hash(base) != before

    def test_only_touches_newsletter_prompts(self):
        base = _prompts_config()
        base["newsletter"]["llm"] = {"model": "keep-me"}
        override = {"newsletter": {"llm": {"model": "IGNORED"}, "prompts": {}}}
        merge_prompts_override(base, override)
        assert base["newsletter"]["llm"]["model"] == "keep-me"


class TestLoadGoldenSet:
    def test_reviewed_only_filters_unreviewed(self, tmp_path):
        path = tmp_path / "g.jsonl"
        _write_golden(path, [
            _newsletter("keep", reviewed=True),
            _newsletter("drop", reviewed=False),
        ])
        loaded, _stats = load_golden_set(path, reviewed_only=True)
        assert [n.thread_id for n in loaded] == ["keep"]

    def test_excluded_dropped_even_when_unreviewed_included(self, tmp_path):
        path = tmp_path / "g.jsonl"
        _write_golden(path, [
            _newsletter("keep", reviewed=False),
            _newsletter("drop", reviewed=False, excluded=True),
        ])
        loaded, _stats = load_golden_set(path, reviewed_only=False)
        assert [n.thread_id for n in loaded] == ["keep"]

    def test_tolerates_blank_lines(self, tmp_path):
        path = tmp_path / "g.jsonl"
        content = json.dumps(_newsletter("keep", reviewed=True).to_dict()) + "\n\n"
        path.write_text(content)
        loaded, _stats = load_golden_set(path, reviewed_only=True)
        assert [n.thread_id for n in loaded] == ["keep"]

    def test_returns_filter_breakdown_stats(self, tmp_path):
        path = tmp_path / "g.jsonl"
        _write_golden(path, [
            _newsletter("a", reviewed=True),
            _newsletter("b", reviewed=False),
            _newsletter("c", reviewed=False, excluded=True),
        ])
        loaded, stats = load_golden_set(path, reviewed_only=True)
        assert [n.thread_id for n in loaded] == ["a"]
        assert stats == {"total": 3, "excluded": 1, "unreviewed": 1}


class TestFormatLoadSummary:
    """The load message must explain WHY newsletters were dropped, not dead-end."""

    def test_all_unreviewed_names_the_fix(self):
        from evals.newsletter_run import format_load_summary
        msg = format_load_summary(
            0, {"total": 6, "excluded": 0, "unreviewed": 6}, "gs.jsonl")
        assert "No newsletters to evaluate" in msg
        assert "6 unreviewed" in msg
        assert "newsletter_label" in msg
        assert "--include-unreviewed" in msg

    def test_empty_file_points_at_harvest(self):
        from evals.newsletter_run import format_load_summary
        msg = format_load_summary(
            0, {"total": 0, "excluded": 0, "unreviewed": 0}, "gs.jsonl")
        assert "No newsletters to evaluate" in msg
        assert "empty" in msg
        assert "newsletter_harvest" in msg

    def test_success_reports_skip_breakdown(self):
        from evals.newsletter_run import format_load_summary
        msg = format_load_summary(
            5, {"total": 7, "excluded": 1, "unreviewed": 1}, "gs.jsonl")
        assert "5 of 7" in msg
        assert "1 unreviewed" in msg
        assert "1 excluded" in msg

    def test_success_with_no_skips_is_terse(self):
        from evals.newsletter_run import format_load_summary
        msg = format_load_summary(
            3, {"total": 3, "excluded": 0, "unreviewed": 0}, "gs.jsonl")
        assert "Loaded 3 newsletters" in msg
        assert "unreviewed" not in msg


class TestEvaluateExtraction:
    def test_records_predicted_and_golden_story_sets(self):
        extraction_out = "STORY: pred body one\n\nSTORY: pred body two"
        llm = FakeLLM([("EXTRACT", extraction_out, "")])
        newsletter = _newsletter(
            "nl1", reviewed=True,
            stories=[
                _story("nl1:0", text="gold body one"),
                _story("nl1:1", text="gold body two"),
            ],
        )
        pred, _thinking = _run(evaluate_extraction(newsletter, _classifier(llm)))

        assert pred.thread_id == "nl1"
        assert pred.predicted_stories == [
            {"text": "pred body one"},
            {"text": "pred body two"},
        ]
        assert pred.golden_stories == [
            {"story_id": "nl1:0", "text": "gold body one"},
            {"story_id": "nl1:1", "text": "gold body two"},
        ]
        assert pred.error is None


class TestEvaluateStory:
    def _story_llm(self):
        quality_out = "SIMPLE: GOOD\nCONCRETE: GOOD\nPERSONAL: GOOD\nDYNAMIC: OK"
        theme_out = "SCRIPTURE: EMPHASIZED\nCHURCH: PRESENT"
        return FakeLLM([
            ("QUALITY", quality_out, "quality reasoning"),
            ("THEME", theme_out, "theme reasoning"),
        ])

    def test_records_scores_themes_and_derives_tier(self):
        llm = self._story_llm()
        story = _story(
            "nl1:0", text="the text",
            expected_scores={"simple": 3, "concrete": 3, "personal": 3, "dynamic": 3},
            expected_tier="excellent", expected_themes={"scripture": "emphasized"},
        )
        pred, thinking = _run(evaluate_story(story, "nl1", _classifier(llm)))

        assert pred.story_id == "nl1:0"
        assert pred.thread_id == "nl1"
        # expected_* carried from the golden story
        assert pred.expected_scores == {"simple": 3, "concrete": 3, "personal": 3, "dynamic": 3}
        assert pred.expected_tier == "excellent"
        assert pred.expected_themes == {"scripture": "emphasized"}
        # predicted_*
        assert pred.predicted_scores == {"simple": 3, "concrete": 3, "personal": 3, "dynamic": 2}
        assert pred.predicted_themes == {"scripture": "emphasized", "church": "present"}
        # tier derived via compute_tier: avg = (3+3+3+2)/4 = 2.75 -> excellent
        assert pred.predicted_tier == "excellent"
        # raw captured
        assert "SIMPLE: GOOD" in pred.scores_raw
        assert "SCRIPTURE" in pred.themes_raw
        assert pred.error is None
        # thinking sidecar carries both cots
        assert thinking.story_id == "nl1:0"
        assert thinking.quality_cot == "quality reasoning"
        assert thinking.theme_cot == "theme reasoning"

    def test_predicted_tier_none_when_scores_unparseable(self):
        llm = FakeLLM([
            ("QUALITY", "garbage no scores", "q"),
            ("THEME", "NONE", "t"),
        ])
        story = _story("nl2:0", text="x")
        pred, _thinking = _run(evaluate_story(story, "nl2", _classifier(llm)))
        assert pred.predicted_scores is None
        assert pred.predicted_tier is None
        assert pred.predicted_themes == {}


def _full_llm():
    return FakeLLM([
        ("EXTRACT", "STORY: body", "ecot"),
        ("QUALITY", "SIMPLE: GOOD\nCONCRETE: GOOD\nPERSONAL: OK\nDYNAMIC: OK", "qcot"),
        ("THEME", "SCRIPTURE: PRESENT", "tcot"),
    ])


class TestRunEvaluation:
    def _golden(self):
        return [_newsletter(
            "nl1", reviewed=True,
            stories=[_story("nl1:0", text="body")],
        )]

    def test_all_mode_produces_extraction_and_story_rows(self, tmp_path):
        cache = tmp_path / "cache.jsonl"
        cached = CachedLLMClient(_full_llm(), cache)
        rows, thinking = _run(run_evaluation(
            self._golden(), _classifier(cached), mode="all", parallelism=1,
        ))
        from evals.newsletter_schemas import ExtractionPrediction, StoryPrediction
        extractions = [r for r in rows if isinstance(r, ExtractionPrediction)]
        stories = [r for r in rows if isinstance(r, StoryPrediction)]
        assert len(extractions) == 1
        # The extractor output ("STORY: body") must actually parse into a story,
        # otherwise the test passes even when extraction is broken.
        assert extractions[0].predicted_stories == [{"text": "body"}]
        assert len(stories) == 1
        assert stories[0].predicted_tier == "good"  # avg (3+3+2+2)/4 = 2.5
        story_entries = [t for t in thinking if t.story_id]
        assert story_entries[0].quality_cot == "qcot"
        # all-mode also carries a newsletter-level extraction thinking entry
        assert [t.thread_id for t in thinking if t.extraction_cot or not t.story_id]

    def test_second_identical_run_records_cache_hits(self, tmp_path):
        cache = tmp_path / "cache.jsonl"
        cached = CachedLLMClient(_full_llm(), cache)
        _run(run_evaluation(self._golden(), _classifier(cached), mode="all", parallelism=1))
        first_misses = cached.misses
        assert first_misses > 0
        cached.flush()

        # Fresh cache client over the same on-disk cache: every call is now a hit.
        cached2 = CachedLLMClient(_full_llm(), cache)
        _run(run_evaluation(self._golden(), _classifier(cached2), mode="all", parallelism=1))
        assert cached2.hits == first_misses
        assert cached2.misses == 0

    def test_extraction_mode_skips_story_scoring(self, tmp_path):
        cache = tmp_path / "cache.jsonl"
        cached = CachedLLMClient(_full_llm(), cache)
        rows, _thinking = _run(run_evaluation(
            self._golden(), _classifier(cached), mode="extraction", parallelism=1,
        ))
        from evals.newsletter_schemas import ExtractionPrediction, StoryPrediction
        assert all(isinstance(r, ExtractionPrediction) for r in rows)
        assert not any(isinstance(r, StoryPrediction) for r in rows)

    def test_quality_mode_skips_extraction(self, tmp_path):
        cache = tmp_path / "cache.jsonl"
        cached = CachedLLMClient(_full_llm(), cache)
        rows, _thinking = _run(run_evaluation(
            self._golden(), _classifier(cached), mode="quality", parallelism=1,
        ))
        from evals.newsletter_schemas import ExtractionPrediction, StoryPrediction
        assert all(isinstance(r, StoryPrediction) for r in rows)
        assert not any(isinstance(r, ExtractionPrediction) for r in rows)

    def test_quality_mode_fires_no_theme_llm_calls(self):
        """--mode quality must not pay for theme calls (and vice versa)."""
        llm = _full_llm()
        rows, _thinking = _run(run_evaluation(
            self._golden(), _classifier(llm), mode="quality", parallelism=1,
        ))
        assert not any("THEME" in user for _sys, user in llm.calls)
        assert rows[0].predicted_scores is not None
        assert rows[0].predicted_themes == {}
        assert rows[0].themes_raw is None

    def test_themes_mode_fires_no_quality_llm_calls(self):
        llm = _full_llm()
        rows, _thinking = _run(run_evaluation(
            self._golden(), _classifier(llm), mode="themes", parallelism=1,
        ))
        assert not any("QUALITY" in user for _sys, user in llm.calls)
        assert rows[0].predicted_themes == {"scripture": "present"}
        assert rows[0].predicted_scores is None
        assert rows[0].predicted_tier is None
        assert rows[0].scores_raw is None

    def test_unlabeled_story_skipped_in_story_modes(self):
        """A Phase-A-only story (reviewed=False) has no ground truth: don't
        spend LLM calls on it or score its empty expected_themes."""
        golden = [_newsletter(
            "nl1", reviewed=True,
            stories=[
                _story("nl1:0", text="body", reviewed=True),
                _story("nl1:1", text="v", reviewed=False),
            ],
        )]
        rows, _thinking = _run(run_evaluation(
            golden, _classifier(_full_llm()), mode="quality", parallelism=1,
        ))
        assert [r.story_id for r in rows] == ["nl1:0"]


class TestSelectStories:
    def test_counts_excluded_and_unlabeled_skips(self):
        from evals.newsletter_run import select_stories
        golden = [_newsletter(
            "nl1", reviewed=True,
            stories=[
                _story("nl1:0", reviewed=True),
                _story("nl1:1", reviewed=False),
                _story("nl1:2", excluded=True),
            ],
        )]
        pairs, n_excluded, n_unlabeled = select_stories(golden)
        assert [(s.story_id, tid) for s, tid in pairs] == [("nl1:0", "nl1")]
        assert n_excluded == 1
        assert n_unlabeled == 1

    def test_excluded_story_skipped_in_quality_mode(self, tmp_path):
        cache = tmp_path / "cache.jsonl"
        cached = CachedLLMClient(_full_llm(), cache)
        golden = [_newsletter(
            "nl1", reviewed=True,
            stories=[
                _story("nl1:0", text="body"),
                _story("nl1:1", text="y", excluded=True),
            ],
        )]
        rows, _thinking = _run(run_evaluation(
            golden, _classifier(cached), mode="quality", parallelism=1,
        ))
        assert [r.story_id for r in rows] == ["nl1:0"]


class TestFormatRunSummary:
    """The run summary must surface silent failures, not just .error rows."""

    def _story_row(self, **kw):
        from evals.newsletter_schemas import StoryPrediction
        base = dict(story_id="nl1:0", thread_id="nl1")
        base.update(kw)
        return StoryPrediction(**base)

    def test_quality_parse_failure_counted(self):
        from evals.newsletter_run import format_run_summary
        rows = [self._story_row(scores_raw="garbage output", predicted_scores=None,
                                themes_raw="NONE")]
        summary = format_run_summary(rows)
        assert "1 quality parse failure" in summary
        assert "no errors" not in summary

    def test_skipped_quality_is_not_a_parse_failure(self):
        from evals.newsletter_run import format_run_summary
        # themes-mode row: quality never attempted (scores_raw None)
        rows = [self._story_row(scores_raw=None, predicted_scores=None,
                                themes_raw="NONE")]
        assert "quality parse failure" not in format_run_summary(rows)

    def test_garbage_theme_response_is_a_parse_failure(self):
        from evals.newsletter_run import format_run_summary
        rows = [self._story_row(
            scores_raw="SIMPLE: OK\nCONCRETE: OK\nPERSONAL: OK\nDYNAMIC: OK",
            predicted_scores={"simple": 2, "concrete": 2, "personal": 2, "dynamic": 2},
            themes_raw="just some prose, no labels", predicted_themes={},
        )]
        assert "1 theme parse failure" in format_run_summary(rows)

    def test_none_theme_response_is_not_a_failure(self):
        from evals.newsletter_run import format_run_summary
        rows = [self._story_row(
            scores_raw="SIMPLE: OK\nCONCRETE: OK\nPERSONAL: OK\nDYNAMIC: OK",
            predicted_scores={"simple": 2, "concrete": 2, "personal": 2, "dynamic": 2},
            themes_raw="NONE", predicted_themes={},
        )]
        assert "theme parse failure" not in format_run_summary(rows)

    def test_all_absent_theme_response_is_not_a_failure(self):
        # A response grading every theme ABSENT is a valid empty result (issue
        # #53), not garbage — it must not count as a theme parse failure.
        from evals.newsletter_run import format_run_summary
        rows = [self._story_row(
            scores_raw="SIMPLE: OK\nCONCRETE: OK\nPERSONAL: OK\nDYNAMIC: OK",
            predicted_scores={"simple": 2, "concrete": 2, "personal": 2, "dynamic": 2},
            themes_raw=(
                "SCRIPTURE: ABSENT\nCHRISTLIKENESS: ABSENT\nCHURCH: ABSENT\n"
                "VOCATION_FAMILY: ABSENT\nDISCIPLE_MAKING: ABSENT"
            ),
            predicted_themes={},
        )]
        assert "theme parse failure" not in format_run_summary(rows)

    def test_dropped_theme_tokens_surfaced(self):
        from evals.newsletter_run import format_run_summary
        rows = [self._story_row(
            scores_raw="SIMPLE: OK\nCONCRETE: OK\nPERSONAL: OK\nDYNAMIC: OK",
            predicted_scores={"simple": 2, "concrete": 2, "personal": 2, "dynamic": 2},
            themes_raw="FELLOWSHIP: PRESENT\nCHURCH: EMPHASIZED",
            predicted_themes={"church": "emphasized"},
        )]
        assert "FELLOWSHIP" in format_run_summary(rows)

    def test_theme_near_miss_line_surfaced(self):
        # A near-miss line (misspelled grade) that parse_themes silently drops
        # must surface — otherwise a systematic parser gap reads as a model miss
        # (Finding 3). The response otherwise looks fine (one valid grading), so
        # it is NOT a garbage/empty parse.
        from evals.newsletter_run import format_run_summary
        rows = [self._story_row(
            scores_raw="SIMPLE: OK\nCONCRETE: OK\nPERSONAL: OK\nDYNAMIC: OK",
            predicted_scores={"simple": 2, "concrete": 2, "personal": 2, "dynamic": 2},
            themes_raw="SCRIPTURE: PRESENT\nCHURCH: EMPHASISED",
            predicted_themes={"scripture": "present"},
        )]
        summary = format_run_summary(rows)
        assert "EMPHASISED" in summary
        assert "theme parse failure" not in summary  # not garbage: it had a grading

    def test_bulleted_valid_themes_are_not_a_near_miss(self):
        # Coherence with Finding 1: once parse_themes tolerates bullets, a bulleted
        # VALID response must be clean — no near-miss, no parse failure.
        from evals.newsletter_run import format_run_summary
        rows = [self._story_row(
            scores_raw="SIMPLE: OK\nCONCRETE: OK\nPERSONAL: OK\nDYNAMIC: OK",
            predicted_scores={"simple": 2, "concrete": 2, "personal": 2, "dynamic": 2},
            themes_raw="- SCRIPTURE: PRESENT\n- CHURCH: EMPHASIZED",
            predicted_themes={"scripture": "present", "church": "emphasized"},
        )]
        assert "no errors" in format_run_summary(rows)

    def test_error_grammar_is_singular(self):
        from evals.newsletter_run import format_run_summary
        rows = [self._story_row(error="boom")]
        summary = format_run_summary(rows)
        assert "1 error" in summary
        assert "1 errors" not in summary

    def test_clean_run_reports_no_errors(self):
        from evals.newsletter_run import format_run_summary
        rows = [self._story_row(
            scores_raw="SIMPLE: OK\nCONCRETE: OK\nPERSONAL: OK\nDYNAMIC: OK",
            predicted_scores={"simple": 2, "concrete": 2, "personal": 2, "dynamic": 2},
            themes_raw="SCRIPTURE: PRESENT", predicted_themes={"scripture": "present"},
        )]
        summary = format_run_summary(rows)
        assert "Rows: 1" in summary
        assert "no errors" in summary


def _full_config():
    cfg = _prompts_config()
    cfg["newsletter"]["llm"] = {
        "model": "claude-x", "temperature": 0.0, "max_tokens": 1024, "extra_body": None,
    }
    return cfg


class TestBuildMeta:
    def test_records_prompt_hash_model_and_system_prompts(self):
        cfg = _full_config()
        golden = [_newsletter("nl1", reviewed=True,
                              stories=[_story("nl1:0"), _story("nl1:1")])]
        meta = build_meta(
            config=cfg, config_path="config.toml", golden_path="g.jsonl",
            golden_set=golden, mode="all", tag="baseline", parallelism=2,
        )
        assert meta.prompt_hash == compute_prompt_hash(cfg)
        assert meta.newsletter_model == "claude-x"
        assert meta.mode == "all"
        assert meta.tag == "baseline"
        assert meta.golden_set_count == 1
        assert meta.story_count == 2
        assert meta.parallelism == 2
        assert meta.extraction_system_prompt == "extract sys"
        assert meta.quality_system_prompt == "quality sys"
        assert meta.theme_system_prompt == "theme sys"
        assert meta.temperature == 0.0
        assert meta.max_tokens == 1024

    def test_story_count_counts_only_labeled_stories(self):
        """story_count must match what a story-mode run actually evaluates,
        so the report header can't disagree with metric denominators."""
        golden = [_newsletter(
            "nl1", reviewed=True,
            stories=[
                _story("nl1:0", reviewed=True),
                _story("nl1:1", reviewed=False),
                _story("nl1:2", excluded=True),
            ],
        )]
        meta = build_meta(
            config=_full_config(), config_path="config.toml", golden_path="g.jsonl",
            golden_set=golden, mode="all", tag="", parallelism=1,
        )
        assert meta.story_count == 1

    def test_extraction_mode_counts_all_golden_stories(self):
        """An extraction-only run matches predictions against EVERY golden
        story (evaluate_extraction does not filter on story.reviewed), so an
        --include-unreviewed extraction run must not record story_count=0 —
        the header would disagree with what the run actually evaluated."""
        golden = [_newsletter(
            "nl1", reviewed=False,
            stories=[
                _story("nl1:0", reviewed=False),
                _story("nl1:1", reviewed=False),
            ],
        )]
        meta = build_meta(
            config=_full_config(), config_path="config.toml", golden_path="g.jsonl",
            golden_set=golden, mode="extraction", tag="", parallelism=1,
        )
        assert meta.golden_set_count == 1
        assert meta.story_count == 2


class TestBuildOutputPath:
    def test_filename_has_mode_tag_and_runid(self):
        p = build_output_path(Path("/out"), "all", "baseline", "abcdef1234567890")
        assert p.parent == Path("/out")
        assert p.name.endswith(".jsonl")
        assert "_all_" in p.name
        assert "baseline" in p.name
        assert "abcdef12" in p.name  # first 8 of run id

    def test_omits_empty_tag(self):
        p = build_output_path(Path("/out"), "quality", "", "abcdef1234567890")
        assert "quality" in p.name
        assert "__" not in p.name.replace(".jsonl", "")


class TestWriteResults:
    def test_meta_first_then_rows_roundtrip(self, tmp_path):
        from evals.newsletter_schemas import (
            ExtractionPrediction,
            NewsletterRunMeta,
            StoryPrediction,
        )
        meta = NewsletterRunMeta(
            run_id="r", timestamp="t", config_hash="c", config_path="p",
            newsletter_model="m", golden_set_path="g", golden_set_count=1, story_count=1,
        )
        rows = [
            ExtractionPrediction(thread_id="nl1"),
            StoryPrediction(story_id="nl1:0", thread_id="nl1"),
        ]
        path = tmp_path / "out.jsonl"
        write_results(path, meta, rows)
        lines = path.read_text().splitlines()
        assert json.loads(lines[0])["type"] == "run_meta"
        assert json.loads(lines[1])["type"] == "extraction_prediction"
        assert json.loads(lines[2])["type"] == "story_prediction"


class TestWriteThinkingSidecar:
    def test_writes_only_nonempty_entries(self, tmp_path):
        from evals.newsletter_schemas import NewsletterThinkingEntry
        path = tmp_path / "out.jsonl"
        entries = [
            NewsletterThinkingEntry(story_id="a", quality_cot="q"),
            NewsletterThinkingEntry(story_id="b"),  # empty -> skipped
        ]
        write_thinking_sidecar(path, entries)
        sidecar = path.with_suffix(".cot.jsonl")
        lines = sidecar.read_text().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["story_id"] == "a"

    def test_no_sidecar_when_all_empty(self, tmp_path):
        from evals.newsletter_schemas import NewsletterThinkingEntry
        path = tmp_path / "out.jsonl"
        write_thinking_sidecar(path, [NewsletterThinkingEntry(story_id="a")])
        assert not path.with_suffix(".cot.jsonl").exists()

    def test_returns_sidecar_path_when_written_else_none(self, tmp_path):
        """main() needs the path back so the run summary can mention the sidecar."""
        from evals.newsletter_schemas import NewsletterThinkingEntry
        path = tmp_path / "out.jsonl"
        written = write_thinking_sidecar(
            path, [NewsletterThinkingEntry(story_id="a", quality_cot="q")])
        assert written == path.with_suffix(".cot.jsonl")
        assert write_thinking_sidecar(
            tmp_path / "empty.jsonl", [NewsletterThinkingEntry(story_id="b")]) is None

    def test_sidecar_written_for_extraction_only_entry(self, tmp_path):
        """Extraction CoT alone must qualify an entry for the sidecar."""
        from evals.newsletter_schemas import NewsletterThinkingEntry
        path = tmp_path / "out.jsonl"
        entry = NewsletterThinkingEntry(thread_id="t1", extraction_cot="why I split it so")
        assert write_thinking_sidecar(path, [entry]) == path.with_suffix(".cot.jsonl")
        line = json.loads(path.with_suffix(".cot.jsonl").read_text().splitlines()[0])
        assert line["thread_id"] == "t1"
        assert line["extraction_cot"] == "why I split it so"


class TestExtractionThinking:
    def test_evaluate_extraction_returns_thinking_entry(self):
        llm = FakeLLM([("body text", "STORY: a", "extraction reasoning")])
        newsletter = _newsletter(
            "nl1", reviewed=True, stories=[_story("nl1:0")],
        )
        pred, entry = _run(evaluate_extraction(newsletter, _classifier(llm)))
        assert pred.thread_id == "nl1"
        assert pred.predicted_stories == [{"text": "a"}]
        assert entry.thread_id == "nl1"
        assert entry.extraction_cot == "extraction reasoning"
        assert entry.story_id == ""

    def test_run_evaluation_collects_extraction_thinking(self):
        llm = FakeLLM([("body text", "STORY: a", "xcot")])
        golden = [_newsletter("nl1", reviewed=True, stories=[_story("nl1:0")])]
        rows, thinking = _run(run_evaluation(
            golden, _classifier(llm), mode="extraction", parallelism=1,
        ))
        assert any(
            t.extraction_cot == "xcot" and t.thread_id == "nl1" for t in thinking
        )


def _main_args(**kw):
    base = dict(
        golden_set="", config=None, output_dir="", mode="all", tag=None,
        no_cache=True, parallelism=None, include_unreviewed=False,
        prompts=None, model=None, report=False, compare_to=None, skip_preflight=True,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def _read_meta(output_dir):
    files = sorted(Path(output_dir).glob("*.jsonl"))
    files = [f for f in files if not f.name.endswith(".cot.jsonl")]
    assert files, "no results file written"
    return json.loads(files[-1].read_text().splitlines()[0])


class TestMainPromptsOverride:
    def _patch_classifier(self, monkeypatch):
        """Route main through a FakeLLM-backed classifier so no network is hit."""
        def fake_build(config, no_cache):
            llm = _full_llm()
            return NewsletterClassifier(cloud_llm=llm, config=config), llm
        monkeypatch.setattr(newsletter_run, "build_classifier", fake_build)

    def test_prompts_override_changes_recorded_prompt_hash(
        self, tmp_path, monkeypatch
    ):
        self._patch_classifier(monkeypatch)
        golden = tmp_path / "g.jsonl"
        _write_golden(golden, [_newsletter(
            "nl1", reviewed=True, stories=[_story("nl1:0", text="body")],
        )])
        out = tmp_path / "results"

        # Baseline run over the real config.toml
        asyncio.run(newsletter_run.main(_main_args(
            golden_set=str(golden), output_dir=str(out), mode="quality", tag="base",
        )))
        base_hash = _read_meta(out)["prompt_hash"]

        # Variant run with a --prompts override that changes the quality system prompt
        alt = tmp_path / "alt.toml"
        alt.write_text(
            '[newsletter.prompts.quality_assessment]\n'
            'system = "A COMPLETELY DIFFERENT QUALITY PROMPT"\n'
        )
        out2 = tmp_path / "results2"
        asyncio.run(newsletter_run.main(_main_args(
            golden_set=str(golden), output_dir=str(out2), mode="quality",
            tag="variant", prompts=str(alt),
        )))
        variant_meta = _read_meta(out2)
        assert variant_meta["prompt_hash"] != base_hash
        assert variant_meta["quality_system_prompt"] == "A COMPLETELY DIFFERENT QUALITY PROMPT"

    def test_model_override_recorded(self, tmp_path, monkeypatch):
        self._patch_classifier(monkeypatch)
        golden = tmp_path / "g.jsonl"
        _write_golden(golden, [_newsletter(
            "nl1", reviewed=True, stories=[_story("nl1:0", text="body")],
        )])
        out = tmp_path / "results"
        asyncio.run(newsletter_run.main(_main_args(
            golden_set=str(golden), output_dir=str(out), mode="extraction",
            model="my-override-model",
        )))
        assert _read_meta(out)["newsletter_model"] == "my-override-model"

    def test_no_newsletters_exits_1(self, tmp_path, monkeypatch):
        self._patch_classifier(monkeypatch)
        golden = tmp_path / "g.jsonl"
        _write_golden(golden, [_newsletter("nl1", reviewed=False)])  # unreviewed only
        with pytest.raises(SystemExit) as excinfo:
            asyncio.run(newsletter_run.main(_main_args(
                golden_set=str(golden), output_dir=str(tmp_path / "r"),
            )))
        assert excinfo.value.code == 1


class TestUnreviewedExtractionNote:
    """--include-unreviewed extraction must warn that uncurated newsletters'
    empty story lists are not real ground truth."""

    def test_note_when_unreviewed_newsletters_in_extraction_mode(self):
        from evals.newsletter_run import format_unreviewed_note
        golden = [_newsletter("a", reviewed=True), _newsletter("b", reviewed=False)]
        note = format_unreviewed_note(golden, "extraction")
        assert note is not None
        assert "1" in note
        assert "false positive" in note

    def test_no_note_when_all_reviewed_or_story_mode(self):
        from evals.newsletter_run import format_unreviewed_note
        golden = [_newsletter("a", reviewed=True)]
        assert format_unreviewed_note(golden, "extraction") is None
        unrev = [_newsletter("b", reviewed=False)]
        assert format_unreviewed_note(unrev, "quality") is None

    def test_main_prints_note_with_include_unreviewed(
        self, tmp_path, monkeypatch, capsys
    ):
        def fake_build(config, no_cache):
            llm = _full_llm()
            return NewsletterClassifier(cloud_llm=llm, config=config), llm
        monkeypatch.setattr(newsletter_run, "build_classifier", fake_build)
        golden = tmp_path / "g.jsonl"
        _write_golden(golden, [_newsletter("nl1", reviewed=False)])
        asyncio.run(newsletter_run.main(_main_args(
            golden_set=str(golden), output_dir=str(tmp_path / "r"),
            mode="extraction", include_unreviewed=True,
        )))
        assert "false positive" in capsys.readouterr().err


class TestMainArgErrors:
    """Typo-class mistakes must print one-line errors, not tracebacks."""

    def test_missing_prompts_file_exits_cleanly(self, tmp_path, capsys):
        golden = tmp_path / "g.jsonl"
        _write_golden(golden, [_newsletter("nl1", reviewed=True)])
        with pytest.raises(SystemExit) as excinfo:
            asyncio.run(newsletter_run.main(_main_args(
                golden_set=str(golden), output_dir=str(tmp_path / "r"),
                prompts=str(tmp_path / "nonexistent.toml"),
            )))
        assert excinfo.value.code == 1
        assert "Error" in capsys.readouterr().err

    def test_malformed_prompts_toml_exits_cleanly(self, tmp_path, capsys):
        golden = tmp_path / "g.jsonl"
        _write_golden(golden, [_newsletter("nl1", reviewed=True)])
        bad = tmp_path / "bad.toml"
        bad.write_text("[newsletter\nnot toml")
        with pytest.raises(SystemExit) as excinfo:
            asyncio.run(newsletter_run.main(_main_args(
                golden_set=str(golden), output_dir=str(tmp_path / "r"),
                prompts=str(bad),
            )))
        assert excinfo.value.code == 1
        assert "Error" in capsys.readouterr().err

    def test_missing_golden_set_exits_cleanly(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as excinfo:
            asyncio.run(newsletter_run.main(_main_args(
                golden_set=str(tmp_path / "nonexistent.jsonl"),
                output_dir=str(tmp_path / "r"),
            )))
        assert excinfo.value.code == 1
        assert "Error" in capsys.readouterr().err


class TestEndpointNaming:
    def test_describe_endpoint_prefers_newsletter_var(self, monkeypatch):
        from evals.newsletter_run import describe_endpoint
        monkeypatch.setenv("NEWSLETTER_LLM_URL", "http://nl:1/v1")
        assert describe_endpoint() == "http://nl:1/v1 (from NEWSLETTER_LLM_URL)"
        monkeypatch.delenv("NEWSLETTER_LLM_URL")
        monkeypatch.setenv("CLOUD_LLM_URL", "http://cloud:2/v1")
        assert describe_endpoint() == "http://cloud:2/v1 (from CLOUD_LLM_URL)"

    def test_per_row_error_names_endpoint(self):
        class BoomLLM(FakeLLM):
            base_url = "http://127.0.0.1:8499/v1/chat/completions"

            async def complete(self, *a, **kw):
                raise RuntimeError("LLM endpoint model-x unavailable")

        story = _story("nl1:0", text="x")
        pred, _thinking = _run(evaluate_story(story, "nl1", _classifier(BoomLLM([]))))
        assert pred.error is not None
        assert "http://127.0.0.1:8499" in pred.error

    def test_preflight_error_names_url_and_env_var(
        self, tmp_path, monkeypatch, capsys
    ):
        class DeadLLM(FakeLLM):
            async def is_available(self, timeout=None):
                return False

        def fake_build(config, no_cache):
            llm = DeadLLM([])
            return NewsletterClassifier(cloud_llm=llm, config=config), llm
        monkeypatch.setattr(newsletter_run, "build_classifier", fake_build)
        monkeypatch.setenv("NEWSLETTER_LLM_URL", "http://127.0.0.1:8499/v1")
        golden = tmp_path / "g.jsonl"
        _write_golden(golden, [_newsletter(
            "nl1", reviewed=True, stories=[_story("nl1:0", text="body")],
        )])
        with pytest.raises(SystemExit):
            asyncio.run(newsletter_run.main(_main_args(
                golden_set=str(golden), output_dir=str(tmp_path / "r"),
                skip_preflight=False,
            )))
        err = capsys.readouterr().err
        assert "http://127.0.0.1:8499/v1" in err
        assert "NEWSLETTER_LLM_URL" in err

    def test_preflight_404_shows_model_mismatch_hint(self, tmp_path, monkeypatch, capsys):
        # Issue #41 item 7: the model-mismatch hint appears only on an actual 404.
        class Mismatch(FakeLLM):
            async def probe(self, timeout=None):
                from llm_client import AvailabilityResult
                return AvailabilityResult(ok=False, status_code=404)

        def fake_build(config, no_cache):
            llm = Mismatch([])
            return NewsletterClassifier(cloud_llm=llm, config=config), llm
        monkeypatch.setattr(newsletter_run, "build_classifier", fake_build)
        golden = tmp_path / "g.jsonl"
        _write_golden(golden, [_newsletter("nl1", reviewed=True, stories=[_story("nl1:0", text="b")])])
        with pytest.raises(SystemExit):
            asyncio.run(newsletter_run.main(_main_args(
                golden_set=str(golden), output_dir=str(tmp_path / "r"), skip_preflight=False,
            )))
        assert "model name does not match" in capsys.readouterr().err

    def test_preflight_non_404_status_shown(self, tmp_path, monkeypatch, capsys):
        # Finding 1: a non-404 status (e.g. 401 bad key) must be surfaced so the
        # user can distinguish a bad key from a down server or a name mismatch.
        class BadKey(FakeLLM):
            async def probe(self, timeout=None):
                from llm_client import AvailabilityResult
                return AvailabilityResult(ok=False, status_code=401)

        def fake_build(config, no_cache):
            llm = BadKey([])
            return NewsletterClassifier(cloud_llm=llm, config=config), llm
        monkeypatch.setattr(newsletter_run, "build_classifier", fake_build)
        golden = tmp_path / "g.jsonl"
        _write_golden(golden, [_newsletter("nl1", reviewed=True, stories=[_story("nl1:0", text="b")])])
        with pytest.raises(SystemExit):
            asyncio.run(newsletter_run.main(_main_args(
                golden_set=str(golden), output_dir=str(tmp_path / "r"), skip_preflight=False,
            )))
        err = capsys.readouterr().err
        assert "401" in err
        assert "model name does not match" not in err

    def test_preflight_connection_error_shown(self, tmp_path, monkeypatch, capsys):
        # Finding 1: probe()'s captured exception text must appear so a down
        # server / bad URL is distinguishable from a bad key.
        class Refused(FakeLLM):
            async def probe(self, timeout=None):
                from llm_client import AvailabilityResult
                return AvailabilityResult(ok=False, error="ConnectError: Connection refused")

        def fake_build(config, no_cache):
            llm = Refused([])
            return NewsletterClassifier(cloud_llm=llm, config=config), llm
        monkeypatch.setattr(newsletter_run, "build_classifier", fake_build)
        golden = tmp_path / "g.jsonl"
        _write_golden(golden, [_newsletter("nl1", reviewed=True, stories=[_story("nl1:0", text="b")])])
        with pytest.raises(SystemExit):
            asyncio.run(newsletter_run.main(_main_args(
                golden_set=str(golden), output_dir=str(tmp_path / "r"), skip_preflight=False,
            )))
        assert "ConnectError" in capsys.readouterr().err

    def test_preflight_down_omits_model_mismatch_hint(self, tmp_path, monkeypatch, capsys):
        # No HTTP response (status None) -> plain unreachable, no model-mismatch hint.
        class Down(FakeLLM):
            async def is_available(self, timeout=None):
                return False

        def fake_build(config, no_cache):
            llm = Down([])
            return NewsletterClassifier(cloud_llm=llm, config=config), llm
        monkeypatch.setattr(newsletter_run, "build_classifier", fake_build)
        golden = tmp_path / "g.jsonl"
        _write_golden(golden, [_newsletter("nl1", reviewed=True, stories=[_story("nl1:0", text="b")])])
        with pytest.raises(SystemExit):
            asyncio.run(newsletter_run.main(_main_args(
                golden_set=str(golden), output_dir=str(tmp_path / "r"), skip_preflight=False,
            )))
        assert "model name does not match" not in capsys.readouterr().err


class TestReportForwarding:
    def _meta(self):
        from evals.newsletter_schemas import NewsletterRunMeta
        return NewsletterRunMeta(
            run_id="r", timestamp="t", config_hash="c", config_path="p",
            newsletter_model="m", golden_set_path="g", golden_set_count=0,
            story_count=0,
        )

    def test_maybe_report_forwards_verbose_and_threshold(self, monkeypatch):
        from evals import newsletter_report
        from evals.newsletter_run import maybe_report
        calls = {}

        def fake_compute(story, extraction, match_threshold=0.6):
            calls["compute_thr"] = match_threshold
            return {}

        def fake_print_report(meta, metrics, verbose=False, story_results=None,
                              extraction_results=None, match_threshold=0.6):
            calls["verbose"] = verbose
            calls["print_thr"] = match_threshold

        monkeypatch.setattr(newsletter_report, "compute_all_metrics", fake_compute)
        monkeypatch.setattr(newsletter_report, "print_report", fake_print_report)
        maybe_report(self._meta(), [], True, None, verbose=True, match_threshold=0.9)
        assert calls == {"compute_thr": 0.9, "verbose": True, "print_thr": 0.9}

    def test_comparison_verbosity_follows_flag(self, tmp_path, monkeypatch):
        from evals import newsletter_report
        from evals.newsletter_run import maybe_report, write_results
        calls = {}

        def fake_print_comparison(meta1, metrics1, meta2, metrics2, verbose=False,
                                  story1=None, story2=None):
            calls["verbose"] = verbose

        monkeypatch.setattr(newsletter_report, "print_comparison", fake_print_comparison)
        prior = tmp_path / "prior.jsonl"
        write_results(prior, self._meta(), [])
        maybe_report(self._meta(), [], False, str(prior), verbose=False)
        assert calls["verbose"] is False

    def test_cli_accepts_verbose_and_match_threshold(self, monkeypatch):
        import sys as _sys
        seen = {}

        async def fake_main(args):
            seen["args"] = args

        monkeypatch.setattr(newsletter_run, "main", fake_main)
        monkeypatch.setattr(_sys, "argv",
                            ["prog", "--verbose", "--match-threshold", "0.8"])
        newsletter_run.cli()
        assert seen["args"].verbose is True
        assert seen["args"].match_threshold == 0.8

    def test_cli_quiets_httpx_logging(self, monkeypatch):
        import logging
        import sys as _sys

        async def fake_main(args):
            return None

        monkeypatch.setattr(newsletter_run, "main", fake_main)
        monkeypatch.setattr(_sys, "argv", ["prog"])
        logging.getLogger("httpx").setLevel(logging.NOTSET)
        newsletter_run.cli()
        assert logging.getLogger("httpx").level == logging.WARNING


class TestCliParallelismValidation:
    def test_cli_rejects_non_positive_parallelism(self, monkeypatch, capsys):
        """--parallelism 0 would build asyncio.Semaphore(0): every task blocks
        forever and the run hangs silently after preflight. The parser must
        reject it up front instead."""
        import sys as _sys
        called = {}

        async def fake_main(args):
            called["ran"] = True

        monkeypatch.setattr(newsletter_run, "main", fake_main)
        for bad in ("0", "-2"):
            monkeypatch.setattr(_sys, "argv", ["prog", "--parallelism", bad])
            with pytest.raises(SystemExit) as excinfo:
                newsletter_run.cli()
            assert excinfo.value.code == 2  # argparse usage error
            assert "parallelism" in capsys.readouterr().err
        assert "ran" not in called


class TestProgress:
    def test_progress_lines_written_when_enabled(self, capsys):
        golden = [_newsletter(
            "nl1", reviewed=True,
            stories=[_story("nl1:0", text="body")],
        )]
        _run(run_evaluation(
            golden, _classifier(_full_llm()), mode="all", parallelism=1,
            progress=True,
        ))
        err = capsys.readouterr().err
        assert "[1/2]" in err
        assert "[2/2]" in err

    def test_no_progress_output_by_default(self, capsys):
        golden = [_newsletter(
            "nl1", reviewed=True,
            stories=[_story("nl1:0", text="body")],
        )]
        _run(run_evaluation(golden, _classifier(_full_llm()), mode="all", parallelism=1))
        assert "[1/2]" not in capsys.readouterr().err


class TestMainPreflight:
    def test_unreachable_endpoint_exits_1(self, tmp_path, monkeypatch):
        class DeadLLM(FakeLLM):
            async def is_available(self, timeout=None):
                return False

        def fake_build(config, no_cache):
            llm = DeadLLM([])
            return NewsletterClassifier(cloud_llm=llm, config=config), llm
        monkeypatch.setattr(newsletter_run, "build_classifier", fake_build)

        golden = tmp_path / "g.jsonl"
        _write_golden(golden, [_newsletter(
            "nl1", reviewed=True, stories=[_story("nl1:0", text="body")],
        )])
        args = _main_args(golden_set=str(golden), output_dir=str(tmp_path / "r"),
                          skip_preflight=False)
        with pytest.raises(SystemExit) as excinfo:
            asyncio.run(newsletter_run.main(args))
        assert excinfo.value.code == 1

    def test_skip_preflight_bypasses_check(self, tmp_path, monkeypatch):
        class DeadLLM(FakeLLM):
            async def is_available(self, timeout=None):
                return False

        def fake_build(config, no_cache):
            llm = DeadLLM([
                ("EXTRACT", "NO_STORIES", ""),
                ("QUALITY", "SIMPLE: OK\nCONCRETE: OK\nPERSONAL: OK\nDYNAMIC: OK", ""),
                ("THEME", "NONE", ""),
            ])
            return NewsletterClassifier(cloud_llm=llm, config=config), llm
        monkeypatch.setattr(newsletter_run, "build_classifier", fake_build)

        golden = tmp_path / "g.jsonl"
        _write_golden(golden, [_newsletter(
            "nl1", reviewed=True, stories=[_story("nl1:0", text="body")],
        )])
        out = tmp_path / "r"
        # Endpoint "down" but --skip-preflight -> main runs to completion and writes.
        asyncio.run(newsletter_run.main(_main_args(
            golden_set=str(golden), output_dir=str(out), skip_preflight=True,
        )))
        assert _read_meta(out)["type"] == "run_meta"


class TestMainReport:
    """--report / --compare-to must invoke the real newsletter_report API.

    Guards against the maybe_report() shim drifting from newsletter_report
    (which exposes compute_all_metrics + a 3-tuple load_results, not the
    email report's compute_metrics + 2-tuple load_results).
    """

    def _patch_classifier(self, monkeypatch):
        def fake_build(config, no_cache):
            llm = _full_llm()
            return NewsletterClassifier(cloud_llm=llm, config=config), llm
        monkeypatch.setattr(newsletter_run, "build_classifier", fake_build)

    def test_report_flag_prints_metrics(self, tmp_path, monkeypatch, capsys):
        self._patch_classifier(monkeypatch)
        golden = tmp_path / "g.jsonl"
        _write_golden(golden, [_newsletter(
            "nl1", reviewed=True,
            stories=[_story(
                "nl1:0", text="body",
                expected_scores={"simple": 3, "concrete": 3, "personal": 3, "dynamic": 3},
                expected_tier="good", expected_themes={"scripture": "present"},
            )],
        )])
        out = tmp_path / "r"
        # Must not raise (AttributeError on compute_metrics, tuple-unpack, etc.).
        asyncio.run(newsletter_run.main(_main_args(
            golden_set=str(golden), output_dir=str(out), mode="all", report=True,
        )))
        printed = capsys.readouterr().out
        assert "Newsletter Evaluation Report" in printed

    def test_compare_to_prints_comparison(self, tmp_path, monkeypatch, capsys):
        self._patch_classifier(monkeypatch)
        golden = tmp_path / "g.jsonl"
        _write_golden(golden, [_newsletter(
            "nl1", reviewed=True,
            stories=[_story(
                "nl1:0", text="body",
                expected_scores={"simple": 3, "concrete": 3, "personal": 3, "dynamic": 3},
                expected_tier="good", expected_themes={"scripture": "present"},
            )],
        )])
        # First run to produce a prior results file to compare against.
        out1 = tmp_path / "r1"
        asyncio.run(newsletter_run.main(_main_args(
            golden_set=str(golden), output_dir=str(out1), mode="all",
        )))
        prior = sorted(
            f for f in Path(out1).glob("*.jsonl") if not f.name.endswith(".cot.jsonl")
        )[-1]

        out2 = tmp_path / "r2"
        asyncio.run(newsletter_run.main(_main_args(
            golden_set=str(golden), output_dir=str(out2), mode="all",
            compare_to=str(prior),
        )))
        printed = capsys.readouterr().out
        assert "Newsletter Comparison Report" in printed

    def test_compare_to_meta_less_file_does_not_crash_run(
        self, tmp_path, monkeypatch, capsys
    ):
        """A bad --compare-to file (e.g. a .cot.jsonl sidecar) must not raise
        after the paid run, and the results path must still be printed."""
        self._patch_classifier(monkeypatch)
        golden = tmp_path / "g.jsonl"
        _write_golden(golden, [_newsletter(
            "nl1", reviewed=True, stories=[_story("nl1:0", text="body")],
        )])
        bad = tmp_path / "bad.cot.jsonl"
        bad.write_text('{"type": "thinking", "story_id": "x"}\n')
        out = tmp_path / "r"
        asyncio.run(newsletter_run.main(_main_args(
            golden_set=str(golden), output_dir=str(out), mode="all",
            compare_to=str(bad),
        )))
        err = capsys.readouterr().err
        assert "Error" in err and "compare" in err
        assert "Results written to" in err

    def test_summary_mentions_cot_sidecar(self, tmp_path, monkeypatch, capsys):
        # Patch with the needle-matching prompts config so the FakeLLM returns
        # thinking content and the .cot.jsonl sidecar actually gets written.
        def fake_build(config, no_cache):
            llm = _full_llm()
            return NewsletterClassifier(cloud_llm=llm, config=_prompts_config()), llm
        monkeypatch.setattr(newsletter_run, "build_classifier", fake_build)
        golden = tmp_path / "g.jsonl"
        _write_golden(golden, [_newsletter(
            "nl1", reviewed=True, stories=[_story("nl1:0", text="body")],
        )])
        out = tmp_path / "r"
        asyncio.run(newsletter_run.main(_main_args(
            golden_set=str(golden), output_dir=str(out), mode="all",
        )))
        err = capsys.readouterr().err
        assert "Chain-of-thought written to" in err
        assert ".cot.jsonl" in err

    def test_main_reports_unlabeled_story_skips(self, tmp_path, monkeypatch, capsys):
        self._patch_classifier(monkeypatch)
        golden = tmp_path / "g.jsonl"
        _write_golden(golden, [_newsletter(
            "nl1", reviewed=True,
            stories=[
                _story("nl1:0", text="body", reviewed=True),
                _story("nl1:1", text="v", reviewed=False),
            ],
        )])
        out = tmp_path / "r"
        asyncio.run(newsletter_run.main(_main_args(
            golden_set=str(golden), output_dir=str(out), mode="quality",
        )))
        err = capsys.readouterr().err
        assert "1 story" in err
        assert "unlabeled" in err or "never labeled" in err


class TestCliQuietsHttpLogging:
    def test_cli_quiets_httpx_and_httpcore(self, monkeypatch):
        """cli() must silence BOTH httpx and httpcore per-request INFO logs
        (via the shared daemon.quiet_http_logging helper — the inline copy it
        replaced missed httpcore)."""
        for name in ("httpx", "httpcore"):
            monkeypatch.setattr(logging.getLogger(name), "level", logging.NOTSET)
        monkeypatch.setattr(newsletter_run, "main", AsyncMock())
        monkeypatch.setattr(sys, "argv", ["newsletter_run"])
        newsletter_run.cli()
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING


def _label(text, thread_id="t_900010", status="graded", labeled_at="2026-11-05T21:00:00+00:00",
           **extra):
    fields = {
        "story_key": story_key(text), "thread_id": thread_id, "text": text,
        "status": status, "labeled_at": labeled_at,
    }
    if status == "graded":
        fields["scores"] = {"simple": 3, "concrete": 3, "personal": 2, "dynamic": 2}
        fields["themes"] = {"scripture": "emphasized"}
    else:
        fields["miscut_reason"] = "cut_short"
    fields.update(extra)
    return StoryLabel.from_dict(fields)


class TestLoadStoryLabels:
    def test_a_missing_log_is_no_stories(self, tmp_path):
        pairs, stats = load_story_labels(tmp_path / "labels.jsonl", set())
        assert pairs == []
        assert stats == {"graded": 0, "miscut": 0, "in_curated_threads": 0}

    def test_graded_labels_become_ground_truth_stories(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        append_label(path, _label("A synthetic labeled story."))
        pairs, stats = load_story_labels(path, set())
        ((story, thread_id),) = pairs
        assert thread_id == "t_900010"
        assert story.story_id == story_key("A synthetic labeled story.")
        assert story.text == "A synthetic labeled story."
        assert story.expected_scores == {"simple": 3, "concrete": 3, "personal": 2, "dynamic": 2}
        assert story.expected_themes == {"scripture": "emphasized"}
        assert story.expected_tier == "good"
        assert story.reviewed and not story.excluded
        assert stats["graded"] == 1

    def test_miscut_stories_are_not_ground_truth(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        append_label(path, _label("A synthetic miscut story.", status="miscut"))
        pairs, stats = load_story_labels(path, set())
        assert pairs == []
        assert stats["miscut"] == 1

    def test_only_the_newest_label_per_story_counts(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        append_label(path, _label("A story.", labeled_at="2026-11-01T10:00:00+00:00"))
        append_label(path, _label("A story.", status="miscut",
                                  labeled_at="2026-11-02T10:00:00+00:00"))
        pairs, stats = load_story_labels(path, set())
        assert pairs == []
        assert stats == {"graded": 0, "miscut": 1, "in_curated_threads": 0}

    def test_stories_from_curated_threads_are_left_to_the_curated_set(self, tmp_path):
        # A thread curated after one of its stories was graded here would
        # otherwise be scored twice.
        path = tmp_path / "labels.jsonl"
        append_label(path, _label("A story.", thread_id="nl1"))
        append_label(path, _label("Another story.", thread_id="t_900011"))
        pairs, stats = load_story_labels(path, {"nl1"})
        assert [thread_id for _story_, thread_id in pairs] == ["t_900011"]
        assert stats["in_curated_threads"] == 1


class TestStoryLabelsInTheEvaluation:
    def _golden(self):
        return [_newsletter("nl1", reviewed=True, stories=[_story("nl1:0", text="body")])]

    def _extra(self):
        return [(_label("LABELED story text").to_golden_story(), "t_900010")]

    def test_label_log_stories_are_scored_alongside_curated_ones(self):
        rows, _thinking = _run(run_evaluation(
            self._golden(), _classifier(_full_llm()), mode="quality", parallelism=1,
            extra_stories=self._extra(),
        ))
        by_id = {row.story_id: row for row in rows}
        assert set(by_id) == {"nl1:0", story_key("LABELED story text")}
        assert by_id["nl1:0"].source == "curated"
        labeled = by_id[story_key("LABELED story text")]
        assert labeled.source == "story_labels"
        assert labeled.thread_id == "t_900010"
        assert labeled.expected_scores == {"simple": 3, "concrete": 3, "personal": 2, "dynamic": 2}
        assert labeled.predicted_scores is not None

    def test_label_log_stories_are_never_extraction_truth(self):
        # Their boundaries were drawn by the extractor; scoring the extractor
        # against them would be circular.
        from evals.newsletter_schemas import ExtractionPrediction
        rows, _thinking = _run(run_evaluation(
            self._golden(), _classifier(_full_llm()), mode="extraction", parallelism=1,
            extra_stories=self._extra(),
        ))
        assert [type(row) for row in rows] == [ExtractionPrediction]
        assert rows[0].thread_id == "nl1"

    def test_source_defaults_to_curated_for_result_files_written_before_it_existed(self):
        from evals.newsletter_schemas import StoryPrediction
        row = StoryPrediction.from_dict({"story_id": "nl1:0", "thread_id": "nl1"})
        assert row.source == "curated"
        assert StoryPrediction.from_dict(row.to_dict()).source == "curated"


class TestMainStoryLabels:
    def _patch(self, monkeypatch):
        def fake_build(config, no_cache):
            llm = FakeLLM([("story", "SIMPLE: GOOD\nCONCRETE: GOOD\nPERSONAL: OK\nDYNAMIC: OK", "q")])
            return NewsletterClassifier(cloud_llm=llm, config=config), llm
        monkeypatch.setattr(newsletter_run, "build_classifier", fake_build)

    def _rows(self, output_dir):
        files = [f for f in sorted(Path(output_dir).glob("*.jsonl"))
                 if not f.name.endswith(".cot.jsonl")]
        return [json.loads(line) for line in files[-1].read_text().splitlines()[1:]]

    def test_a_run_scores_the_label_log_and_records_where_stories_came_from(
        self, tmp_path, monkeypatch, capsys,
    ):
        self._patch(monkeypatch)
        golden = tmp_path / "g.jsonl"
        _write_golden(golden, [_newsletter(
            "nl1", reviewed=True, stories=[_story("nl1:0", text="curated story")],
        )])
        labels = tmp_path / "labels.jsonl"
        append_label(labels, _label("labeled story one"))
        append_label(labels, _label("labeled story two", status="miscut"))
        out = tmp_path / "results"
        asyncio.run(newsletter_run.main(_main_args(
            golden_set=str(golden), output_dir=str(out), mode="quality",
            story_labels=str(labels),
        )))
        meta = _read_meta(out)
        assert meta["story_count"] == 2
        assert meta["story_label_count"] == 1
        assert meta["story_labels_path"] == str(labels)
        rows = self._rows(out)
        assert sorted(row["source"] for row in rows) == ["curated", "story_labels"]
        assert "1 graded story from the story-label log" in capsys.readouterr().err

    def test_without_a_label_log_the_run_is_as_before(self, tmp_path, monkeypatch):
        self._patch(monkeypatch)
        golden = tmp_path / "g.jsonl"
        _write_golden(golden, [_newsletter(
            "nl1", reviewed=True, stories=[_story("nl1:0", text="curated story")],
        )])
        out = tmp_path / "results"
        asyncio.run(newsletter_run.main(_main_args(
            golden_set=str(golden), output_dir=str(out), mode="quality",
            story_labels=str(tmp_path / "absent.jsonl"),
        )))
        meta = _read_meta(out)
        assert (meta["story_count"], meta["story_label_count"]) == (1, 0)


class TestLabelStoriesFromUnreviewedNewsletters:
    def test_a_label_story_is_scored_when_its_newsletter_was_only_harvested(
        self, tmp_path, monkeypatch,
    ):
        # --include-unreviewed loads harvested newsletters that carry no scored
        # story. A story graded in newsletter_grade from such a thread must not
        # be "left to the curated set", which has nothing for it.
        def fake_build(config, no_cache):
            llm = FakeLLM([("story", "SIMPLE: GOOD\nCONCRETE: GOOD\nPERSONAL: OK\nDYNAMIC: OK", "q")])
            return NewsletterClassifier(cloud_llm=llm, config=config), llm
        monkeypatch.setattr(newsletter_run, "build_classifier", fake_build)
        golden = tmp_path / "g.jsonl"
        _write_golden(golden, [
            _newsletter("nl1", reviewed=True, stories=[_story("nl1:0", text="curated story")]),
            _newsletter("nl2", reviewed=False),
        ])
        labels = tmp_path / "labels.jsonl"
        append_label(labels, _label("labeled story from nl2", thread_id="nl2"))
        append_label(labels, _label("labeled story from nl1", thread_id="nl1"))
        out = tmp_path / "results"
        asyncio.run(newsletter_run.main(_main_args(
            golden_set=str(golden), output_dir=str(out), mode="quality",
            include_unreviewed=True, story_labels=str(labels),
        )))
        meta = _read_meta(out)
        assert meta["story_label_count"] == 1   # nl2's story; nl1's is left to the curated set


class TestMainLabelLogAlone:
    """The label log is a grading golden set by itself: a machine that has only
    graded ready-made stories, with no curated newsletters, can still be scored."""

    def _patch(self, monkeypatch):
        def fake_build(config, no_cache):
            llm = FakeLLM([("story", "SIMPLE: GOOD\nCONCRETE: GOOD\nPERSONAL: OK\nDYNAMIC: OK", "q")])
            return NewsletterClassifier(cloud_llm=llm, config=config), llm
        monkeypatch.setattr(newsletter_run, "build_classifier", fake_build)

    def test_quality_runs_on_the_label_log_when_told_there_is_no_curated_set(
        self, tmp_path, monkeypatch, capsys,
    ):
        self._patch(monkeypatch)
        labels = tmp_path / "labels.jsonl"
        append_label(labels, _label("labeled story one"))
        out = tmp_path / "results"
        asyncio.run(newsletter_run.main(_main_args(
            golden_set="none", output_dir=str(out), mode="quality", story_labels=str(labels),
        )))
        meta = _read_meta(out)
        assert (meta["golden_set_count"], meta["story_count"], meta["story_label_count"]) == (0, 1, 1)
        assert "scoring the story-label log only" in capsys.readouterr().err

    def test_a_missing_golden_set_file_is_still_an_error(self, tmp_path, monkeypatch, capsys):
        # A mistyped path must not quietly turn into a paid run over the
        # label log alone. Running without a curated set has to be asked for.
        self._patch(monkeypatch)
        labels = tmp_path / "labels.jsonl"
        append_label(labels, _label("labeled story one"))
        with pytest.raises(SystemExit) as exit_info:
            asyncio.run(newsletter_run.main(_main_args(
                golden_set=str(tmp_path / "goldn_set.jsonl"), output_dir=str(tmp_path / "r"),
                mode="quality", story_labels=str(labels),
            )))
        assert exit_info.value.code == 1
        err = capsys.readouterr().err
        assert "golden set file not found" in err
        assert "--golden-set none" in err
        assert not (tmp_path / "r").exists()

    def test_an_empty_curated_set_does_not_block_the_label_log(self, tmp_path, monkeypatch):
        self._patch(monkeypatch)
        golden = tmp_path / "g.jsonl"
        _write_golden(golden, [_newsletter("nl1", reviewed=False)])
        labels = tmp_path / "labels.jsonl"
        append_label(labels, _label("labeled story one"))
        out = tmp_path / "results"
        asyncio.run(newsletter_run.main(_main_args(
            golden_set=str(golden), output_dir=str(out), mode="themes", story_labels=str(labels),
        )))
        assert _read_meta(out)["story_label_count"] == 1

    def test_no_curated_set_and_no_graded_labels_exits_1(self, tmp_path, monkeypatch, capsys):
        self._patch(monkeypatch)
        labels = tmp_path / "labels.jsonl"
        append_label(labels, _label("a miscut story", status="miscut"))
        with pytest.raises(SystemExit) as exit_info:
            asyncio.run(newsletter_run.main(_main_args(
                golden_set="none", output_dir=str(tmp_path / "r"),
                mode="quality", story_labels=str(labels),
            )))
        assert exit_info.value.code == 1
        assert "nothing to score" in capsys.readouterr().err.lower()

    def test_extraction_still_needs_the_curated_set(self, tmp_path, monkeypatch, capsys):
        self._patch(monkeypatch)
        labels = tmp_path / "labels.jsonl"
        append_label(labels, _label("labeled story one"))
        with pytest.raises(SystemExit) as exit_info:
            asyncio.run(newsletter_run.main(_main_args(
                golden_set="none", output_dir=str(tmp_path / "r"),
                mode="extraction", story_labels=str(labels),
            )))
        assert exit_info.value.code == 1
        assert "extraction" in capsys.readouterr().err


def test_the_headless_runner_does_not_load_the_terminal_ui():
    # newsletter_run is a batch tool. Reading the story-label log must not pull
    # in Textual or the grading screen with it.
    import subprocess
    probe = (
        "import sys, evals.newsletter_run; "
        "loaded = [m for m in ('textual', 'evals.newsletter_grade') if m in sys.modules]; "
        "print(loaded); sys.exit(1 if loaded else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0, f"loaded at import: {result.stdout.strip()} {result.stderr[-300:]}"
