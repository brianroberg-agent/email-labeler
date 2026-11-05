"""Tests for evals.newsletter_grade — the story-grading labeler's pure core.

All story text here is synthetic. The shapes are the caller's: assessment
records as ``newsletter.write_assessment`` writes them, label lines as the tool
appends them.
"""

import json
from datetime import date, datetime, timezone

import pytest
from textual.widgets import Input, Static

from daemon import load_config
from evals.newsletter_grade import (
    QUESTIONS,
    Candidate,
    Card,
    GradeApp,
    Rubric,
    answer,
    answer_rest_absent,
    answers_strip,
    append_label,
    back,
    build_queue,
    card_from_label,
    cli,
    count_labeled_on,
    current_question,
    format_miscut_bucket,
    graded_label,
    latest_labels,
    load_rubric,
    miscut_bucket,
    miscut_label,
    question_panel,
    read_label_log,
    story_key,
    text_overlap,
    toggle_torn,
)
from evals.newsletter_schemas import StoryLabel

STORY_A = (
    "Dana had never led anything before this fall. In September she agreed to "
    "host four freshmen in her apartment on Tuesday nights, and by October she "
    "was the one asking the questions."
)
STORY_B = (
    "The retreat drew sixty students to the lake house. There were three "
    "sessions, a bonfire, and a service project on Saturday afternoon."
)
STORY_A_CUT_SHORT = "Dana had never led anything before this fall. In September she agreed to"


def _assessment(thread_id, texts, timestamp="2026-10-30T12:00:00+00:00"):
    """A production assessment record, with the grades a real one carries."""
    return {
        "timestamp": timestamp,
        "schema_version": 1,
        "thread_id": thread_id,
        "message_id": f"m_{thread_id}",
        "from": "Synthetic Sender <sender@example.org>",
        "subject": "SENTINEL-SUBJECT",
        "model": "SENTINEL-MODEL",
        "overall_tier": "excellent",
        "stories": [
            {
                "text": text,
                "scores": {"simple": 3, "concrete": 3, "personal": 3, "dynamic": 3},
                "average_score": 3.0,
                "tier": "excellent",
                "themes": {"scripture": "emphasized"},
                "quality_cot": "SENTINEL-COT",
                "theme_cot": "SENTINEL-COT",
            }
            for text in texts
        ],
    }


def _graded(text, thread_id="t_900001", labeled_at="2026-11-05T21:00:00+00:00", **extra):
    fields = {
        "story_key": story_key(text),
        "thread_id": thread_id,
        "text": text,
        "status": "graded",
        "scores": {"simple": 2, "concrete": 2, "personal": 2, "dynamic": 2},
        "labeled_at": labeled_at,
    }
    fields.update(extra)
    return StoryLabel.from_dict(fields)


def _miscut(text, thread_id="t_900001", reason="cut_short",
            labeled_at="2026-11-05T21:00:00+00:00", **extra):
    fields = {
        "story_key": story_key(text),
        "thread_id": thread_id,
        "text": text,
        "status": "miscut",
        "miscut_reason": reason,
        "labeled_at": labeled_at,
    }
    fields.update(extra)
    return StoryLabel.from_dict(fields)


class TestStoryKey:
    def test_is_sixteen_hex_characters(self):
        key = story_key(STORY_A)
        assert len(key) == 16
        int(key, 16)

    def test_same_text_same_key(self):
        assert story_key(STORY_A) == story_key(STORY_A)

    def test_whitespace_differences_do_not_change_the_key(self):
        reflowed = "  " + STORY_A.replace(". ", ".\n\n") + "\r\n"
        assert story_key(reflowed) == story_key(STORY_A)

    def test_different_text_different_key(self):
        assert story_key(STORY_A) != story_key(STORY_B)
        assert story_key(STORY_A) != story_key(STORY_A_CUT_SHORT)


class TestLabelLog:
    def test_missing_file_reads_as_empty(self, tmp_path):
        assert read_label_log(tmp_path / "labels.jsonl") == []

    def test_append_then_read_round_trips(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        first, second = _graded(STORY_A), _miscut(STORY_B)
        append_label(path, first)
        append_label(path, second)
        assert read_label_log(path) == [first, second]

    def test_append_never_rewrites_earlier_lines(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        append_label(path, _graded(STORY_A))
        before = path.read_bytes()
        append_label(path, _graded(STORY_B))
        assert path.read_bytes().startswith(before)
        assert len(path.read_text().splitlines()) == 2

    def test_append_repairs_a_missing_trailing_newline(self, tmp_path):
        # A hand-merged or interrupted file may end without "\n"; the next
        # label must still land on its own line.
        path = tmp_path / "labels.jsonl"
        append_label(path, _graded(STORY_A))
        path.write_text(path.read_text().rstrip("\n"))
        append_label(path, _graded(STORY_B))
        assert [label.text for label in read_label_log(path)] == [STORY_A, STORY_B]

    def test_blank_lines_are_skipped(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        append_label(path, _graded(STORY_A))
        path.write_text("\n" + path.read_text() + "\n\n")
        assert len(read_label_log(path)) == 1

    def test_malformed_line_names_the_file_and_line(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        append_label(path, _graded(STORY_A))
        with open(path, "a") as f:
            f.write("{not json}\n")
        with pytest.raises(ValueError, match=r"labels\.jsonl:2"):
            read_label_log(path)

    def test_invalid_label_names_the_file_and_line(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        bad = _graded(STORY_A).to_dict()
        bad["status"] = "rejected"
        path.write_text(json.dumps(bad) + "\n")
        with pytest.raises(ValueError, match=r"labels\.jsonl:1"):
            read_label_log(path)


class TestLatestLabels:
    def test_one_entry_per_story_key(self):
        latest = latest_labels([_graded(STORY_A), _graded(STORY_B)])
        assert set(latest) == {story_key(STORY_A), story_key(STORY_B)}

    def test_later_labeled_at_wins(self):
        old = _graded(STORY_A, labeled_at="2026-11-01T10:00:00+00:00")
        new = _miscut(STORY_A, labeled_at="2026-11-02T10:00:00+00:00")
        assert latest_labels([old, new])[story_key(STORY_A)].status == "miscut"

    def test_later_labeled_at_wins_regardless_of_file_order(self):
        # Two machines' logs merged by concatenation: file order is not time order.
        old = _graded(STORY_A, labeled_at="2026-11-01T10:00:00+00:00")
        new = _miscut(STORY_A, labeled_at="2026-11-02T10:00:00+00:00")
        assert latest_labels([new, old])[story_key(STORY_A)].status == "miscut"

    def test_equal_or_unreadable_times_fall_back_to_the_later_line(self):
        first = _graded(STORY_A, labeled_at="")
        second = _miscut(STORY_A, labeled_at="")
        assert latest_labels([first, second])[story_key(STORY_A)].status == "miscut"


class TestTextOverlap:
    def test_contained_text_overlaps_fully(self):
        assert text_overlap(STORY_A_CUT_SHORT, STORY_A) == pytest.approx(1.0)

    def test_is_symmetric(self):
        assert text_overlap(STORY_A, STORY_A_CUT_SHORT) == text_overlap(STORY_A_CUT_SHORT, STORY_A)

    def test_unrelated_stories_barely_overlap(self):
        assert text_overlap(STORY_A, STORY_B) < 0.3

    def test_empty_text_overlaps_nothing(self):
        assert text_overlap("", STORY_A) == 0.0


class TestBuildQueue:
    def test_every_story_becomes_a_candidate(self):
        queue = build_queue([_assessment("t_900001", [STORY_A, STORY_B])], set(), {})
        assert {c.text for c in queue} == {STORY_A, STORY_B}
        candidate = next(c for c in queue if c.text == STORY_A)
        assert candidate == Candidate(
            story_key=story_key(STORY_A),
            thread_id="t_900001",
            text=STORY_A,
            source_timestamp="2026-10-30T12:00:00+00:00",
            recut_of="",
        )

    def test_candidates_carry_nothing_the_model_said(self):
        # Blind labeling (rubric §7): no grade, tier, reasoning, model or
        # subject may ride along into the queue.
        queue = build_queue([_assessment("t_900001", [STORY_A])], set(), {})
        dumped = json.dumps([vars(c) for c in queue])
        for leaked in ("SENTINEL", "excellent", "emphasized", "scores", "cot",
                       "Synthetic Sender", "example.org"):
            assert leaked not in dumped

    def test_threads_in_the_curated_golden_set_are_left_out(self):
        records = [_assessment("t_900001", [STORY_A]), _assessment("t_900002", [STORY_B])]
        queue = build_queue(records, {"t_900001"}, {})
        assert [c.text for c in queue] == [STORY_B]

    def test_stories_already_graded_are_left_out(self):
        labels = latest_labels([_graded(STORY_A)])
        queue = build_queue([_assessment("t_900001", [STORY_A, STORY_B])], set(), labels)
        assert [c.text for c in queue] == [STORY_B]

    def test_stories_marked_miscut_are_left_out_under_that_key(self):
        labels = latest_labels([_miscut(STORY_A)])
        queue = build_queue([_assessment("t_900001", [STORY_A])], set(), labels)
        assert queue == []

    def test_blank_stories_are_left_out(self):
        queue = build_queue([_assessment("t_900001", ["", "   \n", STORY_A])], set(), {})
        assert [c.text for c in queue] == [STORY_A]

    def test_identical_text_is_queued_once(self):
        records = [_assessment("t_900001", [STORY_A]), _assessment("t_900002", [STORY_A])]
        assert len(build_queue(records, set(), {})) == 1

    def test_order_is_stable_and_independent_of_input_order(self):
        one = _assessment("t_900001", [STORY_A, STORY_B])
        two = _assessment("t_900002", ["A third synthetic story about a campus visit."])
        forward = [c.story_key for c in build_queue([one, two], set(), {})]
        backward = [c.story_key for c in build_queue([two, one], set(), {})]
        assert forward == backward == sorted(forward)

    def test_a_new_cut_of_a_miscut_story_comes_back_linked_to_it(self):
        labels = latest_labels([_miscut(STORY_A_CUT_SHORT)])
        queue = build_queue([_assessment("t_900001", [STORY_A, STORY_B])], set(), labels)
        recut = next(c for c in queue if c.text == STORY_A)
        assert recut.recut_of == story_key(STORY_A_CUT_SHORT)
        other = next(c for c in queue if c.text == STORY_B)
        assert other.recut_of == ""

    def test_a_new_cut_overlapping_a_graded_story_is_not_queued_again(self):
        labels = latest_labels([_graded(STORY_A_CUT_SHORT)])
        queue = build_queue([_assessment("t_900001", [STORY_A, STORY_B])], set(), labels)
        assert [c.text for c in queue] == [STORY_B]

    def test_overlap_is_judged_within_the_thread_only(self):
        labels = latest_labels([_graded(STORY_A_CUT_SHORT, thread_id="t_900009")])
        queue = build_queue([_assessment("t_900001", [STORY_A])], set(), labels)
        assert [c.text for c in queue] == [STORY_A]

    def test_graded_overlap_outranks_miscut_overlap(self):
        labels = latest_labels([_miscut(STORY_A_CUT_SHORT), _graded(STORY_A)])
        longer = STORY_A + " She plans to keep hosting in the spring."
        queue = build_queue([_assessment("t_900001", [longer])], set(), labels)
        assert queue == []


class TestMiscutBucket:
    def test_lists_open_miscut_stories_with_reason_and_note(self):
        labels = latest_labels([
            _miscut(STORY_A_CUT_SHORT, note="ending is in the next paragraph"),
            _graded(STORY_B),
        ])
        bucket = miscut_bucket(labels)
        assert [(m.text, m.miscut_reason, m.note) for m in bucket] == [
            (STORY_A_CUT_SHORT, "cut_short", "ending is in the next paragraph"),
        ]

    def test_a_graded_recut_closes_the_miscut_it_replaces(self):
        labels = latest_labels([
            _miscut(STORY_A_CUT_SHORT),
            _graded(STORY_A, recut_of=story_key(STORY_A_CUT_SHORT)),
        ])
        assert miscut_bucket(labels) == []

    def test_a_recut_that_is_itself_miscut_keeps_the_original_open(self):
        labels = latest_labels([
            _miscut(STORY_A_CUT_SHORT),
            _miscut(STORY_A, reason="runs_long", recut_of=story_key(STORY_A_CUT_SHORT)),
        ])
        assert {m.text for m in miscut_bucket(labels)} == {STORY_A_CUT_SHORT, STORY_A}

    def test_a_story_regraded_after_being_marked_miscut_leaves_the_bucket(self):
        labels = latest_labels([
            _miscut(STORY_A, labeled_at="2026-11-01T10:00:00+00:00"),
            _graded(STORY_A, labeled_at="2026-11-02T10:00:00+00:00"),
        ])
        assert miscut_bucket(labels) == []

    def test_bucket_is_grouped_by_thread(self):
        labels = latest_labels([
            _miscut(STORY_B, thread_id="t_900002"),
            _miscut(STORY_A, thread_id="t_900001"),
        ])
        assert [m.thread_id for m in miscut_bucket(labels)] == ["t_900001", "t_900002"]


def _rubric_table() -> dict:
    return {
        "grades": {"poor": "p", "ok": "o", "good": "g"},
        "dimensions": {"simple": "s?", "concrete": "c?", "personal": "p?", "dynamic": "d?"},
        "theme_grades": {"absent": "a", "present": "p", "emphasized": "e"},
        "themes": {
            "scripture": "S", "christlikeness": "C", "church": "H",
            "vocation_family": "V", "disciple_making": "D",
        },
    }


class TestLoadRubric:
    def test_reads_the_four_tables(self):
        rubric = load_rubric({"newsletter": {"rubric": _rubric_table()}})
        assert isinstance(rubric, Rubric)
        assert rubric.grades["good"] == "g"
        assert rubric.dimensions["dynamic"] == "d?"
        assert rubric.theme_grades["emphasized"] == "e"
        assert rubric.themes["church"] == "H"

    def test_missing_table_says_where_it_belongs(self):
        with pytest.raises(ValueError, match=r"\[newsletter\.rubric\]"):
            load_rubric({"newsletter": {}})

    @pytest.mark.parametrize("table, key", [
        ("grades", "ok"), ("dimensions", "personal"),
        ("theme_grades", "present"), ("themes", "vocation_family"),
    ])
    def test_a_missing_entry_is_named(self, table, key):
        data = _rubric_table()
        del data[table][key]
        with pytest.raises(ValueError, match=f"{table}.*{key}"):
            load_rubric({"newsletter": {"rubric": data}})

    def test_an_unknown_entry_is_named(self):
        data = _rubric_table()
        data["themes"]["evangelism"] = "E"
        with pytest.raises(ValueError, match="themes.*evangelism"):
            load_rubric({"newsletter": {"rubric": data}})


class TestShippedRubricMatchesThePrompts:
    """The card shows the operator the same words the model is given.

    Each rubric string in the shipped config.toml must appear verbatim in the
    grading prompt it belongs to, so the labeler and the prompt cannot drift
    apart without this failing.
    """

    @pytest.fixture(scope="class")
    def config(self):
        return load_config()

    @pytest.fixture(scope="class")
    def rubric(self, config):
        return load_rubric(config)

    @pytest.mark.parametrize("grade", ["poor", "ok", "good"])
    def test_grade_anchor_is_in_the_quality_prompt(self, config, rubric, grade):
        prompt = config["newsletter"]["prompts"]["quality_assessment"]["system"]
        assert f"{grade.upper()}: {rubric.grades[grade]}" in prompt

    @pytest.mark.parametrize("dimension", ["simple", "concrete", "personal", "dynamic"])
    def test_dimension_question_is_in_the_quality_prompt(self, config, rubric, dimension):
        prompt = config["newsletter"]["prompts"]["quality_assessment"]["system"]
        assert f"{dimension.upper()}: {rubric.dimensions[dimension]}" in prompt

    @pytest.mark.parametrize("grade", ["absent", "present", "emphasized"])
    def test_theme_grade_anchor_is_in_the_theme_prompt(self, config, rubric, grade):
        prompt = config["newsletter"]["prompts"]["theme_classification"]["system"]
        assert f"{grade.upper()}: {rubric.theme_grades[grade]}" in prompt

    @pytest.mark.parametrize("theme", [
        "scripture", "christlikeness", "church", "vocation_family", "disciple_making",
    ])
    def test_theme_definition_is_in_the_theme_prompt(self, config, rubric, theme):
        prompt = config["newsletter"]["prompts"]["theme_classification"]["system"]
        assert f"{theme.upper()}: {rubric.themes[theme]}" in prompt


def _card(text=STORY_A, thread_id="t_900001", recut_of=""):
    return Card(Candidate(story_key=story_key(text), thread_id=thread_id, text=text,
                          source_timestamp="2026-10-30T12:00:00+00:00", recut_of=recut_of))


def _answered(card, *values):
    for value in values:
        card = answer(card, value)
    return card


class TestCard:
    def test_questions_are_the_four_dimensions_then_the_five_themes(self):
        assert QUESTIONS == (
            "simple", "concrete", "personal", "dynamic",
            "scripture", "christlikeness", "church", "vocation_family", "disciple_making",
        )

    def test_a_new_card_asks_the_first_dimension(self):
        assert current_question(_card()) == "simple"

    def test_each_answer_moves_to_the_next_question(self):
        card = _answered(_card(), 3, 2)
        assert card.answers == (3, 2)
        assert current_question(card) == "personal"

    def test_after_nine_answers_nothing_is_left_to_ask(self):
        card = _answered(_card(), 3, 2, 2, 1, 1, 1, 1, 1, 1)
        assert current_question(card) is None
        assert answer(card, 3) == card

    @pytest.mark.parametrize("value", [0, 4, -1])
    def test_only_one_two_three_are_answers(self, value):
        assert answer(_card(), value) == _card()

    def test_back_removes_the_last_answer(self):
        card = back(_answered(_card(), 3, 2))
        assert card.answers == (3,)
        assert current_question(card) == "concrete"

    def test_back_on_an_untouched_card_changes_nothing(self):
        assert back(_card()) == _card()

    def test_rest_absent_fills_the_remaining_themes(self):
        card = answer_rest_absent(_answered(_card(), 3, 2, 2, 1, 2))
        assert card.answers == (3, 2, 2, 1, 2, 1, 1, 1, 1)

    def test_rest_absent_does_nothing_while_a_dimension_is_being_asked(self):
        card = _answered(_card(), 3, 2)
        assert answer_rest_absent(card) == card

    def test_torn_toggles_on_the_question_being_asked(self):
        card = toggle_torn(_answered(_card(), 3))
        assert card.torn == frozenset({"concrete"})
        assert toggle_torn(card).torn == frozenset()

    def test_torn_survives_answering_and_stepping_back(self):
        card = answer(toggle_torn(_answered(_card(), 3)), 2)
        assert card.torn == frozenset({"concrete"})
        assert back(card).torn == frozenset({"concrete"})

    def test_torn_on_a_finished_card_changes_nothing(self):
        card = _answered(_card(), 3, 2, 2, 1, 1, 1, 1, 1, 1)
        assert toggle_torn(card) == card


class TestLabelsFromCards:
    def test_graded_label_carries_scores_themes_torn_and_note(self):
        card = toggle_torn(_answered(_card(recut_of="feedfacefeedface"), 3))
        card = _answered(card, 2, 2, 1, 1, 3, 2, 1, 1)
        card = Card(card.candidate, card.answers, card.torn, note="one named moment only")
        label = graded_label(card, labeled_at="2026-11-05T21:00:00+00:00", seconds=61.0)
        assert label.status == "graded"
        assert label.story_key == story_key(STORY_A)
        assert label.thread_id == "t_900001"
        assert label.text == STORY_A
        assert label.scores == {"simple": 3, "concrete": 2, "personal": 2, "dynamic": 1}
        assert label.themes == {"christlikeness": "emphasized", "church": "present"}
        assert label.torn == ["concrete"]
        assert label.note == "one named moment only"
        assert label.recut_of == "feedfacefeedface"
        assert label.labeled_at == "2026-11-05T21:00:00+00:00"
        assert label.seconds == 61.0
        assert label.source_timestamp == "2026-10-30T12:00:00+00:00"
        # What is written is what loads back.
        assert StoryLabel.from_dict(json.loads(json.dumps(label.to_dict()))) == label

    def test_an_unfinished_card_cannot_become_a_label(self):
        with pytest.raises(ValueError, match="unanswered"):
            graded_label(_answered(_card(), 3, 2), labeled_at="x", seconds=1.0)

    def test_miscut_label_keeps_the_text_and_carries_no_grades(self):
        card = _answered(_card(), 3, 2)
        label = miscut_label(card, "cut_short", "ending is in the next paragraph",
                             labeled_at="2026-11-05T21:00:00+00:00", seconds=12.5)
        assert label.status == "miscut"
        assert label.miscut_reason == "cut_short"
        assert label.note == "ending is in the next paragraph"
        assert label.text == STORY_A
        assert label.scores is None
        assert label.themes == {}
        assert StoryLabel.from_dict(json.loads(json.dumps(label.to_dict()))) == label

    def test_a_graded_label_reopens_at_its_last_question(self):
        card = toggle_torn(_answered(_card(), 3))
        card = _answered(card, 2, 2, 1, 1, 3, 2, 1, 1)
        label = graded_label(Card(card.candidate, card.answers, card.torn, note="why"),
                             labeled_at="x", seconds=1.0)
        reopened = card_from_label(label)
        assert reopened.answers == (3, 2, 2, 1, 1, 3, 2, 1)
        assert current_question(reopened) == "disciple_making"
        assert reopened.torn == frozenset({"concrete"})
        assert reopened.note == "why"
        assert reopened.candidate == card.candidate

    def test_a_miscut_label_reopens_as_a_fresh_card(self):
        label = miscut_label(_card(), "runs_long", "", labeled_at="x", seconds=1.0)
        reopened = card_from_label(label)
        assert reopened.answers == ()
        assert reopened.candidate.text == STORY_A


RUBRIC = Rubric(
    grades={"poor": "ANCHOR-POOR", "ok": "ANCHOR-OK", "good": "ANCHOR-GOOD"},
    dimensions={"simple": "Q-SIMPLE?", "concrete": "Q-CONCRETE?",
                "personal": "Q-PERSONAL?", "dynamic": "Q-DYNAMIC?"},
    theme_grades={"absent": "ANCHOR-ABSENT", "present": "ANCHOR-PRESENT",
                  "emphasized": "ANCHOR-EMPHASIZED"},
    themes={"scripture": "DEF-SCRIPTURE", "christlikeness": "DEF-CHRISTLIKENESS",
            "church": "DEF-CHURCH", "vocation_family": "DEF-VOCATION",
            "disciple_making": "DEF-DISCIPLE"},
)


class TestWhatTheOperatorSees:
    def test_a_dimension_shows_its_question_and_the_three_grade_anchors(self):
        text = "\n".join(question_panel(_answered(_card(), 3), RUBRIC))
        assert "CONCRETE" in text and "Q-CONCRETE?" in text
        for expected in ("1  Poor", "ANCHOR-POOR", "2  OK", "ANCHOR-OK", "3  Good", "ANCHOR-GOOD"):
            assert expected in text
        assert "Q-SIMPLE?" not in text and "DEF-CHURCH" not in text

    def test_a_theme_shows_its_definition_and_the_three_theme_anchors(self):
        text = "\n".join(question_panel(_answered(_card(), 3, 2, 2, 1, 1, 1), RUBRIC))
        assert "CHURCH" in text and "DEF-CHURCH" in text
        for expected in ("1  Absent", "ANCHOR-ABSENT", "2  Present", "ANCHOR-PRESENT",
                         "3  Emphasized", "ANCHOR-EMPHASIZED"):
            assert expected in text
        assert "ANCHOR-GOOD" not in text

    def test_a_torn_question_shows_the_grade_down_rule(self):
        plain = "\n".join(question_panel(_card(), RUBRIC))
        torn = "\n".join(question_panel(toggle_torn(_card()), RUBRIC))
        assert "grade down" not in plain
        assert "grade down and say why" in torn

    def test_the_strip_names_answers_given_and_marks_the_current_question(self):
        strip = answers_strip(toggle_torn(_answered(_card(), 3, 1)))
        assert "simple Good" in strip
        assert "concrete Poor" in strip
        assert "personal ?" in strip
        assert "torn" in strip

    def test_the_strip_names_theme_grades(self):
        strip = answers_strip(_answered(_card(), 3, 2, 2, 1, 1, 3, 2))
        assert "scripture Absent" in strip
        assert "christlikeness Emphasized" in strip
        assert "church Present" in strip
        assert "vocation_family ?" in strip


class TestCountLabeledOn:
    def test_counts_the_stories_labeled_that_day(self):
        labels = latest_labels([
            _graded(STORY_A, labeled_at="2026-11-05T15:00:00+00:00"),
            _miscut(STORY_B, labeled_at="2026-11-05T16:00:00+00:00"),
            _graded("A third synthetic story.", labeled_at="2026-11-04T16:00:00+00:00"),
            _graded("A fourth synthetic story.", labeled_at=""),
        ])
        assert count_labeled_on(labels, date(2026, 11, 5), tz=timezone.utc) == 2


SIZE = (100, 30)
LONG_STORY = "\n\n".join(
    f"Paragraph {n} of a long synthetic story about a semester on campus." for n in range(1, 60)
)


def _candidates(*texts):
    return [
        Candidate(story_key=story_key(text), thread_id="t_900001", text=text,
                  source_timestamp="2026-10-30T12:00:00+00:00")
        for text in texts
    ]


class _Clock:
    """A clock the test moves by hand, so ``seconds`` is exact."""

    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value


def _app(tmp_path, *texts, clock=None, **kwargs):
    return GradeApp(
        _candidates(*texts), RUBRIC, tmp_path / "labels.jsonl",
        clock=clock or _Clock(),
        now=lambda: datetime(2026, 11, 5, 21, 0, tzinfo=timezone.utc),
        **kwargs,
    )


def _text(app, selector):
    return str(app.query_one(selector, Static).render())


def _screen_text(app):
    return "\n".join(str(widget.render()) for widget in app.query(Static))


def _written(tmp_path):
    return read_label_log(tmp_path / "labels.jsonl")


class TestGradeScreen:
    async def test_opens_on_the_first_story_with_its_first_question(self, tmp_path):
        app = _app(tmp_path, STORY_A, STORY_B)
        async with app.run_test(size=SIZE):
            assert STORY_A in _text(app, "#story-text")
            assert STORY_B not in _screen_text(app)
            question = _text(app, "#question")
            assert "SIMPLE" in question and "Q-SIMPLE?" in question
            assert "ANCHOR-GOOD" in question

    async def test_nine_answers_write_one_label_and_show_the_next_story(self, tmp_path):
        clock = _Clock()
        app = _app(tmp_path, STORY_A, STORY_B, clock=clock)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("3", "2", "2", "1", "1", "3", "1", "1")
            assert _written(tmp_path) == []
            clock.value = 174.5
            await pilot.press("1")
            (label,) = _written(tmp_path)
            assert label.status == "graded"
            assert label.text == STORY_A
            assert label.scores == {"simple": 3, "concrete": 2, "personal": 2, "dynamic": 1}
            assert label.themes == {"christlikeness": "emphasized"}
            assert label.labeled_at == "2026-11-05T21:00:00+00:00"
            assert label.seconds == 74.5
            assert STORY_B in _text(app, "#story-text")
            assert "SIMPLE" in _text(app, "#question")

    async def test_the_question_panel_follows_the_answers(self, tmp_path):
        app = _app(tmp_path, STORY_A)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("3")
            assert "Q-CONCRETE?" in _text(app, "#question")
            await pilot.press("2", "2", "1")
            question = _text(app, "#question")
            assert "DEF-SCRIPTURE" in question and "ANCHOR-EMPHASIZED" in question
            assert "simple Good" in _text(app, "#strip")

    async def test_zero_answers_absent_for_the_remaining_themes(self, tmp_path):
        app = _app(tmp_path, STORY_A)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("0")  # a dimension is being asked: nothing happens
            assert "SIMPLE" in _text(app, "#question")
            await pilot.press("3", "3", "3", "3", "2", "0")
            (label,) = _written(tmp_path)
            assert label.themes == {"scripture": "present"}

    async def test_the_story_scrolls_while_it_is_being_graded(self, tmp_path):
        # Issue #54: scrolling must neither be blocked nor cancel the answers.
        app = _app(tmp_path, LONG_STORY)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("3", "2")
            story = app.query_one("#story")
            assert story.scroll_y == 0
            await pilot.press("pagedown")
            await pilot.pause()
            assert story.scroll_y > 0
            assert "PERSONAL" in _text(app, "#question")
            await pilot.press("2")
            assert "DYNAMIC" in _text(app, "#question")

    async def test_a_new_story_starts_scrolled_to_its_top(self, tmp_path):
        app = _app(tmp_path, LONG_STORY, STORY_B)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("pagedown")
            await pilot.press("3", "3", "3", "3", "0")
            await pilot.pause()
            assert app.query_one("#story").scroll_y == 0

    async def test_a_torn_card_asks_why_before_it_is_written(self, tmp_path):
        app = _app(tmp_path, STORY_A)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("3", "t")
            assert "grade down and say why" in _text(app, "#question")
            await pilot.press("2", "2", "1", "0")
            assert _written(tmp_path) == []
            note = app.query_one("#note", Input)
            assert note.display and note.has_focus
            note.value = "one named moment, the rest is summary"
            await pilot.press("enter")
            (label,) = _written(tmp_path)
            assert label.torn == ["concrete"]
            assert label.note == "one named moment, the rest is summary"

    async def test_escape_from_the_why_returns_to_the_last_question(self, tmp_path):
        app = _app(tmp_path, STORY_A)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("t", "3", "2", "2", "1", "0")
            await pilot.press("escape")
            assert _written(tmp_path) == []
            assert not app.query_one("#note", Input).display
            assert "DISCIPLE_MAKING" in _text(app, "#question")

    async def test_backspace_takes_back_the_last_answer(self, tmp_path):
        app = _app(tmp_path, STORY_A)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("3", "2", "backspace")
            assert "CONCRETE" in _text(app, "#question")
            await pilot.press("backspace", "backspace")  # second one has nothing to undo
            assert "SIMPLE" in _text(app, "#question")
            assert STORY_A in _text(app, "#story-text")

    async def test_backspace_at_a_story_start_reopens_the_previous_story(self, tmp_path):
        app = _app(tmp_path, STORY_A, STORY_B)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("3", "3", "3", "3", "1", "1", "1", "1", "3")
            assert STORY_B in _text(app, "#story-text")
            await pilot.press("backspace")
            assert STORY_A in _text(app, "#story-text")
            assert "DISCIPLE_MAKING" in _text(app, "#question")
            await pilot.press("1")
            assert STORY_B in _text(app, "#story-text")
        lines = _written(tmp_path)
        assert [line.text for line in lines] == [STORY_A, STORY_A]
        assert latest_labels(lines)[story_key(STORY_A)].themes == {}

    async def test_miscut_sets_the_story_aside_with_a_reason_and_note(self, tmp_path):
        app = _app(tmp_path, STORY_A, STORY_B)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("3", "x")
            reasons = _text(app, "#question")
            for expected in ("cut short", "runs past", "several stories", "not a story"):
                assert expected in reasons
            await pilot.press("1")
            note = app.query_one("#note", Input)
            assert note.has_focus
            note.value = "the ending is in the next paragraph"
            await pilot.press("enter")
            (label,) = _written(tmp_path)
            assert label.status == "miscut"
            assert label.miscut_reason == "cut_short"
            assert label.note == "the ending is in the next paragraph"
            assert label.scores is None
            assert STORY_B in _text(app, "#story-text")

    async def test_miscut_needs_no_note(self, tmp_path):
        app = _app(tmp_path, STORY_A)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("x", "4", "enter")
            (label,) = _written(tmp_path)
            assert (label.miscut_reason, label.note) == ("not_a_story", "")

    async def test_escape_cancels_miscut_and_keeps_the_answers(self, tmp_path):
        app = _app(tmp_path, STORY_A)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("3", "2", "x", "escape")
            assert "PERSONAL" in _text(app, "#question")
            await pilot.press("x", "2", "escape")  # cancel at the note step too
            assert "PERSONAL" in _text(app, "#question")
            assert _written(tmp_path) == []

    async def test_skip_moves_the_story_to_the_end_of_the_queue(self, tmp_path):
        app = _app(tmp_path, STORY_A, STORY_B)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("3", "s")
            assert STORY_B in _text(app, "#story-text")
            assert "SIMPLE" in _text(app, "#question")
            await pilot.press("s")
            assert STORY_A in _text(app, "#story-text")
            assert "SIMPLE" in _text(app, "#question")  # the earlier answer is not kept
            assert _written(tmp_path) == []

    async def test_a_note_can_be_added_at_any_point_and_typing_does_not_answer(self, tmp_path):
        app = _app(tmp_path, STORY_A)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("3", "n")
            await pilot.press("1", "2", "space", "x", "t")
            assert app.query_one("#note", Input).value == "12 xt"
            await pilot.press("enter")
            assert "CONCRETE" in _text(app, "#question")
            await pilot.press("2", "2", "1", "0")
            (label,) = _written(tmp_path)
            assert label.note == "12 xt"
            assert label.torn == []

    async def test_progress_counts_today_and_what_is_left(self, tmp_path):
        app = _app(tmp_path, STORY_A, STORY_B, labeled_today=4)
        async with app.run_test(size=SIZE) as pilot:
            assert "4 labeled today" in _text(app, "#progress")
            assert "2 queued" in _text(app, "#progress")
            await pilot.press("3", "3", "3", "3", "0")
            assert "5 labeled today" in _text(app, "#progress")
            assert "1 queued" in _text(app, "#progress")

    async def test_an_empty_queue_says_so_and_ignores_grading_keys(self, tmp_path):
        app = _app(tmp_path, STORY_A)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("3", "3", "3", "3", "0")
            assert "Nothing left to grade" in _screen_text(app)
            await pilot.press("3", "t", "x", "s", "0", "n")
            assert len(_written(tmp_path)) == 1

    async def test_backspace_on_the_empty_queue_reopens_the_last_story(self, tmp_path):
        app = _app(tmp_path, STORY_A)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("3", "3", "3", "3", "0", "backspace")
            assert STORY_A in _text(app, "#story-text")
            assert "DISCIPLE_MAKING" in _text(app, "#question")

    async def test_quitting_mid_story_writes_nothing(self, tmp_path):
        app = _app(tmp_path, STORY_A, STORY_B)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("3", "3", "3", "3", "0")  # one story labeled
            await pilot.press("3", "2", "q")            # second left half-answered
        assert app.return_value == 1  # stories labeled this session
        assert [label.text for label in _written(tmp_path)] == [STORY_A]

    async def test_a_label_that_cannot_be_saved_keeps_the_story_on_screen(self, tmp_path):
        # Nothing is dropped silently: the card stays, the queue does not
        # advance, and the operator is told.
        app = GradeApp(_candidates(STORY_A, STORY_B), RUBRIC,
                       tmp_path / "no-such-directory" / "labels.jsonl")
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("3", "3", "3", "3", "0")
            assert STORY_A in _text(app, "#story-text")
            assert "Could not save" in _screen_text(app)
            assert "2 queued" in _text(app, "#progress")
            assert "DISCIPLE_MAKING" in _text(app, "#question")

    async def test_control_characters_in_a_story_are_not_drawn_but_are_kept_in_the_label(
        self, tmp_path,
    ):
        hostile = "First line\x1b[2J of a synthetic story.\r\nSecond\x07 line."
        app = _app(tmp_path, hostile)
        async with app.run_test(size=SIZE) as pilot:
            shown = _text(app, "#story-text")
            for control in ("\x1b", "\x07", "\r"):
                assert control not in shown
            assert "First line" in shown and "Second" in shown and "line." in shown
            await pilot.press("3", "3", "3", "3", "0")
        (label,) = _written(tmp_path)
        assert label.text == hostile  # the eval grades the text as extracted

    async def test_nothing_the_model_said_reaches_the_screen(self, tmp_path):
        queue = build_queue([_assessment("t_900001", [STORY_A])], set(), {})
        app = GradeApp(queue, RUBRIC, tmp_path / "labels.jsonl")
        async with app.run_test(size=SIZE) as pilot:
            assert STORY_A in _screen_text(app)
            for key in ("3", "3", "3", "3", "1"):
                assert "SENTINEL" not in _screen_text(app)
                assert "excellent" not in _screen_text(app).lower()
                await pilot.press(key)


def _write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


class TestCli:
    def _paths(self, tmp_path):
        return [
            "--assessments", str(tmp_path / "assessments.jsonl"),
            "--labels", str(tmp_path / "labels.jsonl"),
            "--golden-set", str(tmp_path / "golden.jsonl"),
        ]

    def test_a_missing_assessments_file_is_named_and_exits_1(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exit_info:
            cli(self._paths(tmp_path))
        assert exit_info.value.code == 1
        assert "assessments.jsonl" in capsys.readouterr().err

    def test_nothing_to_grade_exits_0_without_opening_the_screen(self, tmp_path, capsys, monkeypatch):
        _write_jsonl(tmp_path / "assessments.jsonl", [_assessment("t_900001", [STORY_A])])
        append_label(tmp_path / "labels.jsonl", _graded(STORY_A))
        monkeypatch.setattr(GradeApp, "run", lambda self: pytest.fail("screen opened"))
        with pytest.raises(SystemExit) as exit_info:
            cli(self._paths(tmp_path))
        assert exit_info.value.code == 0
        assert "Nothing to grade" in capsys.readouterr().out

    def test_the_screen_gets_the_queue_minus_curated_threads_and_labeled_stories(
        self, tmp_path, monkeypatch,
    ):
        _write_jsonl(tmp_path / "assessments.jsonl", [
            _assessment("t_900001", [STORY_A, STORY_B]),
            _assessment("t_900002", ["A story from a curated newsletter."]),
        ])
        _write_jsonl(tmp_path / "golden.jsonl", [{
            "type": "golden_newsletter", "thread_id": "t_900002", "message_id": "m",
            "sender": "s", "subject": "s", "body": "b", "stories": [],
        }])
        append_label(tmp_path / "labels.jsonl", _graded(STORY_B))
        opened = {}
        monkeypatch.setattr(GradeApp, "run", lambda self: opened.update(app=self))
        cli(self._paths(tmp_path))
        assert [candidate.text for candidate in opened["app"].queue] == [STORY_A]
        assert opened["app"].label_path == tmp_path / "labels.jsonl"

    def test_a_missing_curated_golden_set_is_not_an_error(self, tmp_path, monkeypatch):
        _write_jsonl(tmp_path / "assessments.jsonl", [_assessment("t_900001", [STORY_A])])
        opened = {}
        monkeypatch.setattr(GradeApp, "run", lambda self: opened.update(app=self))
        cli(self._paths(tmp_path))
        assert len(opened["app"].queue) == 1

    def test_miscut_lists_the_bucket_and_does_not_open_the_screen(
        self, tmp_path, capsys, monkeypatch,
    ):
        _write_jsonl(tmp_path / "assessments.jsonl", [_assessment("t_900001", [STORY_A])])
        append_label(tmp_path / "labels.jsonl",
                     _miscut(STORY_A_CUT_SHORT, note="ending is in the next paragraph"))
        monkeypatch.setattr(GradeApp, "run", lambda self: pytest.fail("screen opened"))
        cli(self._paths(tmp_path) + ["--miscut"])
        out = capsys.readouterr().out
        assert "1 miscut" in out
        assert "t_900001" in out
        assert "cut short" in out
        assert "ending is in the next paragraph" in out
        assert "Dana had never led" in out

    def test_an_empty_miscut_bucket_says_so(self, tmp_path, capsys):
        _write_jsonl(tmp_path / "assessments.jsonl", [])
        cli(self._paths(tmp_path) + ["--miscut"])
        assert "No miscut stories" in capsys.readouterr().out


class TestFormatMiscutBucket:
    def test_control_characters_never_reach_the_terminal(self):
        # Story text comes from email and notes may be pasted. Printed raw, an
        # escape sequence in either could repaint or retitle the terminal.
        hostile = "A synthetic story\x1b[2J\x1b]0;owned\x07 with an escape in it."
        label = _miscut(hostile, thread_id="t_900001\x1b[31m",
                        note="pasted \x1b[5mnote\x07")
        out = format_miscut_bucket([label])
        assert "\x1b" not in out
        assert "\x07" not in out
        assert "A synthetic story" in out and "with an escape in it." in out
        assert "pasted" in out and "note" in out
        assert "t_900001" in out
