"""Tests for evals.newsletter_grade — the story-grading labeler's pure core.

All story text here is synthetic. The shapes are the caller's: assessment
records as ``newsletter.write_assessment`` writes them, label lines as the tool
appends them.
"""

import json
from datetime import date, timezone

import pytest

from daemon import load_config
from evals.newsletter_grade import (
    QUESTIONS,
    Candidate,
    Card,
    Rubric,
    answer,
    answer_rest_absent,
    answers_strip,
    append_label,
    back,
    build_queue,
    card_from_label,
    count_labeled_on,
    current_question,
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
        for leaked in ("SENTINEL", "excellent", "emphasized", "scores", "cot"):
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
