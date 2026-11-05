"""E2E driver for evals.newsletter_grade (GradeApp) — grade ready-made stories, no model.

Builds the queue from the synthetic production assessments exactly as the CLI
does (``build_queue`` over ``load_assessments``-shaped records), then drives the
whole session the way an operator would: nine single-key answers per story, the
rest-absent shortcut, a torn question with its why, a note, a miscut with a
reason, skip, stepping back into the previous story, scrolling mid-answer, a
label that cannot be saved, and the empty queue. Outcomes are asserted on the
append-only label log read back through ``read_label_log``.
"""

import asyncio
import os
import sys
import tempfile

import synth_data
from _e2e import SIZE, report, run_scenarios, screen_text, static_text
from textual.widgets import Input

from evals.newsletter_grade import (
    Candidate,
    GradeApp,
    Rubric,
    build_queue,
    latest_labels,
    miscut_bucket,
    read_label_log,
    story_key,
)

RUBRIC = Rubric(
    grades={"poor": "anchor-poor", "ok": "anchor-ok", "good": "anchor-good"},
    dimensions={d: f"question-{d}?" for d in ("simple", "concrete", "personal", "dynamic")},
    theme_grades={"absent": "anchor-absent", "present": "anchor-present",
                  "emphasized": "anchor-emphasized"},
    themes={t: f"definition-{t}" for t in (
        "scripture", "christlikeness", "church", "vocation_family", "disciple_making")},
)

ALL_GOOD_NO_THEMES = ("3", "3", "3", "3", "0")


def _queue():
    """The queue as the CLI builds it: no curated threads, nothing labeled yet."""
    return build_queue(synth_data.assessment_records(), set(), {})


def _app(directory, queue=None):
    return GradeApp(queue if queue is not None else _queue(), RUBRIC,
                    os.path.join(directory, "labels.jsonl"))


def _log(directory):
    return read_label_log(os.path.join(directory, "labels.jsonl"))


async def s_whole_queue_graded(chk):
    """Every synthetic story is opened, its text shown verbatim, graded and logged."""
    queue = _queue()
    chk.that(len(queue) >= 3, f"synthetic data should queue several stories, got {len(queue)}")
    with tempfile.TemporaryDirectory() as directory:
        app = _app(directory, queue)
        async with app.run_test(size=SIZE) as pilot:
            for candidate in queue:
                shown = static_text(app, "#story-text")
                chk.eq(shown, candidate.text, "story text shown verbatim")
                chk.that("question-simple?" in static_text(app, "#question"),
                         "each story starts at the first dimension")
                await pilot.press(*ALL_GOOD_NO_THEMES)
            chk.that("Nothing left to grade" in screen_text(app), "queue drained")
            await pilot.press("q")
        labels = _log(directory)
        chk.eq(len(labels), len(queue), "one label line per story")
        chk.eq([label.story_key for label in labels], [c.story_key for c in queue],
               "labels in queue order")
        chk.that(all(label.status == "graded" and label.themes == {} for label in labels),
                 "all graded, no themes")
        chk.eq(app.return_value, len(queue), "session count returned on quit")


async def s_nothing_the_model_said_is_shown(chk):
    """Grades, tiers and reasoning in the assessment records never reach the screen."""
    said = set()
    for record in synth_data.assessment_records():
        for story in record.get("stories", []):
            said.update(filter(None, (story.get("quality_cot"), story.get("theme_cot"))))
    with tempfile.TemporaryDirectory() as directory:
        app = _app(directory)
        async with app.run_test(size=SIZE) as pilot:
            while app.queue:
                text = screen_text(app)
                for reasoning in said:
                    chk.that(reasoning not in text, "model reasoning leaked onto the screen")
                await pilot.press(*ALL_GOOD_NO_THEMES)


async def s_full_answers_torn_and_note(chk):
    """Nine explicit answers, a torn flag, the why prompt, and the stored label."""
    queue = _queue()
    with tempfile.TemporaryDirectory() as directory:
        app = _app(directory, queue)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("3", "t")
            chk.that("grade down and say why" in static_text(app, "#question"),
                     "torn shows the grade-down rule")
            await pilot.press("2", "1", "2", "1", "3", "2", "1", "1")
            chk.eq(_log(directory), [], "a torn card is not written before its why")
            note = app.query_one("#note", Input)
            chk.that(note.has_focus, "why prompt focused")
            note.value = "one particular moment, the rest is summary"
            await pilot.press("enter")
        (label,) = _log(directory)
        chk.eq(label.scores, {"simple": 3, "concrete": 2, "personal": 1, "dynamic": 2}, "scores")
        chk.eq(label.themes, {"christlikeness": "emphasized", "church": "present"}, "themes")
        chk.eq(label.torn, ["concrete"], "torn question recorded")
        chk.eq(label.note, "one particular moment, the rest is summary", "why recorded")
        chk.eq(label.text, queue[0].text, "label carries the story text")


async def s_miscut_skip_and_bucket(chk):
    """A miscut story is set aside with its reason; a skipped one comes back later."""
    queue = _queue()
    with tempfile.TemporaryDirectory() as directory:
        app = _app(directory, queue)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("3", "x")
            chk.that("several stories in one" in static_text(app, "#question"), "reason menu")
            await pilot.press("3")
            app.query_one("#note", Input).value = "two students, two arcs"
            await pilot.press("enter")
            chk.eq(static_text(app, "#story-text"), queue[1].text, "advanced past the miscut")
            await pilot.press("s")
            chk.eq(static_text(app, "#story-text"), queue[2].text, "skip shows the next story")
            chk.eq(app.queue[-1].story_key, queue[1].story_key, "skipped story is last in line")
        labels = latest_labels(_log(directory))
        bucket = miscut_bucket(labels)
        chk.eq([(m.story_key, m.miscut_reason, m.note) for m in bucket],
               [(queue[0].story_key, "several_stories", "two students, two arcs")],
               "miscut bucket holds the story with its reason and note")
        chk.that(bucket[0].scores is None, "a miscut story carries no grades")


async def s_back_into_previous_story(chk):
    """Backspace at a story's start reopens the one before; redoing it supersedes."""
    queue = _queue()
    with tempfile.TemporaryDirectory() as directory:
        app = _app(directory, queue)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("3", "3", "3", "3", "1", "1", "1", "1", "3")
            await pilot.press("2", "backspace", "backspace")
            chk.eq(static_text(app, "#story-text"), queue[0].text, "previous story reopened")
            chk.that("DISCIPLE_MAKING" in static_text(app, "#question"),
                     "reopened at its last question")
            await pilot.press("1")
            chk.eq(static_text(app, "#story-text"), queue[1].text, "moved on again")
        lines = _log(directory)
        chk.eq(len(lines), 2, "the redo appended a second line")
        chk.eq(latest_labels(lines)[queue[0].story_key].themes, {}, "the later line is in force")


async def s_scroll_while_grading(chk):
    """Scrolling a long story neither blocks nor cancels the answers (issue #54)."""
    long_text = "\n\n".join(f"Paragraph {n} of a long synthetic story." for n in range(1, 80))
    queue = [Candidate(story_key(long_text), "t_900050", long_text)]
    with tempfile.TemporaryDirectory() as directory:
        app = _app(directory, queue)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("3", "2", "pagedown", "pagedown", "down")
            await pilot.pause()
            story = app.query_one("#story")
            chk.that(story.scroll_y > 0, f"story scrolled (offset {story.scroll_y})")
            chk.that("question-personal?" in static_text(app, "#question"),
                     "answers kept while scrolling")
            await pilot.press("n")
            await pilot.press("pageup")
            await pilot.pause()
            chk.that(app.query_one("#note", Input).has_focus, "note still focused while scrolling")
            await pilot.press("escape", "home")
            await pilot.pause()
            chk.eq(story.scroll_y, 0, "home returns to the top")


async def s_unsaveable_label_keeps_the_story(chk):
    """A label that cannot be written leaves the story on screen and the queue intact."""
    queue = _queue()
    with tempfile.TemporaryDirectory() as directory:
        app = GradeApp(queue, RUBRIC, os.path.join(directory, "missing", "labels.jsonl"))
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press(*ALL_GOOD_NO_THEMES)
            chk.that("Could not save" in screen_text(app), "operator is told")
            chk.eq(static_text(app, "#story-text"), queue[0].text, "story stays")
            chk.eq(len(app.queue), len(queue), "queue did not advance")
            chk.that(app.is_running, "app still running")


async def s_labeled_stories_do_not_come_back(chk):
    """A second launch over the same data queues only what is still unlabeled."""
    queue = _queue()
    with tempfile.TemporaryDirectory() as directory:
        app = _app(directory, queue)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press(*ALL_GOOD_NO_THEMES)
            await pilot.press("x", "4", "enter")
            await pilot.press("q")
        labels = latest_labels(_log(directory))
        again = build_queue(synth_data.assessment_records(), set(), labels)
        chk.eq([c.story_key for c in again], [c.story_key for c in queue[2:]],
               "graded and miscut stories are not queued again")


def scenarios():
    return [
        ("whole queue graded, text shown verbatim", s_whole_queue_graded),
        ("nothing the model said is shown", s_nothing_the_model_said_is_shown),
        ("nine answers, torn, why", s_full_answers_torn_and_note),
        ("miscut with reason, skip, bucket", s_miscut_skip_and_bucket),
        ("back into the previous story", s_back_into_previous_story),
        ("scroll while grading and while noting", s_scroll_while_grading),
        ("unsaveable label keeps the story", s_unsaveable_label_keeps_the_story),
        ("labeled stories do not come back", s_labeled_stories_do_not_come_back),
    ]


if __name__ == "__main__":
    sys.exit(report("newsletter_grade", asyncio.run(run_scenarios(scenarios()))))
