"""E2E driver for evals.review (ReviewApp) — normal + blind review, no model.

Drives confirm/sender/label/notes/skip/exclude/undo, both review modes, stage
restrictions, body scrolling, quit, and the last-thread double-key regression.
"""

import asyncio
import sys

import synth_data
from _e2e import SIZE, report, run_scenarios

from evals.review import ReviewApp


def _threads():
    return synth_data.threads()


def _menu(app) -> str:
    from textual.widgets import Static
    return str(app.query_one("#menu", Static).render())


def _content(app) -> str:
    from textual.widgets import Static
    return str(app.query_one("#thread-content", Static).render())


async def s_normal_confirm_advances(chk):
    ths = _threads()[:2]
    app = ReviewApp(ths, blind=False)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter")          # confirm t0
        await pilot.pause()
        chk.that(ths[0].reviewed, "confirm marked t0 reviewed")
        chk.eq(app.session.index, 1, "advanced to t1")


async def s_normal_edit_sender_then_label(chk):
    ths = _threads()[:1]
    ths[0].expected_sender_type = "service"
    ths[0].expected_label = "low_priority"
    app = ReviewApp(ths, blind=False)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("s")              # sender submenu
        await pilot.pause()
        await pilot.press("p")              # -> person, reviewed, advance (single -> exit)
        await pilot.pause()
        chk.eq(ths[0].expected_sender_type, "person", "sender edited to person")
        chk.that(ths[0].reviewed, "reviewed after sender edit")
        chk.eq(app.return_value, "done", "queue exhausted -> done")


async def s_normal_label_edit(chk):
    ths = _threads()[:2]
    app = ReviewApp(ths, blind=False)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("l")
        await pilot.pause()
        await pilot.press("r")              # needs_response
        await pilot.pause()
        chk.eq(ths[0].expected_label, "needs_response", "label edited")
        chk.that(ths[0].reviewed, "reviewed after label edit")


async def s_normal_notes(chk):
    ths = _threads()[:1]
    app = ReviewApp(ths, blind=False)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("n")
        await pilot.pause()
        from textual.widgets import Input
        app.screen.query_one(Input).value = "note from e2e"
        await pilot.press("enter")
        await pilot.pause()
        chk.eq(ths[0].notes, "note from e2e", "notes saved")
        chk.that(not ths[0].reviewed, "notes alone does not confirm")


async def s_normal_skip_and_exclude(chk):
    ths = _threads()[:3]
    app = ReviewApp(ths, blind=False)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("k")              # skip t0 (no judgment)
        await pilot.pause()
        chk.that(not ths[0].reviewed, "skip does not mark reviewed")
        chk.eq(app.session.index, 1, "skip advanced")
        await pilot.press("e")              # exclude t1
        await pilot.pause()
        chk.that(ths[1].excluded and ths[1].reviewed, "exclude sets excluded+reviewed")
        chk.eq(app.session.index, 2, "exclude advanced")


async def s_undo(chk):
    ths = _threads()[:2]
    orig = ths[0].reviewed
    app = ReviewApp(ths, blind=False)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter")          # confirm t0 -> advance to t1
        await pilot.pause()
        await pilot.press("z")              # undo -> back to t0, reviewed reverted
        await pilot.pause()
        chk.eq(app.session.index, 0, "undo stepped back to t0")
        chk.eq(ths[0].reviewed, orig, "undo reverted t0 reviewed flag")


async def s_blind_full_flow(chk):
    # needs_response opens a third step (issue #78): sender -> label -> assistant.
    ths = _threads()[:2]
    app = ReviewApp(ths, blind=True)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("p")              # sender step -> person
        await pilot.pause()
        chk.eq(ths[0].expected_sender_type, "person", "blind sender set")
        await pilot.press("r")              # label step -> needs_response, assistant step opens
        await pilot.pause()
        chk.eq(ths[0].expected_label, "needs_response", "blind label set")
        chk.that(ths[0].reviewed, "blind flow marks reviewed")
        chk.eq(app.session.index, 0, "needs_response holds at the assistant step")
        await pilot.press("y")              # assistant step -> yes, advance
        await pilot.pause()
        chk.eq(ths[0].expected_assistant, True, "assistant step recorded yes")
        chk.eq(app.session.index, 1, "advanced after the assistant step")


async def s_blind_fyi_skips_assistant(chk):
    # The assistant question is asked on needs_response and nowhere else, so a
    # thread labelled fyi advances straight away with the field left unset.
    ths = _threads()[:2]
    app = ReviewApp(ths, blind=True)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("p")              # sender step -> person
        await pilot.pause()
        await pilot.press("f")              # label step -> fyi
        await pilot.pause()
        chk.eq(ths[0].expected_label, "fyi", "blind label set to fyi")
        chk.eq(ths[0].expected_assistant, None, "fyi leaves the assistant field unset")
        chk.eq(app.session.index, 1, "fyi advances without an assistant step")


async def s_blind_undo_mid_flow(chk):
    ths = _threads()[:1]
    ths[0].expected_sender_type = "service"
    app = ReviewApp(ths, blind=True)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("p")              # sender -> person, now at label step
        await pilot.pause()
        await pilot.press("z")              # undo reverts the whole thread
        await pilot.pause()
        chk.eq(ths[0].expected_sender_type, "service", "blind undo reverted in-progress sender")


async def s_stage1_sender_only(chk):
    ths = _threads()[:1]
    app = ReviewApp(ths, blind=True, stage=1)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("p")              # stage 1: sender confirms + advances (no label step)
        await pilot.pause()
        chk.that(ths[0].reviewed, "stage 1 marks reviewed after sender")
        chk.eq(app.return_value, "done", "single-thread stage1 exits")


async def s_stage2_label_only(chk):
    ths = _threads()[:1]
    app = ReviewApp(ths, blind=True, stage=2)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("f")              # stage 2 blind starts at label step -> fyi
        await pilot.pause()
        chk.eq(ths[0].expected_label, "fyi", "stage 2 sets label")
        chk.that(ths[0].reviewed, "stage 2 marks reviewed")


async def s_scroll_body(chk):
    ths = [t for t in _threads() if t.thread_id == "th-longbody"]
    app = ReviewApp(ths, blind=False)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("down", "pagedown", "end", "home", "up")
        await pilot.pause()
        chk.that(app.is_running, "scrolling the long body does not crash")


async def s_last_thread_double_confirm_no_crash(chk):
    # Regression guard for the on_key session.done fix.
    from textual import events
    ths = _threads()[:1]
    app = ReviewApp(ths, blind=False)
    async with app.run_test(size=SIZE) as pilot:
        app.post_message(events.Key("enter", None))
        app.post_message(events.Key("enter", None))
        await pilot.pause()
        await pilot.pause()
    chk.eq(app.return_value, "done", "double-confirm on last thread exits cleanly")
    chk.that(ths[0].reviewed, "last thread reviewed")


async def s_quit(chk):
    ths = _threads()[:2]
    app = ReviewApp(ths, blind=False)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("q")
    chk.eq(app.return_value, "quit", "q quits")
    chk.that(not ths[0].reviewed, "quitting does not confirm")


async def s_undo_after_skip_and_exclude(chk):
    # Guards ReviewSession's one-snapshot-per-advance invariant across non-confirm
    # mutations: undo must unwind skip and exclude one step at a time.
    ths = _threads()[:3]
    app = ReviewApp(ths, blind=False)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("k")              # skip t0
        await pilot.pause()
        await pilot.press("e")              # exclude t1
        await pilot.pause()
        chk.that(ths[1].excluded, "t1 excluded")
        chk.eq(app.session.index, 2, "advanced to t2")
        await pilot.press("z")              # undo the exclude
        await pilot.pause()
        chk.eq(app.session.index, 1, "undo stepped back to t1")
        chk.that(not ths[1].excluded, "undo reverted t1 exclusion")
        await pilot.press("z")              # undo the skip
        await pilot.pause()
        chk.eq(app.session.index, 0, "second undo stepped back to t0")


async def s_empty_undo(chk):
    ths = _threads()[:1]
    app = ReviewApp(ths, blind=False)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("z")
        await pilot.pause()
        chk.eq(app.session.index, 0, "empty undo stays put")
        chk.that("Nothing to undo" in str(app.query_one("#status").render()),
                 "empty-undo status message shown")


async def s_submenu_cancel_and_notes_esc(chk):
    ths = _threads()[:1]
    app = ReviewApp(ths, blind=False)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("s")              # sender submenu
        await pilot.pause()
        await pilot.press("x")              # unmapped -> CANCEL
        await pilot.pause()
        chk.that(not ths[0].reviewed, "submenu cancel did not confirm")
        chk.eq(app.session.index, 0, "submenu cancel did not advance")
        await pilot.press("n")              # notes prompt
        await pilot.pause()
        await pilot.press("escape")         # cancel notes
        await pilot.pause()
        chk.eq(ths[0].notes, "", "notes Esc leaves notes unchanged")


async def s_edit_sender_to_service(chk):
    ths = _threads()[:1]
    ths[0].expected_sender_type = "person"
    app = ReviewApp(ths, blind=False)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("s")
        await pilot.pause()
        await pilot.press("s")              # -> service (the other submenu option)
        await pilot.pause()
        chk.eq(ths[0].expected_sender_type, "service", "sender submenu 's' selects service")


async def s_diverse_threads_render(chk):
    multimsg = [t for t in _threads() if t.thread_id == "th-multimsg"]
    app = ReviewApp(multimsg, blind=False)
    async with app.run_test(size=SIZE):
        c = _content(app)
        chk.that("[Message 1]" in c and "[Message 3]" in c, "multi-message body renders each message")
    notes = [t for t in _threads() if t.thread_id == "th-notes"]
    app = ReviewApp(notes, blind=False)
    async with app.run_test(size=SIZE):
        chk.that("pre-existing reviewer note" in _content(app), "notes line rendered")


def scenarios():
    return [
        ("normal_confirm_advances", s_normal_confirm_advances),
        ("normal_edit_sender", s_normal_edit_sender_then_label),
        ("normal_label_edit", s_normal_label_edit),
        ("normal_notes", s_normal_notes),
        ("normal_skip_and_exclude", s_normal_skip_and_exclude),
        ("undo", s_undo),
        ("blind_full_flow", s_blind_full_flow),
        ("blind_fyi_skips_assistant", s_blind_fyi_skips_assistant),
        ("blind_undo_mid_flow", s_blind_undo_mid_flow),
        ("stage1_sender_only", s_stage1_sender_only),
        ("stage2_label_only", s_stage2_label_only),
        ("scroll_body", s_scroll_body),
        ("last_thread_double_confirm_no_crash", s_last_thread_double_confirm_no_crash),
        ("quit", s_quit),
        ("undo_after_skip_and_exclude", s_undo_after_skip_and_exclude),
        ("empty_undo", s_empty_undo),
        ("submenu_cancel_and_notes_esc", s_submenu_cancel_and_notes_esc),
        ("edit_sender_to_service", s_edit_sender_to_service),
        ("diverse_threads_render", s_diverse_threads_render),
    ]


if __name__ == "__main__":
    results = asyncio.run(run_scenarios(scenarios()))
    sys.exit(report("review", results))
