"""E2E driver for evals.edit_tui (EditApp) — golden-thread editor, no model.

Drives list navigation/paging, drill-in/back, sender+label edits (incl. the
l/l collision and cancel-no-save), unexclude, detail scrolling, filtered-view
save-all, and quit. All edits are synchronous (no @work workers).
"""

import asyncio
import sys
import tempfile
from pathlib import Path

import synth_data
from _e2e import report, run_scenarios

from evals.edit_tui import DetailScreen, EditApp
from evals.review import load_golden_set

SIZE = (100, 30)


def _threads():
    return synth_data.threads()


def _app(threads, all_threads=None):
    path = Path(tempfile.mkdtemp()) / "golden.jsonl"
    return EditApp(threads, all_threads if all_threads is not None else threads, path), path


def _lv(app):
    from textual.widgets import ListView
    return app.query_one(ListView)


async def s_list_nav_and_drill(chk):
    ths = _threads()
    app, _ = _app(ths)
    async with app.run_test(size=SIZE) as pilot:
        chk.eq(len(_lv(app)), len(ths), "all threads listed")
        await pilot.press("down")
        chk.eq(_lv(app).index, 1, "down moves cursor")
        await pilot.press("end")
        chk.eq(_lv(app).index, len(ths) - 1, "end -> last row")
        await pilot.press("home")
        chk.eq(_lv(app).index, 0, "home -> first row")
        await pilot.press("enter")
        chk.that(isinstance(app.screen, DetailScreen), "enter opens detail")
        await pilot.press("escape")
        chk.that(not isinstance(app.screen, DetailScreen), "escape returns to list")


async def s_edit_sender(chk):
    ths = [t for t in _threads() if t.thread_id == "th-person-fyi"]
    app, path = _app(ths)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter", "s")     # detail -> sender menu
        await pilot.pause()
        await pilot.press("s")              # -> service
        await pilot.pause()
        chk.eq(ths[0].expected_sender_type, "service", "sender edited to service")
        chk.that(path.exists(), "edit auto-saved to disk")


async def s_edit_label_and_collision(chk):
    ths = [t for t in _threads() if t.thread_id == "th-person-fyi"]
    app, _ = _app(ths)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter", "l")
        await pilot.pause()
        await pilot.press("r")              # needs_response
        await pilot.pause()
        chk.eq(ths[0].expected_label, "needs_response", "label -> needs_response")
        # l/l collision: first l opens menu, second l selects low_priority.
        await pilot.press("l")
        await pilot.pause()
        await pilot.press("l")
        await pilot.pause()
        chk.eq(ths[0].expected_label, "low_priority", "l/l collision selects low_priority")


async def s_sender_cancel_no_save(chk):
    ths = [t for t in _threads() if t.thread_id == "th-person-fyi"]
    app, path = _app(ths)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter", "s")
        await pilot.pause()
        await pilot.press("z")              # unmapped key -> CANCEL -> no edit
        await pilot.pause()
        chk.eq(ths[0].expected_sender_type, "person", "sender unchanged on cancel")
        chk.that(not path.exists(), "cancel wrote nothing to disk")


async def s_unexclude(chk):
    ths = [t for t in _threads() if t.thread_id == "th-excluded"]
    app, path = _app(ths)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter")
        chk.that(ths[0].excluded, "starts excluded")
        await pilot.press("e")
        await pilot.pause()
        chk.that(not ths[0].excluded, "e unexcludes")
        chk.that(path.exists(), "unexclude saved")


async def s_exclude_toggle_on_included(chk):
    # Issue #52: `e` is now a symmetric toggle — on an INCLUDED thread it excludes.
    ths = [t for t in _threads() if t.thread_id == "th-person-fyi"]
    app, path = _app(ths)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter")
        chk.that(not ths[0].excluded, "starts included")
        await pilot.press("e")
        await pilot.pause()
        chk.that(ths[0].excluded, "e excludes an included thread")
        chk.that(path.exists(), "exclude saved")
        await pilot.press("e")              # toggle back
        await pilot.pause()
        chk.that(not ths[0].excluded, "second e un-excludes")


async def s_cycle_assistant(chk):
    # Issue #78: `a` cycles the tri-state annotation unset -> yes -> no -> unset,
    # auto-saving each step like the `e` toggle.
    ths = [t for t in _threads() if t.thread_id == "th-service-nr"]
    app, path = _app(ths)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter")
        chk.eq(ths[0].expected_assistant, None, "starts unset")
        await pilot.press("a")
        await pilot.pause()
        chk.eq(ths[0].expected_assistant, True, "a: unset -> yes")
        chk.that(path.exists(), "assistant cycle auto-saved")
        await pilot.press("a")
        await pilot.pause()
        chk.eq(ths[0].expected_assistant, False, "a: yes -> no")
        await pilot.press("a")
        await pilot.pause()
        chk.eq(ths[0].expected_assistant, None, "a: no -> unset")
        saved = load_golden_set(path)
        chk.eq(saved[0].expected_assistant, None, "final unset persisted to disk")


async def s_scroll_detail(chk):
    from textual.containers import VerticalScroll
    ths = [t for t in _threads() if t.thread_id == "th-longbody"]
    app, _ = _app(ths)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter")
        sc = app.screen.query_one("#detail-scroll", VerticalScroll)
        await pilot.press("down", "down", "down", "pagedown")
        await pilot.pause()
        y_down = sc.scroll_offset.y
        chk.that(y_down > 0, "scroll down/pagedown increased the offset")
        await pilot.press("end")
        await pilot.pause()
        chk.that(sc.scroll_offset.y >= y_down, "end scrolled to (at least) the down offset")
        await pilot.press("up", "pageup", "home")   # reverse-scroll actions
        await pilot.pause()
        chk.eq(sc.scroll_offset.y, 0, "home returned to the top")


async def s_detail_renders_multimsg_and_notes(chk):
    from textual.widgets import Static

    def content(app):
        return str(app.screen.query_one("#detail-content", Static).render())

    mm = [t for t in _threads() if t.thread_id == "th-multimsg"]
    app, _ = _app(mm)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter")
        c = content(app)
        chk.that("[Message 1]" in c and "[Message 3]" in c, "multi-message detail renders each message")
    nt = [t for t in _threads() if t.thread_id == "th-notes"]
    app, _ = _app(nt)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter")
        c = content(app)
        chk.that("Notes:" in c and "pre-existing reviewer note" in c, "notes detail line rendered")


async def s_edit_to_person_and_fyi(chk):
    # Cover 'p'->person and 'f'->fyi (other scenarios only test service/needs_response/low).
    ths = [t for t in _threads() if t.thread_id == "th-service-nr"]
    app, _ = _app(ths)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter", "s")
        await pilot.pause()
        await pilot.press("p")              # -> person
        await pilot.pause()
        chk.eq(ths[0].expected_sender_type, "person", "sender -> person")
        await pilot.press("l")
        await pilot.pause()
        await pilot.press("f")              # -> fyi
        await pilot.pause()
        chk.eq(ths[0].expected_label, "fyi", "label -> fyi")


async def s_quit_from_list(chk):
    ths = _threads()
    app, _ = _app(ths)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("q")              # quit without opening a detail
    chk.eq(app.return_value, "quit", "q quits directly from the list")


async def s_filtered_view_saves_all(chk):
    all_threads = _threads()
    shown = [all_threads[0]]                # filtered display list = 1 thread
    path = Path(tempfile.mkdtemp()) / "golden.jsonl"
    app = EditApp(shown, all_threads, path)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter", "s")
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()
    saved = load_golden_set(path)
    chk.eq(len(saved), len(all_threads), "save persists ALL threads, not just the shown subset")
    chk.eq({t.thread_id for t in saved}, {t.thread_id for t in all_threads}, "all thread ids preserved")


async def s_paging_small_terminal(chk):
    ths = _threads()
    app, _ = _app(ths)
    async with app.run_test(size=(100, 8)) as pilot:
        await pilot.press("end")
        chk.eq(_lv(app).index, len(ths) - 1, "end -> last even on a short terminal")
        await pilot.press("pageup")
        chk.that(_lv(app).index < len(ths) - 1, "pageup moved the cursor up a page")
        await pilot.press("home")
        chk.eq(_lv(app).index, 0, "home -> first")


async def s_quit(chk):
    ths = _threads()
    app, _ = _app(ths)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter", "q")     # quit from detail
    chk.eq(app.return_value, "quit", "q from detail quits the app")


def scenarios():
    return [
        ("list_nav_and_drill", s_list_nav_and_drill),
        ("edit_sender", s_edit_sender),
        ("edit_label_and_collision", s_edit_label_and_collision),
        ("sender_cancel_no_save", s_sender_cancel_no_save),
        ("unexclude", s_unexclude),
        ("exclude_toggle_on_included", s_exclude_toggle_on_included),
        ("cycle_assistant", s_cycle_assistant),
        ("scroll_detail", s_scroll_detail),
        ("detail_renders_multimsg_and_notes", s_detail_renders_multimsg_and_notes),
        ("edit_to_person_and_fyi", s_edit_to_person_and_fyi),
        ("filtered_view_saves_all", s_filtered_view_saves_all),
        ("paging_small_terminal", s_paging_small_terminal),
        ("quit_from_list", s_quit_from_list),
        ("quit", s_quit),
    ]


if __name__ == "__main__":
    results = asyncio.run(run_scenarios(scenarios()))
    sys.exit(report("edit_tui", results))
