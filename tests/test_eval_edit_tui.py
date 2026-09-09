"""Tests for evals.edit_tui — pure rendering helpers + Pilot UI tests.

The Textual UI layer is driven with Textual's Pilot: real key presses in,
widget state / rendered content / on-disk golden-set effects out.
"""

import base64

from textual.widgets import Label, ListView, Static

from evals.edit_tui import DetailScreen, EditApp, _build_detail_lines, _format_list_row, run_edit_tui
from evals.review import load_golden_set
from evals.schemas import GoldenThread

SIZE = (100, 30)


def _golden(thread_id, **kw):
    base = dict(
        thread_id=thread_id, messages=[], senders=[], subject="Subj", snippet="snip",
        expected_sender_type="person", expected_label="fyi", reviewed=True,
    )
    base.update(kw)
    return GoldenThread(**base)


def _body_message(text: str) -> dict:
    data = base64.urlsafe_b64encode(text.encode()).decode()
    return {"payload": {"mimeType": "text/plain", "body": {"data": data}}}


def _edit_app(threads, tmp_path, all_threads=None):
    return EditApp(
        threads,
        all_threads if all_threads is not None else threads,
        tmp_path / "golden.jsonl",
    )


def _screen_text(app) -> str:
    parts = [str(w.render()) for w in app.screen.query(Static)]
    parts += [str(w.render()) for w in app.screen.query(Label)]
    return "\n".join(parts)


class TestBuildDetailLines:
    def test_excluded_thread_shows_excluded_label(self):
        thread = _golden("t1", excluded=True)
        lines = _build_detail_lines(thread, 0, 1)
        assert any(line.startswith("Excluded:") for line in lines)
        assert not any("Skipped:" in line for line in lines)

    def test_non_excluded_thread_omits_excluded_label(self):
        thread = _golden("t1", excluded=False)
        lines = _build_detail_lines(thread, 0, 1)
        assert not any(line.startswith("Excluded:") for line in lines)


class TestListRowExcludedMarker:
    def test_excluded_row_is_marked(self):
        row = _format_list_row(_golden("t1", excluded=True), max_x=80)
        assert row.lstrip().startswith("X")

    def test_non_excluded_row_has_no_marker(self):
        row = _format_list_row(_golden("t1", excluded=False), max_x=80)
        assert not row.lstrip().startswith("X")


class TestEditAppList:
    async def test_list_shows_threads_and_q_quits(self, tmp_path):
        threads = [_golden("t1", subject="first"), _golden("t2", subject="second")]
        app = _edit_app(threads, tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            assert len(app.query_one(ListView)) == 2
            assert "Edit Mode — 2 threads" in _screen_text(app)
            await pilot.press("q")
        assert app.return_value == "quit"

    async def test_enter_opens_detail_with_content_and_status(self, tmp_path):
        threads = [_golden("t1"), _golden("t2", subject="The second thread")]
        app = _edit_app(threads, tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("down", "enter")
            assert isinstance(app.screen, DetailScreen)
            text = _screen_text(app)
            assert "Thread 2/2" in text
            assert "The second thread" in text
            assert "Sender: person  Label: fyi" in text

    async def test_escape_returns_to_list_preserving_cursor(self, tmp_path):
        threads = [_golden("t1"), _golden("t2")]
        app = _edit_app(threads, tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("down", "enter")
            assert isinstance(app.screen, DetailScreen)
            await pilot.press("escape")
            assert not isinstance(app.screen, DetailScreen)
            assert app.query_one(ListView).index == 1

    async def test_q_from_detail_quits_whole_app(self, tmp_path):
        app = _edit_app([_golden("t1")], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "q")
        assert app.return_value == "quit"


class TestEditActions:
    async def test_s_then_s_sets_service_and_saves(self, tmp_path):
        thread = _golden("t1")
        app = _edit_app([thread], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("s", "s")  # sender prompt -> service
            assert thread.expected_sender_type == "service"
            assert "Sender: service" in _screen_text(app)
            saved = load_golden_set(tmp_path / "golden.jsonl")
            assert saved[0].expected_sender_type == "service"

    async def test_s_then_other_key_cancels(self, tmp_path):
        thread = _golden("t1")
        app = _edit_app([thread], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("s", "z")  # z is not a sender key -> cancel
            assert thread.expected_sender_type == "person"
            assert not (tmp_path / "golden.jsonl").exists()  # nothing saved

    async def test_l_then_r_sets_needs_response(self, tmp_path):
        thread = _golden("t1")
        app = _edit_app([thread], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("l", "r")
            assert thread.expected_label == "needs_response"
            saved = load_golden_set(tmp_path / "golden.jsonl")
            assert saved[0].expected_label == "needs_response"

    async def test_l_then_l_sets_low_priority(self, tmp_path):
        # 'l' opens the prompt where 'l' then means low_priority — the key
        # collision the modal must not double-fire on.
        thread = _golden("t1")
        app = _edit_app([thread], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("l", "l")
            assert thread.expected_label == "low_priority"

    async def test_e_unexcludes_and_saves(self, tmp_path):
        thread = _golden("t1", excluded=True)
        app = _edit_app([thread], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            assert "[e]unexclude" in _screen_text(app)
            await pilot.press("e")
            assert thread.excluded is False
            assert "[e]unexclude" not in _screen_text(app)
            saved = load_golden_set(tmp_path / "golden.jsonl")
            assert saved[0].excluded is False
            await pilot.press("escape")
            assert "X" not in str(app.query_one(ListView).children[0].query_one(Label).render())

    async def test_e_toggles_exclude_on_and_saves(self, tmp_path):
        # Issue #52: `e` now EXCLUDES an included thread (symmetric toggle), not
        # just un-excludes. The hint is always shown, state-dependent.
        thread = _golden("t1", excluded=False)
        app = _edit_app([thread], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            assert "[e]exclude" in _screen_text(app)
            await pilot.press("e")
            assert thread.excluded is True
            assert "[e]unexclude" in _screen_text(app)
            saved = load_golden_set(tmp_path / "golden.jsonl")
            assert saved[0].excluded is True

    async def test_e_round_trip_toggle(self, tmp_path):
        thread = _golden("t1", excluded=False)
        app = _edit_app([thread], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("e")
            assert thread.excluded is True
            await pilot.press("e")
            assert thread.excluded is False

    async def test_e_leaves_reviewed_untouched(self, tmp_path):
        # Parity with the newsletter toggle: excluding does not flip reviewed.
        thread = _golden("t1", excluded=False, reviewed=True)
        app = _edit_app([thread], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("e")
            assert thread.excluded is True
            assert thread.reviewed is True

    async def test_saves_all_threads_not_filtered_view(self, tmp_path):
        hidden = _golden("hidden", expected_label="low_priority")
        shown = _golden("shown")
        app = _edit_app([shown], tmp_path, all_threads=[hidden, shown])
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("l", "r")
            saved = load_golden_set(tmp_path / "golden.jsonl")
            assert [t.thread_id for t in saved] == ["hidden", "shown"]
            assert saved[0].expected_label == "low_priority"
            assert saved[1].expected_label == "needs_response"


class TestDetailScroll:
    async def test_detail_scrolls_with_arrow_keys(self, tmp_path):
        body = "\n".join(f"line {i}" for i in range(60))
        thread = _golden("t1", messages=[_body_message(body)])
        app = _edit_app([thread], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            scroll = app.screen.query_one("#detail-scroll")
            assert scroll.scroll_offset.y == 0
            await pilot.press("down", "down", "down")
            assert scroll.scroll_offset.y == 3

    async def test_detail_shows_decoded_body(self, tmp_path):
        thread = _golden("t1", messages=[_body_message("Hello from the body")])
        app = _edit_app([thread], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            text = _screen_text(app)
            assert "[Message 1]" in text
            assert "Hello from the body" in text


class TestRunEditTui:
    def test_empty_threads_prints_and_returns(self, capsys):
        run_edit_tui([], [], "unused-path")
        assert "No threads to edit." in capsys.readouterr().out


class TestListCursorKeys:
    async def test_home_and_end_move_the_cursor(self, tmp_path):
        from textual.widgets import ListView

        threads = [_golden(f"t{i}") for i in range(12)]
        app = _edit_app(threads, tmp_path)
        async with app.run_test(size=(100, 8)) as pilot:
            await pilot.press("end")
            assert app.query_one(ListView).index == 11
            await pilot.press("home")
            assert app.query_one(ListView).index == 0


class TestEditAppReviewFindings:
    async def test_enter_auto_repeat_opens_a_single_detail(self, tmp_path):
        from textual import events

        app = _edit_app([_golden("t1"), _golden("t2")], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            app.post_message(events.Key("enter", None))
            app.post_message(events.Key("enter", None))
            await pilot.pause()
            assert len(app.screen_stack) == 2  # base + ONE detail
            await pilot.press("escape")
            assert not isinstance(app.screen, DetailScreen)

    async def test_resize_rerenders_rows_at_new_width(self, tmp_path):
        long_subject = "S" * 90
        app = _edit_app([_golden("t1", subject=long_subject)], tmp_path)
        async with app.run_test(size=(60, 20)) as pilot:
            assert long_subject not in _screen_text(app)  # truncated at 60 cols
            await pilot.resize_terminal(160, 24)
            await pilot.pause()
            assert long_subject in _screen_text(app)  # re-rendered wider


class TestAssistantKey:
    """Issue #78: `a` cycles the tri-state annotation, mirroring `e`."""

    async def test_a_cycles_unset_yes_no_unset_and_saves(self, tmp_path):
        thread = _golden("t1", expected_label="needs_response")
        app = _edit_app([thread], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            assert thread.expected_assistant is None
            await pilot.press("a")
            assert thread.expected_assistant is True
            assert load_golden_set(tmp_path / "golden.jsonl")[0].expected_assistant is True
            await pilot.press("a")
            assert thread.expected_assistant is False
            assert load_golden_set(tmp_path / "golden.jsonl")[0].expected_assistant is False
            await pilot.press("a")
            assert thread.expected_assistant is None
            assert load_golden_set(tmp_path / "golden.jsonl")[0].expected_assistant is None

    async def test_detail_shows_the_annotation_and_offers_the_key(self, tmp_path):
        thread = _golden("t1", expected_label="needs_response")
        app = _edit_app([thread], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            assert "[a]ssistant" in _screen_text(app)
            assert "Assistant: unset" in _screen_text(app)
            await pilot.press("a")
            assert "Assistant: yes" in _screen_text(app)
            await pilot.press("a")
            assert "Assistant: no" in _screen_text(app)

    async def test_a_leaves_reviewed_and_excluded_untouched(self, tmp_path):
        thread = _golden("t1", expected_label="needs_response", reviewed=True, excluded=False)
        app = _edit_app([thread], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("a")
            assert thread.reviewed is True
            assert thread.excluded is False


class TestAssistantUnknownValue:
    """A hand-edited golden-set file can carry a value the maps do not know."""

    def test_detail_lines_render_an_unknown_value_instead_of_crashing(self):
        thread = _golden("t1", expected_label="needs_response")
        thread.expected_assistant = "yes"       # hand-edited string, not a bool
        lines = _build_detail_lines(thread, 0, 1)
        assert any(line.startswith("Assistant:") and "???" in line for line in lines)

    async def test_a_on_an_unknown_value_clears_it(self, tmp_path):
        thread = _golden("t1", expected_label="needs_response")
        thread.expected_assistant = "yes"
        app = _edit_app([thread], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            assert "Assistant: ???" in _screen_text(app)
            await pilot.press("a")
            assert thread.expected_assistant is None
            assert "Assistant: unset" in _screen_text(app)
            await pilot.press("a")
            assert thread.expected_assistant is True


class TestRelabelClearsAssistantInEditTui:
    async def test_relabel_away_from_needs_response_clears_it(self, tmp_path):
        thread = _golden("t1", expected_label="needs_response", expected_assistant=True)
        app = _edit_app([thread], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("l", "f")  # label submenu -> fyi
            assert thread.expected_label == "fyi"
            assert thread.expected_assistant is None
            assert load_golden_set(tmp_path / "golden.jsonl")[0].expected_assistant is None

    async def test_relabel_to_needs_response_keeps_it(self, tmp_path):
        thread = _golden("t1", expected_label="fyi", expected_assistant=True)
        app = _edit_app([thread], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("l", "r")  # label submenu -> needs_response
            assert thread.expected_assistant is True
