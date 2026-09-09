"""Textual TUI for editing already-reviewed golden set threads.

Launched via ``python -m evals.review --edit``.  Provides a list view of
threads with keyboard navigation and a detail view for editing stage 1
(sender type) and stage 2 (label) decisions. Every accepted edit is
auto-saved atomically to the golden-set file.
"""

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Label, ListItem, ListView, Static

# Hotkey maps come from evals.review so the two tools stay in lock-step
# (needs_response is `r`, not `n`). No cycle: review imports edit_tui lazily.
from evals.review import _LABEL_KEY_MAP, _SENDER_KEY_MAP, apply_label, save_golden_set
from evals.schemas import GoldenThread
from gmail_utils import decode_body
from tui_common import CANCEL, HintScreen, KeyMenuScreen, PageListView
from tui_common import truncate as _truncate

# Abbreviation maps for compact list display
_SENDER_ABBREV = {"person": "PER", "service": "SVC"}
_LABEL_ABBREV = {"needs_response": "NR", "fyi": "FYI", "low_priority": "LP"}
# Tri-state assistant annotation (issue #78), for display.
_ASSISTANT_DISPLAY = {None: "unset", True: "yes", False: "no"}
# The `a` cycle: unset -> yes -> no -> unset.
_ASSISTANT_CYCLE = {None: True, True: False, False: None}

# Column widths for list view
_COL_FLAG = 1     # "X" for excluded threads, else blank
_COL_S1 = 5       # "PER" / "SVC"
_COL_S2 = 5       # "NR" / "FYI" / "LP"
_COL_SENDER = 32  # sender name + email
_COL_GAP = 2      # space between columns


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _format_list_row(thread: GoldenThread, max_x: int) -> str:
    """Format a single thread as one list-view line."""
    flag = "X" if thread.excluded else " "
    s1 = _SENDER_ABBREV.get(thread.expected_sender_type, "???")
    s2 = _LABEL_ABBREV.get(thread.expected_label, "???")

    sender = thread.senders[0] if thread.senders else "?"
    sender = _truncate(sender, _COL_SENDER)

    fixed_width = _COL_FLAG + _COL_S1 + _COL_S2 + _COL_SENDER + _COL_GAP * 4
    subject_width = max(10, max_x - fixed_width)
    subject = _truncate(thread.subject, subject_width)

    return (
        f"{flag:<{_COL_FLAG}}"
        f"  {s1:<{_COL_S1}}"
        f"  {s2:<{_COL_S2}}"
        f"  {sender:<{_COL_SENDER}}"
        f"  {subject}"
    )


def _build_detail_lines(thread: GoldenThread, index: int, total: int) -> list[str]:
    """Build content lines for the detail view."""
    lines = [
        f"Thread {index + 1}/{total}  (id: {thread.thread_id})",
        "=" * 60,
        f"Subject:     {thread.subject}",
        f"Senders:     {', '.join(thread.senders)}",
        f"Snippet:     {thread.snippet}",
        f"Source:      {thread.source}",
        f"Reviewed:    {thread.reviewed}",
    ]
    if thread.notes:
        lines.append(f"Notes:       {thread.notes}")
    if thread.excluded:
        lines.append("Excluded:    True")

    lines.append("")
    lines.append(f"Sender type: {thread.expected_sender_type}")
    lines.append(f"Label:       {thread.expected_label}")
    lines.append(f"Assistant:   {_ASSISTANT_DISPLAY[thread.expected_assistant]}")
    lines.append("")
    lines.append("--- Body ---")

    for i, msg in enumerate(thread.messages):
        body = decode_body(msg.get("payload", {}))
        header = f"[Message {i + 1}]"
        lines.append(header)
        lines.extend(body.splitlines())
        lines.append("")

    return lines


# ---------------------------------------------------------------------------
# Textual UI layer
# ---------------------------------------------------------------------------

_SENDER_PROMPT = "[p]erson  [s]ervice  (other key cancels)"
_LABEL_PROMPT = "[r] needs_response  [f]yi  [l]ow_priority  (other key cancels)"


class DetailScreen(Screen):
    """One thread: scrollable detail + s/l/e/a edit actions with auto-save."""

    AUTO_FOCUS = None  # keys go to the screen bindings, not the scroll container

    BINDINGS = [
        Binding("escape", "back", "Back", show=False),
        Binding("q", "quit_app", "Quit", show=False),
        Binding("s", "edit_sender", "Sender", show=False),
        Binding("l", "edit_label", "Label", show=False),
        Binding("e", "toggle_excluded", "Toggle exclude", show=False),
        Binding("a", "cycle_assistant", "Cycle assistant", show=False),
        Binding("up", "scroll_up", "Scroll up", show=False),
        Binding("down", "scroll_down", "Scroll down", show=False),
        Binding("pageup", "page_up", "Page up", show=False),
        Binding("pagedown", "page_down", "Page down", show=False),
        Binding("home", "scroll_home", "Top", show=False),
        Binding("end", "scroll_end", "Bottom", show=False),
    ]

    DEFAULT_CSS = """
    DetailScreen > #detail-scroll {
        height: 1fr;
    }
    DetailScreen #detail-help {
        color: $text-muted;
    }
    DetailScreen #detail-status {
        text-style: bold;
    }
    """

    def __init__(self, threads, index, all_threads, path):
        super().__init__()
        self.thread = threads[index]
        self.index = index
        self.total = len(threads)
        self.all_threads = all_threads
        self.path = path

    def compose(self) -> ComposeResult:
        yield VerticalScroll(Static(id="detail-content", markup=False), id="detail-scroll")
        yield Static(id="detail-help", markup=False)
        yield Static(id="detail-status", markup=False)

    def on_mount(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        lines = _build_detail_lines(self.thread, self.index, self.total)
        self.query_one("#detail-content", Static).update("\n".join(lines))
        excl = "[e]unexclude" if self.thread.excluded else "[e]exclude"
        help_text = (
            f"\u2191/\u2193:Scroll  [s]ender  [l]abel  [a]ssistant  {excl}  Esc:Back  q:Quit"
        )
        self.query_one("#detail-help", Static).update(help_text)
        status = (
            f"Sender: {self.thread.expected_sender_type}  Label: {self.thread.expected_label}"
            f"  Assistant: {_ASSISTANT_DISPLAY[self.thread.expected_assistant]}"
        )
        self.query_one("#detail-status", Static).update(status)

    def _auto_save(self) -> None:
        """Atomically save the FULL golden set, notifying on failure."""
        try:
            save_golden_set(self.all_threads, self.path)
        except Exception as exc:
            # Non-fatal: the in-memory edit survives, the next edit retries.
            self.app.push_screen(HintScreen(f"Save failed: {exc}"))

    # -- edit actions --------------------------------------------------------

    def action_edit_sender(self) -> None:
        def apply(result) -> None:
            if result != CANCEL:
                self.thread.expected_sender_type = result
                self._auto_save()
                self._refresh()

        self.app.push_screen(KeyMenuScreen(_SENDER_PROMPT, _SENDER_KEY_MAP), apply)

    def action_edit_label(self) -> None:
        def apply(result) -> None:
            if result != CANCEL:
                # apply_label drops the assistant annotation when the label
                # moves off needs_response (issue #78 relabel invariant).
                apply_label(self.thread, result)
                self._auto_save()
                self._refresh()

        self.app.push_screen(KeyMenuScreen(_LABEL_PROMPT, _LABEL_KEY_MAP), apply)

    def action_toggle_excluded(self) -> None:
        # Symmetric exclude/unexclude toggle (issue #52). `reviewed` is left
        # untouched — excluded is orthogonal to reviewed, matching the newsletter
        # labeler's toggle. Reversible by pressing `e` again.
        self.thread.excluded = not self.thread.excluded
        self._auto_save()
        self._refresh()

    def action_cycle_assistant(self) -> None:
        # Tri-state cycle unset -> yes -> no -> unset, mirroring the `e` toggle:
        # same auto-save, `reviewed` and `excluded` left untouched. Reversible
        # by pressing `a` until the value comes round again.
        self.thread.expected_assistant = _ASSISTANT_CYCLE[self.thread.expected_assistant]
        self._auto_save()
        self._refresh()

    # -- navigation ----------------------------------------------------------

    def action_back(self) -> None:
        self.dismiss("back")

    def action_quit_app(self) -> None:
        self.dismiss("quit")

    def _scroll(self) -> VerticalScroll:
        return self.query_one("#detail-scroll", VerticalScroll)

    def action_scroll_up(self) -> None:
        self._scroll().scroll_relative(y=-1, animate=False)

    def action_scroll_down(self) -> None:
        self._scroll().scroll_relative(y=1, animate=False)

    def action_page_up(self) -> None:
        self._scroll().scroll_page_up(animate=False)

    def action_page_down(self) -> None:
        self._scroll().scroll_page_down(animate=False)

    def action_scroll_home(self) -> None:
        self._scroll().scroll_home(animate=False)

    def action_scroll_end(self) -> None:
        self._scroll().scroll_end(animate=False)


class EditApp(App):
    """Golden-set editor: list of threads, drill into an editable detail view."""

    BINDINGS = [Binding("q", "quit_app", "Quit")]

    DEFAULT_CSS = """
    EditApp #threads {
        height: 1fr;
    }
    EditApp #list-header {
        text-style: underline;
    }
    EditApp #list-help {
        color: $text-muted;
    }
    """

    def __init__(self, threads, all_threads, path):
        super().__init__()
        self.threads = threads
        self.all_threads = all_threads
        self.path = Path(path)

    def compose(self) -> ComposeResult:
        yield Static(f"Edit Mode \u2014 {len(self.threads)} threads", id="list-title", markup=False)
        hdr = (
            f"{'X':<{_COL_FLAG}}"
            f"  {'S1':<{_COL_S1}}"
            f"  {'S2':<{_COL_S2}}"
            f"  {'Sender':<{_COL_SENDER}}"
            f"  Subject"
        )
        yield Static(hdr, id="list-header", markup=False)
        yield PageListView(id="threads")
        help_text = "\u2191/\u2193:Nav  PgUp/PgDn:Page  Enter:Detail  q:Quit  (X = excluded)"
        yield Static(help_text, id="list-help", markup=False)

    def on_mount(self) -> None:
        self._refresh_list()

    def on_resize(self, event) -> None:
        # Re-render rows so column truncation tracks the new width. self.size
        # is still the OLD size while this handler runs — use the event's.
        self._refresh_list(width=event.size.width)

    def _refresh_list(self, width: int | None = None) -> None:
        width = max(40, width if width is not None else self.size.width)
        listview = self.query_one("#threads", ListView)
        cursor = listview.index or 0
        listview.clear()
        listview.extend(
            ListItem(Label(_format_list_row(t, width), markup=False)) for t in self.threads
        )
        if self.threads:
            listview.index = min(cursor, len(self.threads) - 1)

    def on_list_view_selected(self, event) -> None:
        event.stop()
        if len(self.screen_stack) > 1:
            return  # a detail/modal is already up (Enter auto-repeat)
        index = self.query_one("#threads", ListView).index
        if index is None:
            return

        def on_dismiss(result) -> None:
            if result == "quit":
                self.exit("quit")
            else:
                self._refresh_list()  # X/S1/S2 flags may have changed

        self.push_screen(DetailScreen(self.threads, index, self.all_threads, self.path), on_dismiss)

    def action_quit_app(self) -> None:
        self.exit("quit")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_edit_tui(threads: list[GoldenThread], all_threads: list[GoldenThread], path: Path) -> None:
    """Launch the edit TUI.

    *threads* is the (possibly filtered) list displayed in the TUI.
    *all_threads* is the full golden set used for saving.
    *path* is the golden set file path for auto-save.
    """
    if not threads:
        print("No threads to edit.")
        return
    EditApp(threads, all_threads, Path(path)).run()
