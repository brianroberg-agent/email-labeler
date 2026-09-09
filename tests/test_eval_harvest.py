"""Tests for evals.harvest — ground truth inference and deduplication."""

import argparse
import json

import pytest

import evals.harvest as harvest_mod
from evals.harvest import (
    build_query,
    deduplicate,
    gmail_label_arg,
    harvest_threads,
    infer_ground_truth,
    load_existing_thread_ids,
    write_golden_set,
)
from evals.schemas import GoldenThread

# Label config matching the real config.toml structure
LABELS_CONFIG = {
    "needs_response": "agent/needs-response",
    "fyi": "agent/fyi",
    "low_priority": "agent/low-priority",
    "processed": "agent/processed",
    "personal": "agent/personal",
    "non_personal": "agent/non-personal",
}

# Label ID -> name mapping (as returned by Gmail API list_labels)
LABEL_ID_TO_NAME = {
    "Label_1": "agent/needs-response",
    "Label_2": "agent/fyi",
    "Label_3": "agent/low-priority",
    "Label_4": "agent/processed",
    "Label_5": "agent/personal",
    "Label_6": "agent/non-personal",
    "Label_7": "eval/harvest",  # a user label for hand-picked threads
}


class TestInferGroundTruth:
    def test_person_needs_response(self):
        messages = [
            {"labelIds": ["INBOX", "Label_5", "Label_1", "Label_4"]},
        ]
        sender_type, label = infer_ground_truth(messages, LABEL_ID_TO_NAME, LABELS_CONFIG)
        assert sender_type == "person"
        assert label == "needs_response"

    def test_service_low_priority(self):
        messages = [
            {"labelIds": ["Label_6", "Label_3", "Label_4"]},
        ]
        sender_type, label = infer_ground_truth(messages, LABEL_ID_TO_NAME, LABELS_CONFIG)
        assert sender_type == "service"
        assert label == "low_priority"

    def test_person_fyi(self):
        messages = [
            {"labelIds": ["Label_5", "Label_2", "Label_4"]},
        ]
        sender_type, label = infer_ground_truth(messages, LABEL_ID_TO_NAME, LABELS_CONFIG)
        assert sender_type == "person"
        assert label == "fyi"

    def test_multi_message_thread(self):
        """Labels on any message in the thread should be picked up."""
        messages = [
            {"labelIds": ["INBOX"]},
            {"labelIds": ["Label_5"]},  # personal on second message
            {"labelIds": ["Label_1", "Label_4"]},  # needs_response on third
        ]
        sender_type, label = infer_ground_truth(messages, LABEL_ID_TO_NAME, LABELS_CONFIG)
        assert sender_type == "person"
        assert label == "needs_response"

    def test_missing_sender_type(self):
        """Missing sender type label should return empty string."""
        messages = [
            {"labelIds": ["Label_1", "Label_4"]},  # Has label but no personal/non_personal
        ]
        sender_type, label = infer_ground_truth(messages, LABEL_ID_TO_NAME, LABELS_CONFIG)
        assert sender_type == ""
        assert label == "needs_response"

    def test_missing_classification_label(self):
        """Missing classification label should return empty string."""
        messages = [
            {"labelIds": ["Label_5", "Label_4"]},  # Has personal but no classification
        ]
        sender_type, label = infer_ground_truth(messages, LABEL_ID_TO_NAME, LABELS_CONFIG)
        assert sender_type == "person"
        assert label == ""

    def test_no_labels(self):
        messages = [{"labelIds": []}]
        sender_type, label = infer_ground_truth(messages, LABEL_ID_TO_NAME, LABELS_CONFIG)
        assert sender_type == ""
        assert label == ""

    def test_unknown_label_ids(self):
        messages = [{"labelIds": ["UNKNOWN_1", "UNKNOWN_2"]}]
        sender_type, label = infer_ground_truth(messages, LABEL_ID_TO_NAME, LABELS_CONFIG)
        assert sender_type == ""
        assert label == ""

    def test_missing_labelIds_key(self):
        """Messages without labelIds should be handled gracefully."""
        messages = [{}]
        sender_type, label = infer_ground_truth(messages, LABEL_ID_TO_NAME, LABELS_CONFIG)
        assert sender_type == ""
        assert label == ""


class TestDeduplicate:
    def _make_golden(self, thread_id: str) -> GoldenThread:
        return GoldenThread(
            thread_id=thread_id,
            messages=[],
            senders=["test@example.com"],
            subject="Test",
            snippet="Test",
            expected_sender_type="service",
            expected_label="low_priority",
        )

    def test_no_existing_file(self, tmp_path):
        """With no existing file, all threads should be kept."""
        threads = [self._make_golden("t1"), self._make_golden("t2")]
        result = deduplicate(threads, tmp_path / "nonexistent.jsonl")
        assert len(result) == 2

    def test_dedup_removes_existing(self, tmp_path):
        """Threads already in the file should be removed."""
        existing_file = tmp_path / "golden.jsonl"
        existing_file.write_text(
            json.dumps({"thread_id": "t1", "messages": [], "senders": [], "subject": "",
                        "snippet": "", "expected_sender_type": "service",
                        "expected_label": "low_priority"}) + "\n"
        )
        threads = [self._make_golden("t1"), self._make_golden("t2"), self._make_golden("t3")]
        result = deduplicate(threads, existing_file)
        assert len(result) == 2
        assert {t.thread_id for t in result} == {"t2", "t3"}

    def test_all_duplicates(self, tmp_path):
        """If all threads are duplicates, result should be empty."""
        existing_file = tmp_path / "golden.jsonl"
        lines = [
            json.dumps({"thread_id": "t1"}) + "\n",
            json.dumps({"thread_id": "t2"}) + "\n",
        ]
        existing_file.write_text("".join(lines))
        threads = [self._make_golden("t1"), self._make_golden("t2")]
        result = deduplicate(threads, existing_file)
        assert len(result) == 0

    def test_empty_new_threads(self, tmp_path):
        result = deduplicate([], tmp_path / "golden.jsonl")
        assert len(result) == 0

    def test_skips_malformed_line(self, tmp_path):
        """A corrupt/partial line (e.g. interrupted append) must not crash dedup."""
        existing_file = tmp_path / "golden.jsonl"
        existing_file.write_text(
            json.dumps({"thread_id": "t1"}) + "\n"
            + '{"thread_id": "t2", "messages":\n'  # truncated, invalid JSON
        )
        threads = [self._make_golden("t1"), self._make_golden("t2"), self._make_golden("t3")]
        # t1 is recognized as existing; the malformed t2 line is skipped (so t2
        # is treated as new), and dedup completes without raising.
        result = deduplicate(threads, existing_file)
        assert {t.thread_id for t in result} == {"t2", "t3"}

    def test_skips_line_missing_thread_id(self, tmp_path):
        """A row without a thread_id must be ignored, not raise KeyError."""
        existing_file = tmp_path / "golden.jsonl"
        existing_file.write_text(
            json.dumps({"subject": "no id here"}) + "\n"
            + json.dumps({"thread_id": "t1"}) + "\n"
        )
        threads = [self._make_golden("t1"), self._make_golden("t2")]
        result = deduplicate(threads, existing_file)
        assert {t.thread_id for t in result} == {"t2"}


class FakeProxy:
    """Records the Gmail query passed to list_messages; returns no messages."""

    def __init__(self):
        self.last_query = None

    async def list_labels(self, user_id: str = "me"):
        return {"labels": [{"id": lid, "name": name} for lid, name in LABEL_ID_TO_NAME.items()]}

    # Signature mirrors GmailProxyClient.list_messages so a future positional
    # call (e.g. list_messages(query, ...)) would bind the same way it does in
    # production and not silently false-green.
    async def list_messages(self, user_id="me", max_results=10, q=None, label_ids=None):
        self.last_query = q
        return {"messages": []}


class TestBuildQuery:
    """build_query is the pure quote-and-join over resolved Gmail label names."""

    def test_base_query_is_processed_only(self):
        assert build_query("agent/processed") == "label:agent/processed"

    def test_filter_label_appended_quoted(self):
        assert build_query("agent/processed", "agent/needs-response") == (
            'label:agent/processed label:"agent/needs-response"'
        )

    def test_both_labels_ordered_processed_classification_gmail(self):
        assert build_query("agent/processed", "agent/low-priority", "eval/harvest") == (
            'label:agent/processed label:"agent/low-priority" label:"eval/harvest"'
        )

    def test_user_label_with_slash_and_space_is_one_quoted_term(self):
        # Unquoted, a name with a space would split into two search terms.
        assert build_query("agent/processed", gmail_label="eval/cold pitch") == (
            'label:agent/processed label:"eval/cold pitch"'
        )


class TestGmailLabelArg:
    """--gmail-label must be a real, non-empty label name (argparse type)."""

    def test_strips_surrounding_whitespace(self):
        assert gmail_label_arg("  eval/harvest ") == "eval/harvest"

    @pytest.mark.parametrize("value", ["", "   "])
    def test_rejects_empty(self, value):
        # An unset shell variable must not silently widen the harvest to the
        # whole processed pool.
        with pytest.raises(argparse.ArgumentTypeError):
            gmail_label_arg(value)

    def test_rejects_double_quote(self):
        # The name is embedded in a quoted Gmail term and Gmail has no escape.
        with pytest.raises(argparse.ArgumentTypeError):
            gmail_label_arg('eval/"hot" leads')


class TestHarvestQuery:
    """harvest_threads resolves config keys and user labels into the Gmail query."""

    CONFIG = {"labels": LABELS_CONFIG}

    async def test_no_filters_queries_processed_only(self):
        proxy = FakeProxy()
        await harvest_threads(proxy, self.CONFIG, max_threads=10)
        assert proxy.last_query == "label:agent/processed"

    async def test_label_filter_and_gmail_label_anded_into_query(self, capsys):
        proxy = FakeProxy()
        await harvest_threads(
            proxy, self.CONFIG, max_threads=10, label_filter="low_priority", gmail_label="eval/harvest",
        )
        assert proxy.last_query == (
            'label:agent/processed label:"agent/low-priority" label:"eval/harvest"'
        )
        assert "Error" not in capsys.readouterr().err

    async def test_unmapped_label_filter_is_an_error_before_fetching(self, capsys):
        # An unmapped key can never match a thread (infer_ground_truth only
        # knows mapped keys), so a degraded query would just fetch and drop.
        proxy = FakeProxy()
        with pytest.raises(SystemExit):
            await harvest_threads(proxy, self.CONFIG, max_threads=10, label_filter="bogus")
        assert proxy.last_query is None
        assert "has no mapping in [labels]" in capsys.readouterr().err

    async def test_unknown_gmail_label_is_an_error_before_fetching(self, capsys):
        # A typo must not be reported as "no messages found".
        proxy = FakeProxy()
        with pytest.raises(SystemExit):
            await harvest_threads(proxy, self.CONFIG, max_threads=10, gmail_label="eval/harvst")
        assert proxy.last_query is None
        assert "eval/harvst" in capsys.readouterr().err

    async def test_gmail_label_check_is_case_insensitive(self):
        # Gmail's label: search is case-insensitive, so the check must not
        # reject a name Gmail would accept.
        proxy = FakeProxy()
        await harvest_threads(proxy, self.CONFIG, max_threads=10, gmail_label="Eval/Harvest")
        assert proxy.last_query == 'label:agent/processed label:"Eval/Harvest"'


class TestWriteGoldenSet:
    """Harvest must never overwrite an existing golden set — it always appends.

    The golden set also stores manual review state (confirmed labels,
    exclusions, notes), so a truncating write would silently destroy that work.
    """

    def _make_golden(self, thread_id: str) -> GoldenThread:
        return GoldenThread(
            thread_id=thread_id,
            messages=[],
            senders=["test@example.com"],
            subject="Test",
            snippet="Test",
            expected_sender_type="service",
            expected_label="low_priority",
        )

    def test_appends_to_existing_file(self, tmp_path):
        """Existing lines must survive; new threads are added after them."""
        path = tmp_path / "golden.jsonl"
        write_golden_set([self._make_golden("t1")], path)
        write_golden_set([self._make_golden("t2")], path)

        ids = [json.loads(line)["thread_id"] for line in path.read_text().splitlines() if line]
        assert ids == ["t1", "t2"]

    def test_creates_file_when_absent(self, tmp_path):
        """First write to a non-existent path creates it."""
        path = tmp_path / "golden.jsonl"
        write_golden_set([self._make_golden("t1")], path)

        ids = [json.loads(line)["thread_id"] for line in path.read_text().splitlines() if line]
        assert ids == ["t1"]


class _StubProxy:
    """Fake proxy serving harvestable processed threads (person + needs_response).

    Records the query and every get_thread call so tests can assert on what
    was fetched, not just on what came back.
    """

    def __init__(self, *args, thread_ids=("t1",), next_page_token=None,
                 messages_per_thread=1, **kwargs):
        self.thread_ids = list(thread_ids)
        self.next_page_token = next_page_token
        self.messages_per_thread = messages_per_thread
        self.last_query = None
        self.get_thread_calls: list[str] = []

    async def list_labels(self, user_id="me"):
        return {"labels": [{"id": lid, "name": name} for lid, name in LABEL_ID_TO_NAME.items()]}

    async def list_messages(self, user_id="me", max_results=10, q=None, label_ids=None):
        self.last_query = q
        response = {"messages": [
            {"id": f"m-{tid}-{n}", "threadId": tid}
            for tid in self.thread_ids for n in range(self.messages_per_thread)
        ]}
        if self.next_page_token:
            response["nextPageToken"] = self.next_page_token
        return response

    async def get_thread(self, thread_id, user_id="me", format="full"):
        self.get_thread_calls.append(thread_id)
        return {
            "messages": [
                {
                    "id": f"m-{thread_id}",
                    "internalDate": "1000",
                    "snippet": "hello",
                    # personal + needs-response + processed -> a valid golden thread
                    "labelIds": ["Label_5", "Label_1", "Label_4"],
                    "payload": {
                        "headers": [
                            {"name": "From", "value": "alice@example.com"},
                            {"name": "Subject", "value": "Hi"},
                        ]
                    },
                }
            ]
        }


class TestHarvestLoop:
    """Per-thread behavior of harvest_threads: skips, caps, diagnostics, tagging."""

    CONFIG = {"labels": LABELS_CONFIG}

    async def test_known_threads_are_skipped_before_fetching(self, capsys):
        # Re-running with the Gmail label left in place must not re-download
        # threads already in the golden set.
        proxy = _StubProxy(thread_ids=("t1", "t2"))
        results = await harvest_threads(
            proxy, self.CONFIG, max_threads=10, skip_thread_ids={"t1"},
        )
        assert proxy.get_thread_calls == ["t2"]
        assert [t.thread_id for t in results] == ["t2"]
        assert "already in the golden set" in capsys.readouterr().err

    async def test_known_threads_do_not_consume_the_cap(self):
        # Otherwise a label larger than --max-threads could never reach its
        # older, not-yet-harvested picks on a re-run.
        proxy = _StubProxy(thread_ids=("t1", "t2"))
        results = await harvest_threads(
            proxy, self.CONFIG, max_threads=1, skip_thread_ids={"t1"},
        )
        assert [t.thread_id for t in results] == ["t2"]

    async def test_warns_when_message_budget_exhausted_below_cap(self, capsys):
        # The fetch is a message-level budget with no pagination; when Gmail
        # reports more pages and fewer threads than the cap came back, some
        # matching threads were silently left behind.
        proxy = _StubProxy(thread_ids=("t1",), next_page_token="abc")
        await harvest_threads(proxy, self.CONFIG, max_threads=10)
        assert "budget exhausted" in capsys.readouterr().err

    async def test_budget_warning_counts_only_unharvested_threads(self, capsys):
        # Re-run regime: the label is larger than the message window and every
        # thread in the window is already harvested. The cap is not reached by
        # NEW threads, so the older picks beyond the window are being missed —
        # warn on the post-skip count, not the raw thread count.
        proxy = _StubProxy(thread_ids=("t1", "t2"), next_page_token="abc")
        await harvest_threads(
            proxy, self.CONFIG, max_threads=2, skip_thread_ids={"t1", "t2"},
        )
        assert "budget exhausted" in capsys.readouterr().err

    async def test_warns_when_window_is_full_without_page_token(self, capsys):
        # Gmail need not return nextPageToken; a page that fills the whole
        # message window (max_threads * 3) with fewer threads than the cap is
        # the same silent truncation.
        proxy = _StubProxy(thread_ids=("t1",), messages_per_thread=6)
        await harvest_threads(proxy, self.CONFIG, max_threads=2)
        assert "budget exhausted" in capsys.readouterr().err

    async def test_no_budget_warning_when_cap_reached(self, capsys):
        proxy = _StubProxy(thread_ids=("t1", "t2"), next_page_token="abc")
        await harvest_threads(proxy, self.CONFIG, max_threads=2)
        assert "budget exhausted" not in capsys.readouterr().err

    async def test_filter_mismatch_names_the_skipped_thread(self, capsys):
        # "Labelled 10, harvested 7" must be diagnosable per thread.
        proxy = _StubProxy(thread_ids=("t1",))
        results = await harvest_threads(
            proxy, self.CONFIG, max_threads=10, sender_type_filter="service",
        )
        assert results == []
        assert "Skipping thread t1" in capsys.readouterr().err

    async def test_hand_picked_rows_are_tagged_in_notes(self):
        # Hand-picked rows carry labels the daemon (by hypothesis) got wrong;
        # the review TUI shows notes, so name the label there.
        proxy = _StubProxy(thread_ids=("t1",))
        results = await harvest_threads(proxy, self.CONFIG, max_threads=10, gmail_label="eval/harvest")
        assert "eval/harvest" in results[0].notes

    async def test_bulk_rows_have_no_notes(self):
        proxy = _StubProxy(thread_ids=("t1",))
        results = await harvest_threads(proxy, self.CONFIG, max_threads=10)
        assert results[0].notes == ""


class TestLoadExistingThreadIds:
    def test_reads_ids_and_tolerates_bad_rows(self, tmp_path):
        path = tmp_path / "golden.jsonl"
        path.write_text(
            json.dumps({"thread_id": "t1"}) + "\n"
            + '{"thread_id": "t2", "messages":\n'  # truncated
            + json.dumps({"subject": "no id"}) + "\n"
        )
        assert load_existing_thread_ids(path) == {"t1"}

    def test_missing_file_is_empty(self, tmp_path):
        assert load_existing_thread_ids(tmp_path / "nope.jsonl") == set()


def _main_args(output, **overrides) -> argparse.Namespace:
    """A Namespace shaped like cli()'s parser output (no getattr defaults in main)."""
    fields = dict(
        output=str(output), max_threads=10, sender_type=None, label=None,
        gmail_label=None, config=None, proxy_url="http://x",
    )
    fields.update(overrides)
    return argparse.Namespace(**fields)


class TestMain:
    """main() wiring: dedup on every run, no refetch of known threads, flag forwarding."""

    async def test_rerun_does_not_duplicate_thread(self, tmp_path, monkeypatch):
        # Guards the always-append behavior: a second harvest of an
        # already-present thread must not append a duplicate row.
        monkeypatch.setattr(harvest_mod, "GmailProxyClient", _StubProxy)
        output = tmp_path / "golden.jsonl"
        args = _main_args(output)

        await harvest_mod.main(args)
        await harvest_mod.main(args)

        ids = [json.loads(line)["thread_id"] for line in output.read_text().splitlines() if line]
        assert ids == ["t1"]

    async def test_rerun_does_not_refetch_known_threads(self, tmp_path, monkeypatch):
        proxies: list[_StubProxy] = []

        def make_proxy(*args, **kwargs):
            proxies.append(_StubProxy())
            return proxies[-1]

        monkeypatch.setattr(harvest_mod, "GmailProxyClient", make_proxy)
        args = _main_args(tmp_path / "golden.jsonl")

        await harvest_mod.main(args)
        await harvest_mod.main(args)

        assert proxies[0].get_thread_calls == ["t1"]
        assert proxies[1].get_thread_calls == []

    async def test_gmail_label_reaches_the_query(self, tmp_path, monkeypatch):
        proxy = _StubProxy()
        monkeypatch.setattr(harvest_mod, "GmailProxyClient", lambda *a, **kw: proxy)

        await harvest_mod.main(_main_args(tmp_path / "golden.jsonl", gmail_label="eval/harvest"))

        assert proxy.last_query == 'label:agent/processed label:"eval/harvest"'
