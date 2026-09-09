"""Tests for evals.harvest — ground truth inference and deduplication."""

import argparse
import json

import evals.harvest as harvest_mod
from evals.harvest import deduplicate, harvest_threads, infer_ground_truth, write_golden_set
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

    # Issue #84: a thread the daemon upgraded (e.g. fyi -> needs_response) carries BOTH
    # classification labels, and Gmail's labelIds are unordered. Ground truth must be
    # the highest-priority label regardless of listing order, never last-wins.

    def test_two_labels_needs_response_listed_first(self):
        messages = [{"labelIds": ["Label_1", "Label_2"]}]  # needs_response, fyi
        _, label = infer_ground_truth(messages, LABEL_ID_TO_NAME, LABELS_CONFIG)
        assert label == "needs_response"

    def test_two_labels_needs_response_listed_last(self):
        messages = [{"labelIds": ["Label_2", "Label_1"]}]  # fyi, needs_response
        _, label = infer_ground_truth(messages, LABEL_ID_TO_NAME, LABELS_CONFIG)
        assert label == "needs_response"

    def test_three_labels_across_messages_prefers_highest(self):
        """Lower labels listed later, on later messages, must not win."""
        messages = [
            {"labelIds": ["Label_1", "Label_4"]},  # needs_response on the first message
            {"labelIds": ["Label_2"]},  # fyi added later
            {"labelIds": ["Label_3"]},  # low_priority added later still
        ]
        _, label = infer_ground_truth(messages, LABEL_ID_TO_NAME, LABELS_CONFIG)
        assert label == "needs_response"

    def test_fyi_beats_low_priority_either_order(self):
        for ids in (["Label_2", "Label_3"], ["Label_3", "Label_2"]):
            _, label = infer_ground_truth([{"labelIds": ids}], LABEL_ID_TO_NAME, LABELS_CONFIG)
            assert label == "fyi", ids

    def test_priority_parity_with_daemon(self):
        """Harvest's preference must equal the daemon's own never-downgrade rule.

        Oracle: ``LabelManager.get_existing_priority`` ranks the same labelIds; the
        harvest answer must be the label at that rank in ``labeler._PRIORITY_ORDER``
        (or "" when the daemon sees no classification label). Every subset of the
        three labels, in every order, so the two cannot drift apart silently.
        """
        # Imports are local so this test adds no module-level import lines (PR #76
        # rewrites this file's import block; keeping the rebase trivial).
        from itertools import permutations
        from unittest.mock import MagicMock

        from classifier import EmailLabel
        from evals.harvest import _CLASSIFICATION_LABELS
        from labeler import _PRIORITY_ORDER, LabelManager

        # Vocabulary parity: a daemon label harvest does not know would be silently
        # ignored; a harvest label the daemon does not know has no priority.
        assert set(_CLASSIFICATION_LABELS) == {e.value for e in EmailLabel}
        assert set(_PRIORITY_ORDER) == set(EmailLabel)

        lm = LabelManager(MagicMock(), {"labels": LABELS_CONFIG})
        lm.label_ids = {name: label_id for label_id, name in LABEL_ID_TO_NAME.items()}

        classification_ids = ["Label_1", "Label_2", "Label_3"]
        checked = 0
        for n in range(len(classification_ids) + 1):
            for ids in permutations(classification_ids, n):
                messages = [{"labelIds": ["Label_5", *ids, "Label_4"]}]
                rank = lm.get_existing_priority(messages)
                expected = _PRIORITY_ORDER[rank].value if rank is not None else ""
                _, label = infer_ground_truth(messages, LABEL_ID_TO_NAME, LABELS_CONFIG)
                assert label == expected, (ids, label, expected)
                checked += 1
        assert checked == 16  # 1 + 3 + 6 + 6 orderings


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


class TestHarvestQuery:
    """The Gmail query should AND in the classification label when filtering."""

    CONFIG = {"labels": LABELS_CONFIG}

    async def test_no_label_filter_queries_processed_only(self):
        proxy = FakeProxy()
        await harvest_threads(proxy, self.CONFIG, max_threads=10)
        assert proxy.last_query == "label:agent/processed"

    async def test_label_filter_anded_into_query(self):
        proxy = FakeProxy()
        await harvest_threads(proxy, self.CONFIG, max_threads=10, label_filter="needs_response")
        # The classification label is quoted: its internal hyphen must not be
        # read by Gmail as the NOT operator.
        assert proxy.last_query == 'label:agent/processed label:"agent/needs-response"'

    async def test_unknown_label_filter_falls_back_to_processed(self):
        proxy = FakeProxy()
        await harvest_threads(proxy, self.CONFIG, max_threads=10, label_filter="bogus")
        assert proxy.last_query == "label:agent/processed"


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


class _OneThreadProxy:
    """Fake proxy that always surfaces a single harvestable processed thread."""

    def __init__(self, *args, **kwargs):
        pass

    async def list_labels(self, user_id="me"):
        return {"labels": [{"id": lid, "name": name} for lid, name in LABEL_ID_TO_NAME.items()]}

    async def list_messages(self, user_id="me", max_results=10, q=None, label_ids=None):
        return {"messages": [{"id": "m1", "threadId": "t1"}]}

    async def get_thread(self, thread_id, user_id="me", format="full"):
        return {
            "messages": [
                {
                    "id": "m1",
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


class TestMainDeduplicates:
    """main() must dedup on EVERY run, not only under the (removed) --append flag.

    Guards the central behavior flip of this change: a second harvest of an
    already-present thread must not append a duplicate row.
    """

    async def test_rerun_does_not_duplicate_thread(self, tmp_path, monkeypatch):
        monkeypatch.setattr(harvest_mod, "GmailProxyClient", _OneThreadProxy)
        output = tmp_path / "golden.jsonl"
        args = argparse.Namespace(
            output=str(output),
            max_threads=10,
            sender_type=None,
            label=None,
            config=None,
            proxy_url="http://x",
        )

        await harvest_mod.main(args)
        await harvest_mod.main(args)

        ids = [json.loads(line)["thread_id"] for line in output.read_text().splitlines() if line]
        assert ids == ["t1"]
