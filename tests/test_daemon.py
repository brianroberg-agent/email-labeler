"""Tests for daemon orchestration."""

import asyncio
import base64
import copy
import json
import logging
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import daemon
from classifier import (
    ClassificationResult,
    EmailLabel,
    SenderType,
)
from daemon import (
    DaemonHalt,
    FailureTracker,
    IdleState,
    format_thread_transcript,
    idle_report,
    load_config,
    log_local_deferrals,
    process_single_thread,
    resolve_int_env,
    summarize_cycle,
)
from labeler import LabelManager, _get_priority
from llm_client import AvailabilityResult, LLMBalanceError, LLMClient, LLMUnavailableError
from newsletter import NewsletterTier, StoryResult
from proxy_client import (
    ProxyAuthError,
    ProxyError,
    ProxyForbiddenError,
    ProxyUnavailableError,
)


@pytest.fixture
def mock_classifier():
    classifier = AsyncMock()
    classifier.classify.return_value = ClassificationResult(
        sender_type=SenderType.PERSON,
        sender_type_raw="PERSON",
        label=EmailLabel.NEEDS_RESPONSE,
        label_raw="NEEDS_RESPONSE",
    )
    classifier.classify_sender.return_value = (SenderType.PERSON, "PERSON", "")
    return classifier


@pytest.fixture
def mock_thread_response():
    """Sample thread with two messages."""
    body1 = "Hey, can we meet tomorrow at 3pm?"
    body2 = "Sure, works for me. See you then!"
    return {
        "id": "thread_001",
        "snippet": "Sure, works for me. See you then!",
        "messages": [
            {
                "id": "msg_001",
                "threadId": "thread_001",
                "internalDate": "1704067200000",
                "labelIds": ["INBOX", "UNREAD"],
                "payload": {
                    "headers": [
                        {"name": "From", "value": "John Doe <john@example.com>"},
                        {"name": "Subject", "value": "Meeting tomorrow"},
                        {"name": "Date", "value": "Mon, 1 Jan 2024 12:00:00 +0000"},
                    ],
                    "body": {
                        "data": base64.urlsafe_b64encode(body1.encode()).decode(),
                    },
                },
            },
            {
                "id": "msg_002",
                "threadId": "thread_001",
                "internalDate": "1704070800000",
                "labelIds": ["INBOX", "UNREAD"],
                "payload": {
                    "headers": [
                        {"name": "From", "value": "Jane Smith <jane@example.com>"},
                        {"name": "Subject", "value": "Re: Meeting tomorrow"},
                        {"name": "Date", "value": "Mon, 1 Jan 2024 13:00:00 +0000"},
                    ],
                    "body": {
                        "data": base64.urlsafe_b64encode(body2.encode()).decode(),
                    },
                },
            },
        ],
    }


async def drive_attribution_cycle(
    threads, proxy, classifier, label_manager, cloud_sem, local_sem,
    tracker, masquerade=None, **kwargs,
):
    """One poll cycle in miniature (decision D5 Rule 2, Wave 2 T8).

    Processes every thread with a shared per-cycle collector, then runs the
    post-gather attribution exactly as run_daemon does — attribute → strike →
    mark. Returns the per-thread results (give-ups return False now: they are
    failures, not handled work).
    """
    failures: list[daemon.CycleFailure] = []
    results = [
        await process_single_thread(
            tid, msg_ids, proxy, classifier, label_manager, cloud_sem, local_sem,
            max_thread_chars=16000, cycle_failures=failures, **kwargs,
        )
        for tid, msg_ids in threads
    ]
    struck_out = daemon.attribute_cycle_failures(
        threads, results, failures, tracker,
        masquerade if masquerade is not None else daemon.MasqueradeTracker(),
    )
    for entry in struck_out:
        await daemon._mark_thread_attempted(
            entry.thread_id, entry.ids_to_mark, tracker, label_manager
        )
    return results


class TestFormatThreadTranscript:
    def test_formats_chronologically(self, mock_thread_response):
        messages = mock_thread_response["messages"]
        transcript = format_thread_transcript(messages, 50000)
        # John's message should come before Jane's
        john_pos = transcript.index("John Doe")
        jane_pos = transcript.index("Jane Smith")
        assert john_pos < jane_pos

    def test_includes_sender_and_date(self, mock_thread_response):
        messages = mock_thread_response["messages"]
        transcript = format_thread_transcript(messages, 50000)
        assert "John Doe <john@example.com>" in transcript
        assert "Mon, 1 Jan 2024 12:00:00 +0000" in transcript

    def test_includes_body(self, mock_thread_response):
        messages = mock_thread_response["messages"]
        transcript = format_thread_transcript(messages, 50000)
        assert "Hey, can we meet tomorrow at 3pm?" in transcript
        assert "Sure, works for me" in transcript

    def test_truncates_oldest_first(self, mock_thread_response):
        messages = mock_thread_response["messages"]
        # Set a very small max to force truncation
        transcript = format_thread_transcript(messages, 100)
        assert "[Earlier messages truncated]" in transcript or len(transcript) <= 100


class TestProcessSingleThread:
    async def test_classifies_thread(
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        cloud_sem,
        local_sem,
        mock_thread_response,
    ):
        mock_proxy.get_thread.return_value = mock_thread_response
        mock_classifier.classify.return_value = ClassificationResult(
            sender_type=SenderType.PERSON,
            sender_type_raw="PERSON",
            label=EmailLabel.NEEDS_RESPONSE,
            label_raw="NEEDS_RESPONSE",
        )

        result = await process_single_thread(
            "thread_001",
            ["msg_001", "msg_002"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
        )

        assert result is True
        mock_proxy.get_thread.assert_called_once_with("thread_001")
        mock_classifier.classify.assert_called_once()
        # Labels applied to ALL messages in thread
        mock_label_manager.apply_classification.assert_called_once()
        call_args = mock_label_manager.apply_classification.call_args
        assert call_args.args[0] == ["msg_001", "msg_002"]  # all message IDs

    async def test_person_thread_returns_false_when_mlx_unreachable(
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        cloud_sem,
        local_sem,
        mock_thread_response,
    ):
        """PERSON threads fail gracefully when local LLM is unreachable."""
        mock_proxy.get_thread.return_value = mock_thread_response
        mock_classifier.classify_sender.return_value = (SenderType.PERSON, "PERSON", "")
        mock_classifier.classify.side_effect = httpx.ConnectError("Connection refused")

        result = await process_single_thread(
            "thread_001",
            ["msg_001", "msg_002"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
        )

        assert result is False
        mock_label_manager.apply_classification.assert_not_called()

    async def test_skips_downgrade(
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        cloud_sem,
        local_sem,
        mock_thread_response,
    ):
        """Thread already at FYI should not be downgraded to LOW_PRIORITY."""
        mock_proxy.get_thread.return_value = mock_thread_response
        mock_classifier.classify.return_value = ClassificationResult(
            sender_type=SenderType.SERVICE,
            sender_type_raw="SERVICE",
            label=EmailLabel.LOW_PRIORITY,
            label_raw="LOW_PRIORITY",
        )
        # Existing priority = FYI (2), new = LOW_PRIORITY (1) -> skip
        mock_label_manager.get_existing_priority.return_value = _get_priority(EmailLabel.FYI)

        result = await process_single_thread(
            "thread_001",
            ["msg_001", "msg_002"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
        )

        assert result is True
        mock_label_manager.apply_classification.assert_not_called()
        mock_label_manager.mark_processed.assert_called_once_with(["msg_001", "msg_002"])

    async def test_already_at_max_priority_marks_processed(
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        cloud_sem,
        local_sem,
        mock_thread_response,
    ):
        """A thread already at max priority is marked processed (not re-fetched forever).

        Without the mark_processed, the thread keeps no agent/processed label and
        re-matches the unprocessed query every poll cycle, burning a get_thread
        round-trip per thread per cycle. It should be skipped from classification but
        still marked processed so it drops out of the query.

        The query surfaced only msg_001, but the fetched thread holds msg_001 +
        msg_002. mark_processed must cover the FULL thread (all_msg_ids), not just the
        query stub — otherwise the unmarked sibling keeps re-matching the query, the
        exact loop this branch exists to break. Passing a strict subset as the stubs
        makes the assertion sensitive to that distinction.
        """
        mock_proxy.get_thread.return_value = mock_thread_response
        # Already at the top priority -> no classification needed.
        mock_label_manager.get_existing_priority.return_value = _get_priority(
            EmailLabel.NEEDS_RESPONSE
        )

        result = await process_single_thread(
            "thread_001",
            ["msg_001"],  # query stub is a strict subset of the fetched thread
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
        )

        assert result is True
        # No (re)classification, but the thread is marked processed so it stops
        # re-matching the unprocessed query.
        mock_classifier.classify_sender.assert_not_called()
        mock_classifier.classify.assert_not_called()
        mock_label_manager.apply_classification.assert_not_called()
        # Full thread, not just the query stub.
        mock_label_manager.mark_processed.assert_called_once_with(["msg_001", "msg_002"])

    async def test_allows_upgrade(
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        cloud_sem,
        local_sem,
        mock_thread_response,
    ):
        """Thread at FYI can be upgraded to NEEDS_RESPONSE."""
        mock_proxy.get_thread.return_value = mock_thread_response
        mock_classifier.classify.return_value = ClassificationResult(
            sender_type=SenderType.PERSON,
            sender_type_raw="PERSON",
            label=EmailLabel.NEEDS_RESPONSE,
            label_raw="NEEDS_RESPONSE",
        )
        # Existing priority = FYI (2), new = NEEDS_RESPONSE (3) -> upgrade
        mock_label_manager.get_existing_priority.return_value = _get_priority(EmailLabel.FYI)

        result = await process_single_thread(
            "thread_001",
            ["msg_001", "msg_002"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
        )

        assert result is True
        mock_label_manager.apply_classification.assert_called_once()

    async def test_error_in_processing_returns_false(
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        cloud_sem,
        local_sem,
    ):
        """Errors during processing don't crash — return False."""
        mock_proxy.get_thread.side_effect = RuntimeError("API error")

        result = await process_single_thread(
            "thread_001",
            ["msg_001"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
        )

        assert result is False

    async def test_gives_up_on_thread_after_repeated_failures(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
    ):
        """A thread that keeps failing for a thread-attributable reason is marked
        agent/attempted after max_failures strikes, breaking the loop.

        Reworked for the post-gather attribution path (decision D5, Wave 2 T8):
        strikes are counted by attribute_cycle_failures, marking happens in the
        poll loop, and give-ups return False from process_single_thread (a
        failure, not handled work)."""
        mock_proxy.get_thread.side_effect = RuntimeError("API error")
        tracker = FailureTracker(max_failures=3)

        results = [
            (await drive_attribution_cycle(
                [("thread_stuck", ["msg_1"])], mock_proxy, mock_classifier,
                mock_label_manager, cloud_sem, local_sem, tracker=tracker,
            ))[0]
            for _ in range(3)
        ]

        # Every failing cycle defers; the third strike hits the threshold and the
        # thread is marked agent/attempted (not agent/processed) so the abandoned
        # thread stays findable.
        assert results == [False, False, False]
        mock_label_manager.mark_attempted.assert_called_once_with(["msg_1"])
        mock_label_manager.mark_processed.assert_not_called()
        # The give-up is recorded so the cycle summary can report it distinctly.
        assert tracker.take_given_up() == ["thread_stuck"]

    async def test_connect_error_does_not_count_toward_give_up(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
    ):
        """An endpoint outage (ConnectError) is provider-shaped (D5) and must
        never strike — even across many singleton cycles."""
        mock_proxy.get_thread.side_effect = httpx.ConnectError("connection refused")
        tracker = FailureTracker(max_failures=2)

        for _ in range(5):
            results = await drive_attribution_cycle(
                [("thread_down", ["msg_1"])], mock_proxy, mock_classifier,
                mock_label_manager, cloud_sem, local_sem, tracker=tracker,
            )
            assert results == [False]

        mock_label_manager.mark_attempted.assert_not_called()
        mock_label_manager.mark_processed.assert_not_called()

    async def test_llm_unavailable_does_not_count_toward_give_up(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response,
    ):
        """An LLM endpoint outage (LLMUnavailableError) is transient and must never give up.

        Covers review finding #1: a connect-timeout / dropped connection from a down
        local MLX server surfaces as LLMUnavailableError, which must be retried, not
        counted toward marking the thread processed.
        """
        mock_proxy.get_thread.return_value = mock_thread_response
        mock_classifier.classify_sender.side_effect = LLMUnavailableError("MLX endpoint down")
        tracker = FailureTracker(max_failures=2)

        for _ in range(5):
            results = await drive_attribution_cycle(
                [("thread_001", ["msg_001"])], mock_proxy, mock_classifier,
                mock_label_manager, cloud_sem, local_sem, tracker=tracker,
            )
            assert results == [False]

        mock_label_manager.mark_attempted.assert_not_called()
        mock_label_manager.mark_processed.assert_not_called()

    async def test_balance_error_never_counts_toward_give_up_or_marks(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response,
    ):
        """An out-of-funds provider is account-wide, not a poison thread: the thread
        must be left fully unprocessed (no agent/attempted, no agent/processed) so
        it is retried after the admin adds funds and restarts."""
        mock_proxy.get_thread.return_value = mock_thread_response
        mock_classifier.classify_sender.side_effect = LLMBalanceError("out of funds")
        tracker = FailureTracker(max_failures=2)

        for _ in range(5):
            results = await drive_attribution_cycle(
                [("thread_broke", ["msg_1"])], mock_proxy, mock_classifier,
                mock_label_manager, cloud_sem, local_sem, tracker=tracker,
            )
            assert results == [False]

        mock_label_manager.mark_attempted.assert_not_called()
        mock_label_manager.mark_processed.assert_not_called()

    async def test_balance_error_trips_the_email_function_halt(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response,
    ):
        """A balance error on the email tiers trips the EMAIL function's halt slot
        (decision D5's scope rule, D19: halts became per-function in Wave 2 T9),
        signalling the poll loop to stop that function — on the THIRD consecutive
        fault, not the first (issue #73, D22). The slot records the cloud client
        as the one to re-probe, since the fault came from the cloud tier."""
        mock_proxy.get_thread.return_value = mock_thread_response
        mock_classifier.classify_sender.side_effect = LLMBalanceError(
            "out of funds", tier="cloud"
        )
        halts = daemon.FunctionHalts()

        async def attempt():
            return await process_single_thread(
                "thread_broke", ["msg_1"], mock_proxy, mock_classifier, mock_label_manager,
                cloud_sem, local_sem, max_thread_chars=16000, halts=halts,
            )

        assert await attempt() is False
        assert await attempt() is False
        assert halts.email.tripped is False
        assert await attempt() is False
        assert halts.email.tripped is True
        assert "out of funds" in halts.email.reason
        assert halts.email.probe_client is mock_classifier.cloud_llm

    async def test_local_tier_balance_error_records_the_local_client_to_probe(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response,
    ):
        """A balance fault carrying tier="local" (a public stand-in on the local
        slot, D4 — eval-only, but the daemon must still probe the right thing)
        records the LOCAL client for the re-probe, not the cloud one."""
        mock_proxy.get_thread.return_value = mock_thread_response
        mock_classifier.classify.side_effect = LLMBalanceError("out of funds", tier="local")
        halts = daemon.FunctionHalts()
        for _ in range(3):
            await process_single_thread(
                "thread_broke", ["msg_1"], mock_proxy, mock_classifier, mock_label_manager,
                cloud_sem, local_sem, max_thread_chars=16000, halts=halts,
            )
        assert halts.email.tripped is True
        assert halts.email.probe_client is mock_classifier.local_llm

    async def test_a_successful_classification_resets_the_balance_count(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response,
    ):
        """Two balance faults, a success, two more: no halt. The count is of
        CONSECUTIVE faults (D22) — a provider that answers in between is not
        out of funds."""
        mock_proxy.get_thread.return_value = mock_thread_response
        good = mock_classifier.classify_sender.return_value
        mock_classifier.classify_sender.side_effect = [
            LLMBalanceError("out of funds"), LLMBalanceError("out of funds"),
            good,
            LLMBalanceError("out of funds"), LLMBalanceError("out of funds"),
        ]
        halts = daemon.FunctionHalts()
        results = []
        for _ in range(5):
            results.append(await process_single_thread(
                "thread_x", ["msg_1"], mock_proxy, mock_classifier, mock_label_manager,
                cloud_sem, local_sem, max_thread_chars=16000, halts=halts,
            ))
        assert results == [False, False, True, False, False]
        assert halts.email.tripped is False

    async def test_tripped_halt_short_circuits_before_any_work(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
    ):
        """Once every enabled function is halted, sibling threads in the same cycle
        must not keep fetching and hammering the known-dead provider. (Here only
        email triage is enabled, so its halt is the whole daemon's.)"""
        halts = daemon.FunctionHalts()
        halts.email.trip("out of funds")

        result = await process_single_thread(
            "thread_next", ["msg_1"], mock_proxy, mock_classifier, mock_label_manager,
            cloud_sem, local_sem, max_thread_chars=16000, halts=halts,
        )

        assert result is False
        mock_proxy.get_thread.assert_not_called()
        mock_classifier.classify_sender.assert_not_called()

    async def test_halt_tripped_mid_fetch_skips_classification(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response,
    ):
        """A thread already past the entry check when a sibling trips the halt must
        still skip its LLM calls (the expensive step, including any retry ladder).
        The post-fetch check is now function-aware and sits below routing (T9)."""
        halts = daemon.FunctionHalts()

        async def fetch_then_sibling_trips(*args, **kwargs):
            halts.email.trip("out of funds")
            return mock_thread_response

        mock_proxy.get_thread.side_effect = fetch_then_sibling_trips

        result = await process_single_thread(
            "thread_next", ["msg_1"], mock_proxy, mock_classifier, mock_label_manager,
            cloud_sem, local_sem, max_thread_chars=16000, halts=halts,
        )

        assert result is False
        mock_classifier.classify_sender.assert_not_called()

    async def test_local_tier_outage_defers_quietly_and_is_counted(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response, caplog,
    ):
        """A LOCAL-tier LLM outage is a routine condition (issue #24): the laptop
        serving MLX is expected to be offline for hours. The per-thread handler must
        log below WARNING and count the deferral for the one-line cycle summary,
        while still deferring the thread (no give-up, no cloud fallback).
        """
        mock_proxy.get_thread.return_value = mock_thread_response
        mock_classifier.classify_sender.side_effect = LLMUnavailableError(
            "MLX endpoint down", tier="local"
        )
        deferrals = []

        with caplog.at_level(logging.DEBUG, logger="email-labeler"):
            result = await process_single_thread(
                "thread_local", ["msg_1"], mock_proxy, mock_classifier, mock_label_manager,
                cloud_sem, local_sem, max_thread_chars=16000, local_deferrals=deferrals,
            )

        assert result is False
        assert deferrals == ["thread_local"]
        outage_logs = [r for r in caplog.records if "unavailable" in r.getMessage().lower()]
        assert outage_logs, "expected the outage to still be logged (at DEBUG)"
        assert all(r.levelno < logging.WARNING for r in outage_logs)
        mock_label_manager.mark_processed.assert_not_called()

    async def test_cloud_or_tierless_outage_still_warns(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response, caplog,
    ):
        """Cloud (and tier-less) LLM outages remain surprising: keep the WARNING
        and don't count them as local deferrals."""
        mock_proxy.get_thread.return_value = mock_thread_response
        mock_classifier.classify_sender.side_effect = LLMUnavailableError("cloud down")
        deferrals = []

        with caplog.at_level(logging.DEBUG, logger="email-labeler"):
            result = await process_single_thread(
                "thread_cloud", ["msg_1"], mock_proxy, mock_classifier, mock_label_manager,
                cloud_sem, local_sem, max_thread_chars=16000, local_deferrals=deferrals,
            )

        assert result is False
        assert deferrals == []
        outage_logs = [r for r in caplog.records if "unavailable" in r.getMessage().lower()]
        assert any(r.levelno == logging.WARNING for r in outage_logs)

    async def test_proxy_unavailable_never_counts_toward_give_up(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
    ):
        """A per-thread ProxyUnavailableError is provider-shaped and never strikes.

        Reversal of issue #26's give-up counting, recorded in decision D5 (Wave 2
        T8): the proxy failing to serve a request — a 5xx, an exhausted 429, a
        dropped connection — is never the thread's blame. No matter how many
        cycles it persists, the thread defers and the backlog is kept. The
        residual #26 worried about (a deterministic per-thread 5xx masquerading
        as an outage) is watched by MasqueradeTracker instead: it escalates
        loudly on the heartbeat, retried forever rather than abandoned — see
        TestFailureAttribution.
        """
        mock_proxy.get_thread.side_effect = ProxyUnavailableError("proxy 500 for one poison thread")
        tracker = FailureTracker(max_failures=2)

        for _ in range(5):
            results = await drive_attribution_cycle(
                [("thread_poison", ["msg_1"])], mock_proxy, mock_classifier,
                mock_label_manager, cloud_sem, local_sem, tracker=tracker,
            )
            assert results == [False]

        mock_label_manager.mark_attempted.assert_not_called()
        mock_label_manager.mark_processed.assert_not_called()
        assert tracker.take_given_up() == []

    async def test_give_up_write_transient_failure_logs_clean_warning(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem, caplog,
    ):
        """A transient proxy outage DURING the give-up marker write is expected, not a
        bug (issue #32): log a retryable warning (no traceback) and retry next cycle —
        distinct from an unexpected/permanent marker-write failure. The give-up is not
        recorded, so the thread stays give-up-eligible until the marker write lands.
        (The marker write moved to the poll loop's post-gather marking step —
        D5, Wave 2 T8 — and this pin follows it there.)
        """
        mock_proxy.get_thread.side_effect = RuntimeError("classification boom")
        mock_label_manager.mark_attempted.side_effect = ProxyUnavailableError("proxy 503 on write")
        tracker = FailureTracker(max_failures=1)  # strike out on the first failure

        with caplog.at_level(logging.WARNING):
            results = await drive_attribution_cycle(
                [("thread_x", ["msg_1"])], mock_proxy, mock_classifier,
                mock_label_manager, cloud_sem, local_sem, tracker=tracker,
            )

        assert results == [False]             # couldn't mark → retry next cycle
        assert tracker.take_given_up() == []  # not recorded as given up
        retry_logs = [r for r in caplog.records if "will retry" in r.getMessage()]
        assert retry_logs, "expected a clean retry warning for the transient marker-write failure"
        # Transient: a clean WARNING, never an ERROR-with-traceback.
        assert all(r.levelno == logging.WARNING and r.exc_info is None for r in retry_logs)

    async def test_give_up_write_unexpected_failure_keeps_traceback(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem, caplog,
    ):
        """An UNEXPECTED (likely permanent) marker-write failure keeps its full traceback
        so it stays diagnosable rather than being swallowed as benign (issue #32).
        (Marker write followed to the post-gather marking step — D5, Wave 2 T8.)"""
        mock_proxy.get_thread.side_effect = RuntimeError("classification boom")
        mock_label_manager.mark_attempted.side_effect = ValueError("unexpected marker bug")
        tracker = FailureTracker(max_failures=1)

        with caplog.at_level(logging.WARNING):
            results = await drive_attribution_cycle(
                [("thread_y", ["msg_1"])], mock_proxy, mock_classifier,
                mock_label_manager, cloud_sem, local_sem, tracker=tracker,
            )

        assert results == [False]
        # A failed marker write must NOT be recorded as a give-up: the thread wasn't
        # labeled, so reporting it abandoned would be misleading and it keeps re-matching.
        assert tracker.take_given_up() == []
        marker_logs = [r for r in caplog.records if "Could not mark" in r.getMessage()]
        assert marker_logs and any(r.levelno >= logging.ERROR and r.exc_info for r in marker_logs)

    async def test_proxy_4xx_is_give_up_eligible(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
    ):
        """A request-specific proxy 4xx (plain ProxyError) stays a strike candidate.

        Reworked to the cycle-attribution path (D5, Wave 2 T8): the transient
        subclass is provider-shaped and defers, but the base ProxyError (e.g. a
        404 for a thread deleted between listing and fetching) is a candidate —
        correlation blames the thread in these singleton cycles, so it is
        bounded by the FailureTracker rather than retried forever.
        """
        mock_proxy.get_thread.side_effect = ProxyError("404 not found")
        tracker = FailureTracker(max_failures=2)

        results = [
            (await drive_attribution_cycle(
                [("thread_gone", ["msg_1"])], mock_proxy, mock_classifier,
                mock_label_manager, cloud_sem, local_sem, tracker=tracker,
            ))[0]
            for _ in range(2)
        ]

        assert results == [False, False]  # give-ups return False now (D5)
        mock_label_manager.mark_attempted.assert_called_once_with(["msg_1"])
        assert tracker.take_given_up() == ["thread_gone"]

    async def test_give_up_marks_all_thread_messages_not_just_query_stubs(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response,
    ):
        """When a fetched thread is given up, ALL its messages are marked processed.

        Covers review finding #3: the query may return a subset of a thread's messages,
        but once get_thread succeeds the full thread is known. Give-up must mark every
        message (so the thread stops re-matching the query) — not just the query stub.
        """
        # Thread has msg_001 + msg_002, but the query only surfaced msg_001.
        mock_proxy.get_thread.return_value = mock_thread_response
        mock_classifier.classify_sender.side_effect = RuntimeError("classification boom")
        tracker = FailureTracker(max_failures=2)

        results = [
            (await drive_attribution_cycle(
                [("thread_001", ["msg_001"])], mock_proxy, mock_classifier,
                mock_label_manager, cloud_sem, local_sem, tracker=tracker,
            ))[0]
            for _ in range(2)
        ]

        assert results == [False, False]  # give-ups return False now (D5, Wave 2 T8)
        mock_label_manager.mark_attempted.assert_called_once_with(["msg_001", "msg_002"])

    async def test_already_at_max_priority_give_up_marks_all_thread_messages(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response,
    ):
        """The max-priority skip writes a marker, so a failed write there can strike out.

        The branch calls mark_processed inside the try; a candidate-class failure
        there is collected for attribution. ids_to_mark must already be the full
        thread (all_msg_ids), not the query stub — otherwise the
        abandoned-but-findable marker lands on a subset and the unmarked sibling
        keeps re-surfacing the thread, the very loop the give-up mechanism exists
        to break. The query surfaced only msg_001 while the thread holds msg_001 +
        msg_002.

        Reworked for D5 (Wave 2 T8): the old pin drove the write fault with
        ProxyUnavailableError, which is provider-shaped now and never strikes —
        a request-specific ProxyError (4xx) carries the all-thread-ids assertion
        instead.
        """
        mock_proxy.get_thread.return_value = mock_thread_response
        mock_label_manager.get_existing_priority.return_value = _get_priority(
            EmailLabel.NEEDS_RESPONSE
        )
        # The mark_processed write fails deterministically with a request-specific 4xx.
        mock_label_manager.mark_processed.side_effect = ProxyError("400 on write")
        tracker = FailureTracker(max_failures=2)

        results = [
            (await drive_attribution_cycle(
                [("thread_001", ["msg_001"])], mock_proxy, mock_classifier,
                mock_label_manager, cloud_sem, local_sem, tracker=tracker,
            ))[0]
            for _ in range(2)
        ]

        assert results == [False, False]  # give-ups return False now (D5)
        # No classification was ever attempted (it's the max-priority skip path)...
        mock_classifier.classify_sender.assert_not_called()
        # ...and give-up marks the FULL thread, not just the query stub.
        mock_label_manager.mark_attempted.assert_called_once_with(["msg_001", "msg_002"])

    async def test_rejected_write_defers_without_strike_or_traceback(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response, caplog,
    ):
        """A proxy 403 on a gated write is a human answer, not a failure (D6, #28).

        The operator saying "not now" must never strike toward give-up: no matter
        how many cycles the rejection persists, the thread is re-offered next cycle
        with one clean INFO line — no traceback, no agent/attempted.
        """
        mock_proxy.get_thread.return_value = mock_thread_response
        mock_label_manager.apply_classification.side_effect = ProxyForbiddenError(
            "modify_message blocked by proxy"
        )
        tracker = FailureTracker(max_failures=2)

        with caplog.at_level(logging.INFO):
            results = [
                (await drive_attribution_cycle(
                    [("thread_001", ["msg_001"])], mock_proxy, mock_classifier,
                    mock_label_manager, cloud_sem, local_sem, tracker=tracker,
                ))[0]
                for _ in range(3)  # past max_failures: rejections never accumulate
            ]

        assert results == [False, False, False]  # re-offered every cycle, never "handled"
        mock_label_manager.mark_attempted.assert_not_called()  # a rejection can never end in give-up
        assert tracker.take_given_up() == []
        # One clean line per rejection — never an exception traceback.
        assert all(r.exc_info is None for r in caplog.records)
        reoffer_logs = [r for r in caplog.records if "re-offering next cycle" in r.getMessage()]
        assert reoffer_logs and all(r.levelno == logging.INFO for r in reoffer_logs)

    async def test_rejected_marker_write_re_offers(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem, caplog,
    ):
        """A proxy 403 on the agent/attempted marker write is a human answer too (D6).

        The rejection blocks the marker — one clean line, no traceback — and the
        thread stays give-up-eligible, so the marker write is re-offered next cycle.
        (The rejection never *causes* the give-up: the strikes that reached the
        threshold came from elsewhere — here, a RuntimeError during classification.
        Marker write followed to the post-gather marking step — D5, Wave 2 T8.)
        """
        mock_proxy.get_thread.side_effect = RuntimeError("classification boom")
        mock_label_manager.mark_attempted.side_effect = ProxyForbiddenError(
            "marker write rejected"
        )
        tracker = FailureTracker(max_failures=1)  # threshold on the first failure

        with caplog.at_level(logging.INFO):
            results = await drive_attribution_cycle(
                [("thread_x", ["msg_1"])], mock_proxy, mock_classifier,
                mock_label_manager, cloud_sem, local_sem, tracker=tracker,
            )

        assert results == [False]             # marker blocked → not handled
        assert tracker.take_given_up() == []  # give-up not recorded: the marker never landed
        # A clean line for the rejection, no traceback anywhere.
        assert all(r.exc_info is None for r in caplog.records)
        reoffer_logs = [r for r in caplog.records if "re-offering next cycle" in r.getMessage()]
        assert reoffer_logs and all(r.levelno == logging.INFO for r in reoffer_logs)

    async def test_get_thread_is_bounded_by_fetch_sem(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response,
    ):
        """get_thread runs under fetch_sem, so proxy fetches can't fan out unbounded.

        Covers review finding #5: a large max_emails_per_cycle would otherwise burst
        one simultaneous get_thread per thread (the LLM semaphores gate only the
        classify calls, not the fetch).
        """
        mock_proxy.get_thread.return_value = mock_thread_response
        exhausted = asyncio.Semaphore(0)  # no permits available

        task = asyncio.create_task(process_single_thread(
            "thread_001", ["msg_001"], mock_proxy, mock_classifier, mock_label_manager,
            cloud_sem, local_sem, max_thread_chars=16000, fetch_sem=exhausted,
        ))
        await asyncio.sleep(0.05)
        # Blocked acquiring the fetch semaphore — get_thread must not have run.
        mock_proxy.get_thread.assert_not_called()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_processes_normally_with_available_fetch_sem(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response,
    ):
        """With a permitting fetch_sem, the thread is fetched and classified normally."""
        mock_proxy.get_thread.return_value = mock_thread_response

        result = await process_single_thread(
            "thread_001", ["msg_001"], mock_proxy, mock_classifier, mock_label_manager,
            cloud_sem, local_sem, max_thread_chars=16000, fetch_sem=asyncio.Semaphore(2),
        )

        assert result is True
        mock_proxy.get_thread.assert_called_once()

    async def test_label_application_is_bounded_by_write_sem(
        self, mock_proxy, mock_classifier, cloud_sem, local_sem,
        mock_thread_response,
    ):
        """Label-application writes run under write_sem, so they can't fan out unbounded.

        Covers issue #17: the cloud/local/fetch semaphores gate reads + classify, but
        the label-application phase (modify_message via apply_classification etc.)
        previously ran with no bound, so a large max_emails_per_cycle could burst many
        concurrent writes at the api-proxy / Gmail.

        Reworked for issue #33 (Wave 2 T5): the semaphore moved from the daemon's
        per-method acquisition sites into LabelManager itself, which acquires one
        slot per modify_message write — so this drives a real LabelManager holding
        an exhausted semaphore. Classification must still complete before the
        write blocks: the semaphore gates only the writes.
        """
        mock_proxy.get_thread.return_value = mock_thread_response
        exhausted = asyncio.Semaphore(0)  # no write permits available
        label_manager = LabelManager(
            proxy_client=mock_proxy,
            config={"labels": {
                "needs_response": "agent/needs-response",
                "fyi": "agent/fyi",
                "low_priority": "agent/low-priority",
                "processed": "agent/processed",
                "attempted": "agent/attempted",
                "personal": "agent/personal",
                "non_personal": "agent/non-personal",
                "actions": {
                    "needs_response": "inbox", "fyi": "inbox", "low_priority": "archive",
                },
            }},
            write_sem=exhausted,
        )
        label_manager.label_ids = {
            "agent/needs-response": "Label_1",
            "agent/processed": "Label_4",
            "agent/personal": "Label_5",
        }

        task = asyncio.create_task(process_single_thread(
            "thread_001", ["msg_001"], mock_proxy, mock_classifier, label_manager,
            cloud_sem, local_sem, max_thread_chars=16000,
            fetch_sem=asyncio.Semaphore(2),
        ))
        await asyncio.sleep(0.05)
        # Fetch + classify ran, but the first label write is blocked acquiring the
        # LabelManager-owned semaphore — nothing written yet.
        mock_proxy.get_thread.assert_called_once()
        mock_classifier.classify.assert_called_once()
        mock_proxy.modify_message.assert_not_called()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_service_thread_classified_via_cloud(
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        cloud_sem,
        local_sem,
        mock_thread_response,
    ):
        """Service threads are classified via cloud LLM regardless of local LLM state."""
        mock_proxy.get_thread.return_value = mock_thread_response
        mock_classifier.classify_sender.return_value = (SenderType.SERVICE, "SERVICE", "")
        mock_classifier.classify.return_value = ClassificationResult(
            sender_type=SenderType.SERVICE,
            sender_type_raw="SERVICE",
            label=EmailLabel.LOW_PRIORITY,
            label_raw="LOW_PRIORITY",
        )

        result = await process_single_thread(
            "thread_001",
            ["msg_001", "msg_002"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
        )

        assert result is True
        mock_classifier.classify.assert_called_once()
        mock_label_manager.apply_classification.assert_called_once()


class TestFailureAttribution:
    """D5 Rule 2 (Wave 2 T8): strikes are decided post-gather by cycle-level
    correlation. A candidate failure (Timeout/RuntimeError/unexpected Exception)
    counts only when its signature is unique among the cycle's candidate
    failures AND — in a multi-thread cycle — at least one sibling was handled
    successfully. Provider-shaped failures never strike; the single-thread
    masquerade (provider-shaped errors while siblings succeed) escalates on the
    status heartbeat instead of being abandoned."""

    async def test_same_signature_failures_in_one_cycle_count_no_strikes(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response, caplog,
    ):
        """Adjudicated edge (D5): N same-signature threads shield each other.
        Two threads failing identically look like one shared cause (a prompt bug,
        our own code) — no strikes, one shared-cause ERROR, backlog kept.

        A third thread *succeeds* on purpose: that makes thread_blame true, so
        the signature-uniqueness rule is the only thing standing between these
        failures and a strike. Without the succeeding sibling this test would
        merely re-pin the zero-success edge below."""
        def route(tid):
            if tid == "t_good":
                return mock_thread_response
            raise RuntimeError("same boom")

        mock_proxy.get_thread.side_effect = route
        tracker = FailureTracker(max_failures=1)  # a wrongly-counted strike would mark

        with caplog.at_level(logging.ERROR, logger="email-labeler"):
            results = await drive_attribution_cycle(
                [("t_a", ["m1"]), ("t_b", ["m2"]), ("t_good", ["m3"])],
                mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
                tracker=tracker,
            )

        assert results == [False, False, True]
        mock_label_manager.mark_attempted.assert_not_called()
        assert tracker.take_given_up() == []
        shared = [r for r in caplog.records if "shared cause" in r.getMessage()]
        assert len(shared) == 1
        assert shared[0].levelno == logging.ERROR

    async def test_unique_signature_failure_with_succeeding_siblings_strikes(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response,
    ):
        """One thread failing uniquely while a sibling succeeds is the thread's
        own fault: bounded strikes, then agent/attempted (D5)."""
        def route(tid):
            if tid == "t_bad":
                raise ValueError("poison thread")
            return mock_thread_response

        mock_proxy.get_thread.side_effect = route
        tracker = FailureTracker(max_failures=2)

        for _ in range(2):
            results = await drive_attribution_cycle(
                [("t_bad", ["m1"]), ("t_good", ["m2"])],
                mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
                tracker=tracker,
            )

        # Give-ups return False (a failure, not handled work); the sibling
        # keeps classifying normally.
        assert results == [False, True]
        mock_label_manager.mark_attempted.assert_called_once_with(["m1"])
        assert tracker.take_given_up() == ["t_bad"]

    async def test_singleton_cycle_candidate_failure_strikes(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
    ):
        """Adjudicated edge (D5): a lone thread's candidate failure has no
        siblings to correlate against — bounded strikes to a findable
        agent/attempted is the honest fallback, and the poison-thread case is
        typically a singleton (everything else processed away)."""
        mock_proxy.get_thread.side_effect = RuntimeError("boom")
        tracker = FailureTracker(max_failures=2)

        results = [
            (await drive_attribution_cycle(
                [("t_solo", ["m1"])],
                mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
                tracker=tracker,
            ))[0]
            for _ in range(2)
        ]

        assert results == [False, False]
        mock_label_manager.mark_attempted.assert_called_once_with(["m1"])
        assert tracker.take_given_up() == ["t_solo"]

    async def test_zero_success_cycle_counts_no_strikes(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem, caplog,
    ):
        """Adjudicated edge (D5): two different signatures with nothing
        succeeding is still consistent with one shared cause surfacing through
        two code paths — Rule 2 conditions thread-blame on siblings
        *succeeding*. No strikes, one shared-cause ERROR, backlog kept."""
        def route(tid):
            if tid == "t_a":
                raise RuntimeError("boom a")
            raise ValueError("boom b")

        mock_proxy.get_thread.side_effect = route
        tracker = FailureTracker(max_failures=1)

        with caplog.at_level(logging.ERROR, logger="email-labeler"):
            results = await drive_attribution_cycle(
                [("t_a", ["m1"]), ("t_b", ["m2"])],
                mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
                tracker=tracker,
            )

        assert results == [False, False]
        mock_label_manager.mark_attempted.assert_not_called()
        assert tracker.take_given_up() == []
        assert any("shared cause" in r.getMessage() for r in caplog.records)

    async def test_deferral_only_sibling_does_not_shield_a_poisoned_thread(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response,
    ):
        """Correlation weighs only the threads that ATTEMPTED work (D5 Rule 2).
        A sibling that merely DEFERRED — here the local tier is offline, so the
        person thread defers quietly with no CycleFailure; equally a
        halt-deferred thread, a NEWSLETTER_ONLY skip, a 403-rejected write or an
        assessment-sink fault — tried nothing and committed nothing, so it is
        evidence about neither blame nor innocence.

        Counting it as a sibling made this cycle look
        multi-thread-and-zero-success every time, so the genuinely poisoned
        thread never struck and never converged to a findable agent/attempted —
        silently voiding D5 Rule 1's "set aside findably" guarantee. The
        newsletter-halted direction makes the shielding permanent: the deferring
        thread is re-fetched and re-deferred every cycle until restart."""
        def route(tid):
            if tid == "t_poison":
                raise ValueError("poison thread")
            return mock_thread_response

        mock_proxy.get_thread.side_effect = route

        async def sender_route(metadata):
            raise LLMUnavailableError("MLX endpoint down", tier="local")

        mock_classifier.classify_sender.side_effect = sender_route
        tracker = FailureTracker(max_failures=2)

        for _ in range(2):
            results = await drive_attribution_cycle(
                [("t_poison", ["m1"]), ("t_deferred", ["m2"])],
                mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
                tracker=tracker,
            )

        assert results == [False, False]
        mock_label_manager.mark_attempted.assert_called_once_with(["m1"])
        assert tracker.take_given_up() == ["t_poison"]

    async def test_deferrals_do_not_turn_a_zero_success_cycle_into_a_singleton(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response, caplog,
    ):
        """Companion to the test above: dropping deferral-only threads from the
        denominator must not over-correct. Two threads genuinely ATTEMPTED and
        failed with different signatures, and a third only deferred — the two
        attempting threads still correlate as one shared cause under the
        adjudicated zero-success edge (D5), so nobody strikes. max_failures=1,
        so a wrongly-counted strike would mark immediately."""
        def route(tid):
            if tid == "t_a":
                raise RuntimeError("boom a")
            if tid == "t_b":
                raise ValueError("boom b")
            return mock_thread_response

        mock_proxy.get_thread.side_effect = route

        async def sender_route(metadata):
            raise LLMUnavailableError("MLX endpoint down", tier="local")

        mock_classifier.classify_sender.side_effect = sender_route
        tracker = FailureTracker(max_failures=1)

        with caplog.at_level(logging.ERROR, logger="email-labeler"):
            results = await drive_attribution_cycle(
                [("t_a", ["m1"]), ("t_b", ["m2"]), ("t_deferred", ["m3"])],
                mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
                tracker=tracker,
            )

        assert results == [False, False, False]
        mock_label_manager.mark_attempted.assert_not_called()
        assert tracker.take_given_up() == []
        assert any("shared cause" in r.getMessage() for r in caplog.records)

    async def test_timeout_is_a_strike_candidate_and_converges_to_attempted(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response,
    ):
        """The TimeoutError arm records a CANDIDATE CycleFailure (D5): a request
        too slow to serve — a transcript too large to prefill within the timeout —
        is the thread's own problem when siblings are fine, so it must strike and
        converge to a findable agent/attempted rather than be retried forever.
        (Connect/pool timeouts arrive as LLMUnavailableError and are
        provider-shaped; this arm is the request-specific one.) Nothing else in
        the suite drives a daemon TimeoutError, so the arm's collector append is
        pinned here."""
        def route(tid):
            if tid == "t_slow":
                raise TimeoutError("prefill exceeded the request timeout")
            return mock_thread_response

        mock_proxy.get_thread.side_effect = route
        tracker = FailureTracker(max_failures=2)

        for _ in range(2):
            results = await drive_attribution_cycle(
                [("t_slow", ["m1"]), ("t_good", ["m2"])],
                mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
                tracker=tracker,
            )

        assert results == [False, True]
        mock_label_manager.mark_attempted.assert_called_once_with(["m1"])
        assert tracker.take_given_up() == ["t_slow"]

    async def test_successful_mark_clears_the_count_so_the_thread_is_not_re_marked(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response,
    ):
        """A landed agent/attempted marker clears the thread's count (D5).
        Give-ups return False, so summarize_cycle's success-clear never runs for
        them — _mark_thread_attempted must clear beside the successful mark. Left
        uncleared, the count sits at the threshold forever and every later strike
        re-marks the same thread, re-spamming the give-up ERROR and the write.
        The third cycle here is the one that exposes it: it strikes again, and
        must NOT re-mark."""
        def route(tid):
            if tid == "t_bad":
                raise ValueError("poison thread")
            return mock_thread_response

        mock_proxy.get_thread.side_effect = route
        tracker = FailureTracker(max_failures=2)

        for _ in range(3):
            await drive_attribution_cycle(
                [("t_bad", ["m1"]), ("t_good", ["m2"])],
                mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
                tracker=tracker,
            )

        mock_label_manager.mark_attempted.assert_called_once_with(["m1"])
        assert tracker.take_given_up() == ["t_bad"]
        assert tracker.should_give_up("t_bad") is False  # count restarted from zero

    async def test_marking_eligibility_comes_from_this_cycles_strikes_only(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response,
    ):
        """Marking derives from THIS cycle's strikes, never from raw tracker
        counts (D5). A marker write that fails transiently deliberately leaves the
        count AT the threshold so the thread's next strike re-offers the write —
        but a stale at-threshold count must not mark a thread that has since
        stopped being blamed.

        Cycle 1: t_bad fails uniquely beside a success, strikes to the threshold,
        and the marker write fails (proxy down) — count stays at 1. Cycle 2: the
        very same failure now has a same-signature twin, so correlation says
        shared cause and NOBODY strikes; the stale count must not mark t_bad."""
        def route(tid):
            if tid in ("t_bad", "t_twin"):
                raise ValueError("boom")
            return mock_thread_response

        mock_proxy.get_thread.side_effect = route
        mock_label_manager.mark_attempted.side_effect = ProxyUnavailableError("proxy down")
        tracker = FailureTracker(max_failures=1)

        await drive_attribution_cycle(
            [("t_bad", ["m1"]), ("t_good", ["m2"])],
            mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
            tracker=tracker,
        )
        assert mock_label_manager.mark_attempted.call_count == 1
        assert tracker.should_give_up("t_bad") is True  # left at the threshold

        await drive_attribution_cycle(
            [("t_bad", ["m1"]), ("t_twin", ["m3"]), ("t_good", ["m2"])],
            mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
            tracker=tracker,
        )

        # No strike this cycle → no marking, however high the stale count is.
        assert mock_label_manager.mark_attempted.call_count == 1
        assert tracker.take_given_up() == []

    async def test_masquerade_suspect_escalates_on_heartbeat_and_is_never_abandoned(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response,
    ):
        """The single-thread masquerade (D5; issue #26's poison-thread
        scenario): provider-shaped failures on one thread while siblings
        succeed. Never marked, retried forever; at max_failures qualifying
        cycles it becomes a suspect and the distinct ERROR line is emitted,
        repeated at most once per status_interval while the suspect persists."""
        def route(tid):
            if tid == "t_masq":
                raise ProxyUnavailableError("deterministic 500 for this thread")
            return mock_thread_response

        mock_proxy.get_thread.side_effect = route
        tracker = FailureTracker(max_failures=2)
        masq = daemon.MasqueradeTracker(max_failures=2)

        for _ in range(3):
            results = await drive_attribution_cycle(
                [("t_masq", ["m1"]), ("t_good", ["m2"])],
                mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
                tracker=tracker, masquerade=masq,
            )
            assert results == [False, True]

        # Never abandoned — no marker, no give-up, no matter how many cycles.
        mock_label_manager.mark_attempted.assert_not_called()
        assert tracker.take_given_up() == []
        # Suspect at the threshold: the distinct ERROR is emitted...
        line = masq.escalation_line(now=1000.0, status_interval=900)
        assert line is not None
        assert "t_masq" in line
        # ...throttled inside the interval...
        assert masq.escalation_line(now=1500.0, status_interval=900) is None
        # ...and repeated on the next heartbeat while the suspect persists.
        line2 = masq.escalation_line(now=1900.0, status_interval=900)
        assert line2 is not None
        assert "t_masq" in line2

    async def test_local_tier_unavailability_never_counts_as_masquerade(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response,
    ):
        """Variant (D5): local-tier LLMUnavailableError is excluded from
        masquerade tracking entirely — the deliberately-offline MLX laptop makes
        "person threads defer while service siblings succeed" the ROUTINE local
        state (issue #24), and tracking it would false-alarm every night the
        laptop is closed."""
        mock_proxy.get_thread.return_value = mock_thread_response

        async def sender_route(metadata):
            if metadata.thread_id == "t_local":
                raise LLMUnavailableError("MLX endpoint down", tier="local")
            return (SenderType.SERVICE, "SERVICE", "")

        mock_classifier.classify_sender.side_effect = sender_route
        tracker = FailureTracker(max_failures=2)
        masq = daemon.MasqueradeTracker(max_failures=1)

        for _ in range(3):
            results = await drive_attribution_cycle(
                [("t_local", ["m1"]), ("t_good", ["m2"])],
                mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
                tracker=tracker, masquerade=masq,
            )
            assert results == [False, True]

        assert masq.suspects() == {}
        assert masq.escalation_line(now=1000.0, status_interval=900) is None
        mock_label_manager.mark_attempted.assert_not_called()

    async def test_masquerade_not_incremented_in_singleton_cycles(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
    ):
        """Variant (D5): a singleton cycle carries no correlation evidence
        either way — a genuine short provider outage with one pending thread
        must never false-alarm (the per-thread WARNING remains its
        visibility)."""
        mock_proxy.get_thread.side_effect = ProxyUnavailableError("proxy down")
        tracker = FailureTracker(max_failures=2)
        masq = daemon.MasqueradeTracker(max_failures=1)

        for _ in range(3):
            results = await drive_attribution_cycle(
                [("t_solo", ["m1"])],
                mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
                tracker=tracker, masquerade=masq,
            )
            assert results == [False]

        assert masq.suspects() == {}
        assert masq.escalation_line(now=1000.0, status_interval=900) is None
        mock_label_manager.mark_attempted.assert_not_called()

    async def test_masquerade_escalation_wired_into_poll_loop_and_throttled(
        self, monkeypatch, tmp_path, caplog,
    ):
        """Poll-loop wiring: once a suspect exists the distinct ERROR is
        emitted, then throttled to the status heartbeat — not repeated every
        cycle (the eight near-instant cycles here fit inside one
        status_interval, so exactly one ERROR).

        MAX_FAILURES is cleared for the same reason as the marking test below:
        the cycle count has to outrun config.toml's shipped escalation
        threshold, which the knob (T13) would otherwise let the environment
        move."""
        monkeypatch.delenv("MAX_FAILURES", raising=False)

        async def masquerade_process(tid, msg_ids, *args, **kwargs):
            if tid == "masq":
                kwargs["cycle_failures"].append(daemon.CycleFailure(
                    thread_id=tid, ids_to_mark=list(msg_ids),
                    signature="ProxyUnavailableError", provider_shaped=True,
                ))
                return False
            return True

        poll = {"messages": [
            {"id": "m1", "threadId": "masq"}, {"id": "m2", "threadId": "good"},
        ]}
        with caplog.at_level(logging.ERROR, logger="email-labeler"):
            await run_poll_cycles(
                monkeypatch, tmp_path, [poll] * 8, process_mock=masquerade_process,
            )

        escalations = [r for r in caplog.records if "provider-shaped" in r.getMessage()]
        assert len(escalations) == 1
        assert "masq" in escalations[0].getMessage()
        assert "retrying forever" in escalations[0].getMessage()

    async def test_struck_out_thread_is_marked_attempted_by_the_poll_loop(
        self, monkeypatch, tmp_path, caplog,
    ):
        """Poll-loop wiring for the marking step (D5): since T8 the only
        production code that abandons a thread is run_daemon's post-gather loop
        over attribute_cycle_failures' struck-out entries. Drive the real
        run_daemon so the convergence property — a blamed thread does reach
        agent/attempted, and the cycle summary says so — is pinned on the
        production path, not only through the tests' miniature-cycle helper.

        The five cycles driven here are config.toml's shipped max_failures, so
        MAX_FAILURES is cleared: since T13 it is an operator knob (D5), and an
        exported override would move the threshold out from under the count."""
        monkeypatch.delenv("MAX_FAILURES", raising=False)

        async def poison_process(tid, msg_ids, *args, **kwargs):
            if tid == "poison":
                kwargs["cycle_failures"].append(daemon.CycleFailure(
                    thread_id=tid, ids_to_mark=list(msg_ids), signature="ValueError",
                ))
                return False
            return True  # the sibling succeeds: correlation blames the thread

        label_manager = MagicMock()
        label_manager.mark_attempted = AsyncMock()
        poll = {"messages": [
            {"id": "m1", "threadId": "poison"}, {"id": "m2", "threadId": "good"},
        ]}
        with caplog.at_level(logging.INFO, logger="email-labeler"):
            await run_poll_cycles(
                monkeypatch, tmp_path, [poll] * 5,
                process_mock=poison_process, label_manager=label_manager,
            )

        label_manager.mark_attempted.assert_awaited_once_with(["m1"])
        assert any(
            "abandoned after repeated failures" in r.getMessage() and "poison" in r.getMessage()
            for r in caplog.records
        )

    async def test_keyword_free_reply_commits_nothing_and_strikes_as_candidate(
        self, mock_proxy, mock_label_manager, cloud_sem, local_sem, mock_thread_response,
    ):
        """D5 Rule 1 (Wave 2 T10): a Stage-2 reply carrying no label keyword is
        an unusable answer, not a LOW_PRIORITY → archive outcome. Driven through
        a REAL EmailClassifier (the mock classifier fixture would bypass the
        parser): nothing is committed — no labels, no processed marker — and the
        failure is a strike candidate, so a thread whose model keeps babbling
        converges to a findable agent/attempted instead of being archived."""
        from classifier import EmailClassifier

        mock_proxy.get_thread.return_value = mock_thread_response
        cloud_llm, local_llm = AsyncMock(), AsyncMock()
        cloud_llm.complete.return_value = ("PERSON", "")  # Stage 1 → local tier
        local_llm.complete.return_value = ("Hmm, hard to say either way.", "")
        classifier = EmailClassifier(
            cloud_llm=cloud_llm,
            local_llm=local_llm,
            config={
                "prompts": {
                    "sender_classification": {
                        "system": "s", "user_template": "{sender}{subject}{snippet}",
                    },
                    "email_classification": {
                        "preamble": "p", "postamble": "q",
                        "user_template": "{sender}{subject}{body}",
                        "categories": {"NEEDS_RESPONSE": "a", "FYI": "b", "LOW_PRIORITY": "c"},
                    },
                }
            },
        )
        failures: list[daemon.CycleFailure] = []

        result = await process_single_thread(
            "thread_001", ["msg_001", "msg_002"], mock_proxy, classifier,
            mock_label_manager, cloud_sem, local_sem, max_thread_chars=50000,
            cycle_failures=failures,
        )

        assert result is False
        mock_label_manager.apply_classification.assert_not_called()
        mock_label_manager.mark_processed.assert_not_called()
        assert [f.signature for f in failures] == ["LLMContentError"]
        assert failures[0].provider_shaped is False

        # Strike candidate under the cycle attribution (D5 Rule 2): a singleton
        # cycle blames the thread, so this failure counts toward give-up.
        tracker = FailureTracker(max_failures=1)
        struck_out = daemon.attribute_cycle_failures(
            [("thread_001", ["msg_001", "msg_002"])], [result], failures, tracker,
            daemon.MasqueradeTracker(),
        )
        assert [entry.thread_id for entry in struck_out] == ["thread_001"]


class TestFailureTracker:
    def test_gives_up_only_at_threshold(self):
        t = FailureTracker(max_failures=3)
        assert t.should_give_up("x") is False
        t.record_failure("x")
        assert t.should_give_up("x") is False
        t.record_failure("x")
        assert t.should_give_up("x") is False
        t.record_failure("x")
        assert t.should_give_up("x") is True

    def test_clear_resets_count(self):
        t = FailureTracker(max_failures=2)
        t.record_failure("x")
        t.record_failure("x")
        assert t.should_give_up("x") is True
        t.clear("x")
        assert t.should_give_up("x") is False

    def test_threads_tracked_independently(self):
        t = FailureTracker(max_failures=2)
        t.record_failure("a")
        t.record_failure("a")
        t.record_failure("b")
        assert t.should_give_up("a") is True
        assert t.should_give_up("b") is False

    def test_records_and_takes_give_ups(self):
        t = FailureTracker(max_failures=1)
        assert t.take_given_up() == []
        t.record_give_up("a")
        t.record_give_up("b")
        assert t.take_given_up() == ["a", "b"]
        assert t.take_given_up() == []  # draining resets the per-cycle list

    def test_prune_evicts_counts_for_absent_threads(self):
        # A thread that fails a few times then vanishes from the query must not
        # leak its count forever (review finding #7).
        t = FailureTracker(max_failures=2)
        t.record_failure("gone")
        t.record_failure("gone")
        assert t.should_give_up("gone") is True
        t.prune({"still_here"})
        assert t.should_give_up("gone") is False  # evicted

    def test_prune_keeps_active_threads(self):
        t = FailureTracker(max_failures=2)
        t.record_failure("active")
        t.record_failure("active")
        t.prune({"active"})
        assert t.should_give_up("active") is True  # still counted toward give-up


class TestMasqueradeTracker:
    """Mirrors TestFailureTracker for the masquerade counter (D5): the
    bookkeeping attribute_cycle_failures performs on it — success clears, the
    per-cycle prune, and the exactly-one-provider-shaped-thread-plus-a-success
    increment condition — plus escalation_line's throttle."""

    @staticmethod
    def _attribute(thread_items, results, failures, masq):
        """Run the real attribution step with a throwaway FailureTracker whose
        threshold is out of reach, so only masquerade bookkeeping is in play."""
        return daemon.attribute_cycle_failures(
            thread_items, results, failures, FailureTracker(max_failures=99), masq,
        )

    def test_success_clears_the_threads_masquerade_count(self):
        # A thread that finally processes is no longer a masquerade suspect —
        # without the success-clear its count only ever grows, so one bad
        # afternoon leaves a permanent false suspect escalating on every
        # heartbeat for the rest of the session.
        masq = daemon.MasqueradeTracker(max_failures=1)
        masq.record_masquerade("t_recovered")
        assert masq.suspects() == {"t_recovered": 1}

        self._attribute(
            [("t_recovered", ["m1"]), ("t_good", ["m2"])], [True, True], [], masq,
        )

        assert masq.suspects() == {}

    def test_prune_evicts_counts_for_threads_gone_from_the_query(self):
        # Mirrors FailureTracker.prune (and its wiring in summarize_cycle): a
        # thread that accrues masquerade cycles and then leaves the query — read,
        # archived, relabeled externally — must not leak its count for the
        # daemon's lifetime, nor keep escalating as a suspect that no longer
        # exists.
        masq = daemon.MasqueradeTracker(max_failures=1)
        masq.record_masquerade("t_gone")
        assert masq.suspects() == {"t_gone": 1}

        self._attribute([("t_still_here", ["m1"])], [True], [], masq)

        assert masq.suspects() == {}

    def test_two_provider_shaped_threads_are_an_outage_not_a_masquerade(self):
        # The masquerade is a SINGLE thread drawing provider-shaped errors while
        # siblings succeed (issue #26's deterministic per-thread 5xx). Two threads
        # failing that way is a provider/proxy problem touching more than one
        # thread — no thread is singled out, so nobody's counter moves, even
        # though a third thread succeeded.
        masq = daemon.MasqueradeTracker(max_failures=1)
        failures = [
            daemon.CycleFailure("t_a", ["m1"], "ProxyUnavailableError", provider_shaped=True),
            daemon.CycleFailure("t_b", ["m2"], "ProxyUnavailableError", provider_shaped=True),
        ]

        self._attribute(
            [("t_a", ["m1"]), ("t_b", ["m2"]), ("t_good", ["m3"])],
            [False, False, True], failures, masq,
        )

        assert masq.suspects() == {}
        assert masq.escalation_line(now=1000.0, status_interval=900) is None

    def test_zero_success_multi_thread_cycle_never_increments(self):
        # The mirror of the adjudicated singleton edge (D5): one provider-shaped
        # failure in a cycle where nothing succeeded is exactly what a genuine
        # provider outage looks like. The counter moves only on POSITIVE evidence
        # (a sibling that actually got work done), so a multi-thread cycle with no
        # successes must leave it alone.
        masq = daemon.MasqueradeTracker(max_failures=1)
        failures = [
            daemon.CycleFailure("t_masq", ["m1"], "ProxyUnavailableError", provider_shaped=True),
            daemon.CycleFailure("t_other", ["m2"], "RuntimeError"),
        ]

        self._attribute(
            [("t_masq", ["m1"]), ("t_other", ["m2"])], [False, False], failures, masq,
        )

        assert masq.suspects() == {}
        assert masq.escalation_line(now=1000.0, status_interval=900) is None

    def test_throttle_resets_once_no_suspect_remains(self):
        # The throttle is per-suspect-stretch, not a global rate limit: once the
        # suspects clear, a NEW suspect appearing inside the same status_interval
        # must escalate immediately rather than be silenced by the previous
        # stretch's timestamp.
        masq = daemon.MasqueradeTracker(max_failures=1)
        masq.record_masquerade("t_a")
        assert masq.escalation_line(now=1000.0, status_interval=900) is not None
        # Throttled while that suspect persists.
        assert masq.escalation_line(now=1100.0, status_interval=900) is None

        masq.clear("t_a")
        assert masq.escalation_line(now=1200.0, status_interval=900) is None  # quiet

        masq.record_masquerade("t_b")
        line = masq.escalation_line(now=1300.0, status_interval=900)
        assert line is not None
        assert "t_b" in line


class TestResultCache:
    """Mirrors TestFailureTracker/TestMasqueradeTracker for the session result
    cache (issue #29, Wave 2 T6): the memory hygiene its class docstring
    promises — cleared on a successful label write, pruned each cycle — which
    the process_single_thread-level reuse tests never observe (they only ever
    drive cycles whose write is still failing)."""

    def test_clear_drops_the_entry(self):
        # Called after a successful label write: the thread is done, and its
        # payload (for a newsletter, the whole story_results list) must not
        # linger for the daemon's session.
        cache = daemon.ResultCache()
        cache.put("t1", ("m1",), "payload")
        cache.clear("t1")
        assert cache.get("t1", ("m1",)) is None

    def test_prune_evicts_entries_for_threads_absent_this_cycle(self):
        # A thread that leaves the query — labeled externally, archived, or
        # given up to agent/attempted (the give-up path does NOT clear the
        # cache) — would otherwise leak its payload for the daemon's lifetime.
        cache = daemon.ResultCache()
        cache.put("t_gone", ("m1",), "payload")
        cache.put("t_here", ("m2",), "kept")

        cache.prune(["t_here"])

        assert cache.get("t_gone", ("m1",)) is None
        assert cache.get("t_here", ("m2",)) == "kept"


class TestDaemonHalt:
    """In-memory halt state for ONE function (out-of-funds); FunctionHalts pairs
    two of these (D5 scope, D19). Restart is the only reset."""

    def test_starts_untripped(self):
        halt = DaemonHalt()
        assert halt.tripped is False
        assert halt.reason is None

    def test_trip_sets_reason_and_tripped(self):
        halt = DaemonHalt()
        halt.trip("cloud provider out of funds")
        assert halt.tripped is True
        assert halt.reason == "cloud provider out of funds"

    def test_first_trip_wins(self):
        # Threads in one asyncio.gather cycle may race to trip; the reason must
        # stay stable (first wins) rather than churn with each late tripper.
        halt = DaemonHalt()
        halt.trip("first")
        halt.trip("second")
        assert halt.reason == "first"

    def test_trips_only_on_the_third_consecutive_balance_error(self):
        """Issue #73 / D22: one balance-403 was treated as permanent, and on
        2026-08-25 it was a provider-side false 403 that cost fourteen days.
        The slot now trips on the THIRD consecutive fault; the first two are
        plain deferrals."""
        halt = DaemonHalt()
        exc = LLMBalanceError("out of funds", tier="cloud", status_code=403)
        assert halt.record_balance_error(exc) is False
        assert halt.tripped is False
        assert halt.record_balance_error(exc) is False
        assert halt.tripped is False
        assert halt.record_balance_error(exc) is True
        assert halt.tripped is True
        assert halt.reason == "out of funds"
        assert halt.fault is exc
        # A fourth fault on a tripped slot changes nothing (first tripper wins).
        assert halt.record_balance_error(LLMBalanceError("later")) is False
        assert halt.reason == "out of funds"

    def test_a_success_resets_the_consecutive_count(self):
        halt = DaemonHalt()
        exc = LLMBalanceError("out of funds")
        halt.record_balance_error(exc)
        halt.record_balance_error(exc)
        halt.record_success()
        halt.record_balance_error(exc)
        halt.record_balance_error(exc)
        assert halt.tripped is False
        halt.record_balance_error(exc)
        assert halt.tripped is True

    def test_a_success_does_not_clear_a_tripped_slot(self):
        """Only the re-probe clears a halt (D22); a stray success (a request that
        was in flight when the slot tripped) must not un-halt the function."""
        halt = DaemonHalt()
        halt.trip("out of funds")
        halt.record_success()
        assert halt.tripped is True

    def test_tripping_records_the_probe_client(self):
        halt = DaemonHalt()
        client = MagicMock()
        exc = LLMBalanceError("out of funds")
        for _ in range(3):
            halt.record_balance_error(exc, probe_client=client)
        assert halt.probe_client is client

    def test_probe_is_due_once_per_interval(self):
        """The re-probe is paced from the trip, then from the last probe: one
        cheap request per interval, not one per poll cycle."""
        halt = DaemonHalt()
        halt.trip("out of funds", now=100.0)
        assert halt.probe_due(100.0 + 3599, 3600) is False
        assert halt.probe_due(100.0 + 3600, 3600) is True
        halt.last_probe_at = 3700.0
        assert halt.probe_due(3700.0 + 3599, 3600) is False
        assert halt.probe_due(3700.0 + 3600, 3600) is True

    def test_untripped_slot_is_never_due(self):
        assert DaemonHalt().probe_due(1e9, 0) is False

    def test_clear_returns_to_the_initial_state(self):
        halt = DaemonHalt()
        exc = LLMBalanceError("out of funds")
        for _ in range(3):
            halt.record_balance_error(exc, probe_client=MagicMock(), now=5.0)
        halt.notified = True
        halt.clear()
        fresh = DaemonHalt()
        assert vars(halt) == vars(fresh)


def _probe_client(*results):
    """An LLMClient stand-in whose probe() yields the given AvailabilityResults in turn."""
    client = MagicMock()
    client.probe = AsyncMock(side_effect=list(results))
    return client


_OK = AvailabilityResult(ok=True, status_code=200)
_STILL_403 = AvailabilityResult(ok=False, status_code=403)


class TestReprobeHalts:
    """A halted function re-probes its own provider on a slow schedule and
    resumes by itself when the probe answers (decision D22, issue #73)."""

    async def test_due_probe_that_answers_clears_the_slot(self, caplog):
        halts = daemon.FunctionHalts()
        client = _probe_client(_OK)
        halts.email.trip("out of funds", probe_client=client, now=0.0)

        with caplog.at_level(logging.INFO, logger="email-labeler"):
            resumed = await daemon.reprobe_halts(halts, now=3600.0, interval=3600)

        assert [name for name, _downtime in resumed] == ["email triage"]
        assert resumed[0][1] == pytest.approx(3600.0)
        assert halts.email.tripped is False
        client.probe.assert_awaited_once()
        lines = [r for r in caplog.records if "resumed" in r.getMessage()]
        assert len(lines) == 1
        assert lines[0].levelno == logging.INFO
        assert "email triage" in lines[0].getMessage()

    async def test_due_probe_that_fails_keeps_the_halt_quietly(self, caplog):
        """A failed hourly probe is expected while the account is genuinely
        empty: it logs below ERROR (the per-cycle halt line is already the
        loudness) and reschedules from now."""
        halts = daemon.FunctionHalts()
        client = _probe_client(_STILL_403)
        halts.email.trip("out of funds", probe_client=client, now=0.0)

        with caplog.at_level(logging.DEBUG, logger="email-labeler"):
            resumed = await daemon.reprobe_halts(halts, now=3600.0, interval=3600)

        assert resumed == []
        assert halts.email.tripped is True
        assert halts.email.last_probe_at == 3600.0
        probe_lines = [r for r in caplog.records if "re-probe" in r.getMessage()]
        assert len(probe_lines) == 1
        assert probe_lines[0].levelno < logging.ERROR
        assert "403" in probe_lines[0].getMessage()

    async def test_probe_not_yet_due_is_not_sent(self):
        halts = daemon.FunctionHalts()
        client = _probe_client(_OK)
        halts.email.trip("out of funds", probe_client=client, now=0.0)
        resumed = await daemon.reprobe_halts(halts, now=1800.0, interval=3600)
        assert resumed == []
        client.probe.assert_not_awaited()
        assert halts.email.tripped is True

    async def test_each_halted_function_probes_its_own_client(self):
        """Email and newsletter may sit on different providers: each slot probes
        the client that raised for it, and only the one that answers resumes."""
        halts = daemon.FunctionHalts(newsletter_enabled=True)
        email_client = _probe_client(_STILL_403)
        nl_client = _probe_client(_OK)
        halts.email.trip("email out of funds", probe_client=email_client, now=0.0)
        halts.newsletter.trip("nl out of funds", probe_client=nl_client, now=0.0)

        resumed = await daemon.reprobe_halts(halts, now=3600.0, interval=3600)

        assert [name for name, _d in resumed] == ["newsletter grading"]
        assert halts.email.tripped is True
        assert halts.newsletter.tripped is False
        email_client.probe.assert_awaited_once()
        nl_client.probe.assert_awaited_once()

    async def test_disabled_function_is_not_probed(self):
        halts = daemon.FunctionHalts(email_enabled=False, newsletter_enabled=True)
        client = _probe_client(_OK)
        halts.email.trip("out of funds", probe_client=client, now=0.0)
        await daemon.reprobe_halts(halts, now=3600.0, interval=3600)
        client.probe.assert_not_awaited()

    async def test_probe_exception_counts_as_a_failed_probe(self, caplog):
        """probe() already swallows httpx errors; anything else must still not
        escape into the poll loop, which sits outside the cycle's try/except."""
        halts = daemon.FunctionHalts()
        client = MagicMock()
        client.probe = AsyncMock(side_effect=ValueError("boom"))
        halts.email.trip("out of funds", probe_client=client, now=0.0)
        with caplog.at_level(logging.WARNING, logger="email-labeler"):
            resumed = await daemon.reprobe_halts(halts, now=3600.0, interval=3600)
        assert resumed == []
        assert halts.email.tripped is True
        assert any("boom" in r.getMessage() for r in caplog.records)

    async def test_slot_without_a_client_stays_halted_without_crashing(self, caplog):
        halts = daemon.FunctionHalts()
        halts.email.trip("out of funds", now=0.0)
        with caplog.at_level(logging.INFO, logger="email-labeler"):
            resumed = await daemon.reprobe_halts(halts, now=3600.0, interval=3600)
        assert resumed == []
        assert halts.email.tripped is True


class TestSummarizeCycle:
    def test_counts_handled_threads_and_drains_give_ups(self):
        # Give-ups return False from process_single_thread (D5, Wave 2 T8): they
        # are failures recorded by the poll loop's marking step, so given_up is
        # no longer a subset of the handled count.
        t = FailureTracker(max_failures=1)
        t.record_give_up("gaveup")  # recorded by _mark_thread_attempted
        items = [("ok", ["1"]), ("retry", ["2"]), ("gaveup", ["3"])]
        results = [True, False, False]  # gaveup returned False (a failure)
        processed, given_up = summarize_cycle(items, results, t)
        assert processed == 1  # only ok was handled
        assert given_up == ["gaveup"]  # reported distinctly, not part of processed
        assert t.take_given_up() == []  # already drained

    def test_clears_counts_for_handled_threads(self):
        t = FailureTracker(max_failures=2)
        t.record_failure("ok")  # failed last cycle, succeeds now
        summarize_cycle([("ok", ["1"])], [True], t)
        t.record_failure("ok")
        assert t.should_give_up("ok") is False  # count was reset on success

    def test_prunes_counts_for_threads_absent_this_cycle(self):
        t = FailureTracker(max_failures=2)
        t.record_failure("stale")
        t.record_failure("stale")  # at threshold from a prior cycle
        assert t.should_give_up("stale") is True
        # This cycle only has "active"; "stale" is gone from the query.
        summarize_cycle([("active", ["1"])], [False], t)
        assert t.should_give_up("stale") is False  # pruned


class TestLogLocalDeferrals:
    """One INFO summary per cycle for local-LLM deferrals (issue #24) — the
    per-thread handler stays below WARNING, this is the single visible line."""

    def test_emits_one_info_line_with_the_count(self, caplog):
        with caplog.at_level(logging.INFO, logger="email-labeler"):
            log_local_deferrals(["t1", "t2", "t3"])
        lines = [r for r in caplog.records if "Local LLM offline" in r.getMessage()]
        assert len(lines) == 1
        assert lines[0].levelno == logging.INFO
        assert "3" in lines[0].getMessage()

    def test_silent_when_nothing_deferred(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="email-labeler"):
            log_local_deferrals([])
        assert caplog.records == []


class TestIdleReport:
    """Pure decision helper for the idle transition + heartbeat lines (issue #58).
    One call per successful poll cycle; mutates IdleState and returns the line
    to log (or None)."""

    def test_busy_cycle_returns_none_and_clears_idle_state(self):
        state = IdleState(idle_since=100.0, last_heartbeat=100.0)
        line = idle_report(had_work=True, now=200.0, state=state, status_interval=900)
        assert line is None
        assert state.idle_since is None
        assert state.last_heartbeat is None

    def test_first_idle_cycle_logs_caught_up_and_stamps_state(self):
        state = IdleState()
        line = idle_report(had_work=False, now=100.0, state=state, status_interval=900)
        assert line == "Inbox caught up — nothing to process"
        assert state.idle_since == 100.0
        assert state.last_heartbeat == 100.0

    def test_idle_before_interval_stays_silent(self):
        state = IdleState(idle_since=100.0, last_heartbeat=100.0)
        line = idle_report(had_work=False, now=100.0 + 899, state=state, status_interval=900)
        assert line is None
        assert state.last_heartbeat == 100.0  # unchanged — heartbeat not consumed

    def test_idle_past_interval_logs_heartbeat_with_minute_math(self):
        state = IdleState(idle_since=100.0, last_heartbeat=100.0)
        line = idle_report(had_work=False, now=100.0 + 900, state=state, status_interval=900)
        assert line == "Still idle (15m) — last poll ok"
        assert state.last_heartbeat == 100.0 + 900

    def test_heartbeat_minutes_measure_total_idle_time_not_interval(self):
        # Two heartbeats in: idle_since is 30m ago even though the last
        # heartbeat was only 15m ago.
        state = IdleState(idle_since=100.0, last_heartbeat=100.0 + 900)
        line = idle_report(had_work=False, now=100.0 + 1800, state=state, status_interval=900)
        assert line == "Still idle (30m) — last poll ok"

    def test_work_after_idle_resets_so_next_idle_logs_caught_up_again(self):
        state = IdleState(idle_since=100.0, last_heartbeat=100.0)
        assert idle_report(had_work=True, now=200.0, state=state, status_interval=900) is None
        line = idle_report(had_work=False, now=300.0, state=state, status_interval=900)
        assert line == "Inbox caught up — nothing to process"
        assert state.idle_since == 300.0


class TestQuietHttpLogging:
    def test_quiet_http_logging_raises_httpx_loggers_to_warning(self, monkeypatch):
        """quiet_http_logging() must set the httpx AND httpcore logger levels to
        WARNING so per-poll 'HTTP Request: … 200 OK' INFO lines are suppressed
        (their URLs embed -label:agent/processed and read as false alarms)."""
        for name in ("httpx", "httpcore"):
            monkeypatch.setattr(logging.getLogger(name), "level", logging.NOTSET)
        daemon.quiet_http_logging()
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING

    def test_daemon_logger_still_emits_info_after_quieting(self, monkeypatch):
        """Production shape: 'email-labeler' has no explicit level and inherits
        INFO from basicConfig's root logger. Quieting the HTTP libraries must
        not suppress the daemon's own INFO lines — e.g. via a future edit that
        widens the tuple to "" (the root logger)."""
        monkeypatch.setattr(logging.getLogger("email-labeler"), "level", logging.NOTSET)
        monkeypatch.setattr(logging.getLogger(), "level", logging.INFO)
        daemon.quiet_http_logging()
        assert logging.getLogger("email-labeler").isEnabledFor(logging.INFO)

    def test_importing_daemon_does_not_quiet_httpx(self):
        """The evals import daemon for shared helpers; that import must not
        mutate process-wide logging. Quieting is applied by entry points
        (daemon.main, the eval CLIs), never at import."""
        code = (
            "import logging, daemon; "
            "print(logging.getLogger('httpx').level, logging.getLogger('httpcore').level)"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True,
            cwd=Path(daemon.__file__).parent,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == f"{logging.NOTSET} {logging.NOTSET}"

    def test_main_entry_point_quiets_http_logging(self, monkeypatch):
        for name in ("httpx", "httpcore"):
            monkeypatch.setattr(logging.getLogger(name), "level", logging.NOTSET)

        async def noop_daemon():
            return None

        monkeypatch.setattr(daemon, "run_daemon", noop_daemon)
        daemon.main()
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING


class _StopLoop(Exception):
    """Raised by the mocked inter-cycle sleep to break run_daemon's while True."""


async def run_poll_cycles(
    monkeypatch, tmp_path, poll_outcomes, process_mock=None, cycles=None,
    keep_newsletter=False, newsletter_output_file=None, label_manager=None,
    daemon_overrides=None,
):
    """Drive run_daemon through poll cycles, then stop; returns the proxy mock.

    Each poll_outcomes element is either a list_messages response dict or an
    exception instance to raise from that cycle's poll. All collaborators
    (proxy, LLMs, label verification, thread processing) are mocked; the
    inter-cycle asyncio.sleep is replaced with a counter that breaks the loop
    after `cycles` sleeps (default: one per scripted outcome). `process_mock`
    replaces the default always-succeeds process_single_thread stub — halted
    cycles sleep without polling, so cycles may exceed len(poll_outcomes).
    `keep_newsletter` retains the [newsletter] config so startup wiring (e.g.
    the assessments-path log line and the sink preflight) can be exercised;
    NEWSLETTER_ONLY stays unset unless the test sets it itself, so the loop
    remains on the plain email pipeline by default.
    `newsletter_output_file` overrides the configured sink path.
    `label_manager` supplies the instance the daemon's LabelManager constructor
    returns, so a test can assert on the writes the poll loop itself performs
    (e.g. the post-gather agent/attempted marking).
    `daemon_overrides` patches [daemon] config keys (e.g. max_failures) so a
    test can exercise a value config.toml does not ship.
    """
    config = copy.deepcopy(load_config())
    if not keep_newsletter:
        config.pop("newsletter", None)  # keep the loop on the plain email pipeline
    elif newsletter_output_file is not None:
        config["newsletter"]["output_file"] = str(newsletter_output_file)
    config["daemon"]["healthcheck_file"] = str(tmp_path / "healthcheck")
    if daemon_overrides:
        config["daemon"].update(daemon_overrides)
    monkeypatch.setattr(daemon, "load_config", lambda: config)

    proxy = MagicMock()
    proxy.proxy_url = "http://proxy.test"
    proxy.list_messages = AsyncMock(side_effect=poll_outcomes)
    monkeypatch.setattr(daemon, "GmailProxyClient", MagicMock(return_value=proxy))
    monkeypatch.setattr(daemon, "LLMClient", MagicMock())
    monkeypatch.setattr(daemon, "EmailClassifier", MagicMock())
    monkeypatch.setattr(
        daemon, "LabelManager",
        MagicMock(return_value=label_manager) if label_manager is not None else MagicMock(),
    )
    monkeypatch.setattr(daemon, "verify_labels_with_retry", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        daemon, "process_single_thread",
        process_mock if process_mock is not None else AsyncMock(return_value=True),
    )

    remaining = cycles if cycles is not None else len(poll_outcomes)

    async def cycle_sleep(_seconds):
        nonlocal remaining
        remaining -= 1
        if remaining <= 0:
            raise _StopLoop

    monkeypatch.setattr(daemon.asyncio, "sleep", cycle_sleep)
    with pytest.raises(_StopLoop):
        await daemon.run_daemon()
    return proxy


class TestPollLoopObservability:
    """Wiring tests for the issue #58 log lines: drive run_daemon through a
    scripted sequence of poll outcomes and assert on the log narrative."""

    async def test_reconnect_logged_after_connection_loss_recovery(
        self, monkeypatch, tmp_path, caplog
    ):
        with caplog.at_level(logging.INFO, logger="email-labeler"):
            await run_poll_cycles(
                monkeypatch, tmp_path,
                [ProxyUnavailableError("connection refused"), {"messages": []}],
            )
        assert any("Reconnected to api-proxy" in r.getMessage() for r in caplog.records)

    async def test_no_reconnect_line_when_connection_was_never_lost(
        self, monkeypatch, tmp_path, caplog
    ):
        # 4xx responses and in-cycle bugs grow the backoff too, but the proxy
        # stayed reachable — recovery from them must not claim a reconnect.
        with caplog.at_level(logging.INFO, logger="email-labeler"):
            await run_poll_cycles(
                monkeypatch, tmp_path,
                [
                    ProxyError("400 malformed query"),
                    {"messages": []},
                    ValueError("bug in the cycle"),
                    {"messages": []},
                ],
            )
        assert not any("Reconnected" in r.getMessage() for r in caplog.records)

    async def test_reconnect_line_precedes_the_recovery_cycle_work(
        self, monkeypatch, tmp_path, caplog
    ):
        # The narrative must read lost → reconnected → found/processed, not
        # have the reconnect line trail the work it explains.
        with caplog.at_level(logging.INFO, logger="email-labeler"):
            await run_poll_cycles(
                monkeypatch, tmp_path,
                [
                    ProxyUnavailableError("connection refused"),
                    {"messages": [{"id": "m1", "threadId": "t1"}]},
                ],
            )
        lines = [r.getMessage() for r in caplog.records]
        reconnect = next(i for i, m in enumerate(lines) if "Reconnected to api-proxy" in m)
        found = next(i for i, m in enumerate(lines) if "Found 1 unprocessed message(s)" in m)
        assert reconnect < found

    async def test_failed_cycle_resets_the_idle_heartbeat_clock(
        self, monkeypatch, tmp_path, caplog
    ):
        # idle → outage → idle: the second quiet cycle starts a fresh idle
        # stretch ("caught up" again) instead of extending one interrupted by
        # the outage — heartbeat minutes must never include downtime.
        with caplog.at_level(logging.INFO, logger="email-labeler"):
            await run_poll_cycles(
                monkeypatch, tmp_path,
                [
                    {"messages": []},
                    ProxyUnavailableError("connection refused"),
                    {"messages": []},
                ],
            )
        caught_up = [
            r for r in caplog.records
            if r.getMessage() == "Inbox caught up — nothing to process"
        ]
        assert len(caught_up) == 2


class TestStartupBuildLog:
    """Release identity (decision D11): run_daemon logs the baked build SHA once
    at startup so "what is deployed?" has an answer in the logs."""

    async def test_startup_logs_build_sha(self, monkeypatch, tmp_path, caplog):
        monkeypatch.setenv("GIT_SHA", "abc1234")
        with caplog.at_level(logging.INFO, logger="email-labeler"):
            await run_poll_cycles(monkeypatch, tmp_path, [{"messages": []}])
        assert any(
            r.getMessage() == "email-labeler starting — build abc1234"
            for r in caplog.records
        )


def _capture_trackers(monkeypatch):
    """Record the FailureTracker / MasqueradeTracker instances run_daemon builds.

    Wraps the real classes (rather than mocking them) so the daemon's own
    ``MasqueradeTracker(max_failures=failure_tracker.max_failures)`` wiring
    still reads a real threshold.
    """
    trackers: list[daemon.FailureTracker] = []
    masquerades: list[daemon.MasqueradeTracker] = []
    real_failure = daemon.FailureTracker
    real_masquerade = daemon.MasqueradeTracker

    def make_failure(*args, **kwargs):
        tracker = real_failure(*args, **kwargs)
        trackers.append(tracker)
        return tracker

    def make_masquerade(*args, **kwargs):
        tracker = real_masquerade(*args, **kwargs)
        masquerades.append(tracker)
        return tracker

    monkeypatch.setattr(daemon, "FailureTracker", make_failure)
    monkeypatch.setattr(daemon, "MasqueradeTracker", make_masquerade)
    return trackers, masquerades


class TestMaxFailuresKnob:
    """The strike bound is an operator knob (decision D5's `max_failures`
    corollary): config.toml `[daemon] max_failures` is its authoritative home
    (D7), `MAX_FAILURES` overrides it per run, and the same number sets the
    masquerade escalation threshold (D5's masquerade corollary)."""

    async def test_max_failures_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MAX_FAILURES", "2")
        trackers, masquerades = _capture_trackers(monkeypatch)

        await run_poll_cycles(monkeypatch, tmp_path, [{"messages": []}])

        assert trackers[0].max_failures == 2
        assert masquerades[0].max_failures == 2

    async def test_max_failures_defaults_to_config_value(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MAX_FAILURES", raising=False)
        trackers, masquerades = _capture_trackers(monkeypatch)

        await run_poll_cycles(
            monkeypatch, tmp_path, [{"messages": []}],
            daemon_overrides={"max_failures": 7},
        )

        assert trackers[0].max_failures == 7
        assert masquerades[0].max_failures == 7


async def _out_of_funds_process(*args, **kwargs):
    """process_single_thread stand-in: every thread hits an out-of-funds provider.

    Trips the email function's slot (decision D5/D19, Wave 2 T9: halts are
    per-function now). These tests run without [newsletter], so email triage is
    the only enabled function and its halt stands the whole daemon down.
    """
    kwargs["halts"].email.trip(
        "LLM provider out of funds — status 403 [tier=cloud]: NOT_ENOUGH_BALANCE"
    )
    return False


class TestOutOfFundsHalt:
    """Once a balance error halts every enabled function, the poll loop must stand
    down: no more polling or processing, a recurring admin instruction at ERROR,
    and a fresh heartbeat (the daemon is alive by design, not hung). Restart-only
    reset. Reworked for per-function halts (D5 scope, D19; Wave 2 T9) — here only
    email triage is enabled, so the daemon-wide behavior these pin is the
    all-enabled-functions-halted case. TestPerFunctionHalt covers partial halts."""

    async def test_halt_stops_polling(self, monkeypatch, tmp_path):
        proxy = await run_poll_cycles(
            monkeypatch, tmp_path,
            [{"messages": [{"id": "m1", "threadId": "t1"}]}],
            process_mock=_out_of_funds_process,
            cycles=3,
        )
        # Cycle 1 polls and trips the halt; cycles 2–3 must not poll again.
        assert proxy.list_messages.call_count == 1

    async def test_halt_logs_admin_instruction_every_cycle(self, monkeypatch, tmp_path, caplog):
        with caplog.at_level(logging.ERROR, logger="email-labeler"):
            await run_poll_cycles(
                monkeypatch, tmp_path,
                [{"messages": [{"id": "m1", "threadId": "t1"}]}],
                process_mock=_out_of_funds_process,
                cycles=3,
            )
        halted = [r for r in caplog.records if "resumes on its own" in r.getMessage()]
        # One line per halted cycle (2 of the 3) — outage-severity, must not
        # scroll out of a long-running container's logs.
        assert len(halted) == 2
        assert all(r.levelno == logging.ERROR for r in halted)
        assert all("add funds" in r.getMessage().lower() for r in halted)
        # The reason (provider identity included) is carried into the line.
        assert all("out of funds" in r.getMessage() for r in halted)

    async def test_halt_keeps_heartbeat_fresh(self, monkeypatch, tmp_path):
        heartbeat = MagicMock()
        real_path = daemon.Path

        def spy_path(arg):
            # Only the healthcheck file gets the spy; load_config still needs
            # a real Path to find config.toml.
            return heartbeat if "healthcheck" in str(arg) else real_path(arg)

        monkeypatch.setattr(daemon, "Path", spy_path)
        await run_poll_cycles(
            monkeypatch, tmp_path,
            [{"messages": [{"id": "m1", "threadId": "t1"}]}],
            process_mock=_out_of_funds_process,
            cycles=3,
        )
        # One write per cycle, halted or not: a stale heartbeat would misreport
        # a deliberately-halted daemon as hung.
        assert heartbeat.write_text.call_count == 3

    async def test_heartbeat_write_failure_while_halted_does_not_crash(
        self, monkeypatch, tmp_path, caplog
    ):
        """The halted branch sits outside the poll loop's try/except; a transient
        filesystem fault (disk full, read-only remount) must not kill the daemon —
        the recurring admin instruction is the whole point of the halt state."""
        heartbeat = MagicMock()
        heartbeat.write_text.side_effect = OSError("read-only file system")
        real_path = daemon.Path

        def spy_path(arg):
            return heartbeat if "healthcheck" in str(arg) else real_path(arg)

        monkeypatch.setattr(daemon, "Path", spy_path)
        with caplog.at_level(logging.ERROR, logger="email-labeler"):
            await run_poll_cycles(
                monkeypatch, tmp_path,
                [{"messages": [{"id": "m1", "threadId": "t1"}]}],
                process_mock=_out_of_funds_process,
                cycles=3,
            )
        # Both halted cycles survived the failed write and still logged the line.
        halted = [r for r in caplog.records if "resumes on its own" in r.getMessage()]
        assert len(halted) == 2


def _halt_email_with(client):
    """process_single_thread stand-in that trips the email slot with a probe client."""

    async def process(*args, **kwargs):
        kwargs["halts"].email.trip("out of funds", probe_client=client)
        return False

    return process


class TestHaltReprobeWiring:
    """run_daemon re-probes halted functions each cycle the interval allows
    (D22): a stood-down daemon polls again once the provider answers, an
    email-only halt's query narrowing is undone on resume, and the interval
    comes from config.toml with an env override (the MAX_FAILURES pattern)."""

    async def test_stood_down_daemon_resumes_polling_when_the_probe_answers(
        self, monkeypatch, tmp_path, caplog
    ):
        client = _probe_client(_STILL_403, _OK)
        with caplog.at_level(logging.INFO, logger="email-labeler"):
            proxy = await run_poll_cycles(
                monkeypatch, tmp_path,
                [
                    {"messages": [{"id": "m1", "threadId": "t1"}]},
                    {"messages": []},
                ],
                process_mock=_halt_email_with(client),
                cycles=3,
                daemon_overrides={"halt_probe_interval_seconds": 0},
            )
        # Cycle 1 polls and trips; cycle 2's probe fails (no poll); cycle 3's
        # probe answers, so cycle 3 polls again.
        assert proxy.list_messages.call_count == 2
        assert client.probe.await_count == 2
        assert any("resumed" in r.getMessage() for r in caplog.records)

    async def test_email_resume_restores_the_poll_query(self, monkeypatch, tmp_path):
        recipient = load_config()["newsletter"]["recipient"]
        client = _probe_client(_STILL_403, _OK, _STILL_403)
        proxy = await run_poll_cycles(
            monkeypatch, tmp_path,
            [
                {"messages": [{"id": "m1", "threadId": "t1"}]},
                {"messages": []},
                {"messages": []},
            ],
            process_mock=_halt_email_with(client),
            keep_newsletter=True,
            newsletter_output_file=tmp_path / "assessments.jsonl",
            daemon_overrides={"halt_probe_interval_seconds": 0},
        )
        queries = [c.kwargs["q"] for c in proxy.list_messages.call_args_list]
        # base → narrowed (probe failed, still halted) → base again (probe answered).
        assert [q.count(f"to:{recipient}") for q in queries] == [0, 1, 0]

    async def test_probe_interval_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HALT_PROBE_INTERVAL_SECONDS", "120")
        seen = []

        async def record(halts, now, interval):
            seen.append(interval)
            return []

        monkeypatch.setattr(daemon, "reprobe_halts", record)
        await run_poll_cycles(monkeypatch, tmp_path, [{"messages": []}])
        assert seen == [120]

    async def test_probe_interval_defaults_to_config_value(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HALT_PROBE_INTERVAL_SECONDS", raising=False)
        seen = []

        async def record(halts, now, interval):
            seen.append(interval)
            return []

        monkeypatch.setattr(daemon, "reprobe_halts", record)
        await run_poll_cycles(
            monkeypatch, tmp_path, [{"messages": []}],
            daemon_overrides={"halt_probe_interval_seconds": 777},
        )
        assert seen == [777]

    def test_shipped_config_probes_hourly(self):
        assert load_config()["daemon"]["halt_probe_interval_seconds"] == 3600


class TestPerFunctionHalt:
    """Functions fail independently (decision D5's scope rule; resolves D19's
    "today daemon-wide" note): a provider-balance fault halts the function whose
    provider reported it — loudly — while the other function keeps working. The
    daemon stands down entirely only when every enabled function is halted.
    """

    def test_no_enabled_function_is_not_all_halted(self):
        """Enabled-ness is a deployment fact, and `all([])` is True — without the
        `bool(slots)` guard a daemon with nothing enabled (NEWSLETTER_ONLY=1 on a
        config with no [newsletter]) would stand down on cycle 1 claiming every
        function had stopped, with an empty reason list."""
        halts = daemon.FunctionHalts(email_enabled=False, newsletter_enabled=False)
        assert halts.all_halted is False
        assert halts.any_halted is False

    def test_a_disabled_function_is_never_counted_as_halted(self):
        """The whole point of the enabled flags: a function that isn't running
        can't be halted, so the one enabled function's halt is the ONLY thing
        that decides the stand-down. Tripping the disabled slot changes nothing."""
        halts = daemon.FunctionHalts(email_enabled=False, newsletter_enabled=True)
        halts.email.trip("email provider out of funds")
        assert halts.all_halted is False
        assert halts.any_halted is False
        assert halts.halted_summary() == ""

        halts.newsletter.trip("newsletter provider out of funds")
        assert halts.all_halted is True
        assert halts.halted_summary() == "newsletter grading: newsletter provider out of funds"

    def test_email_only_halt_needs_both_functions_enabled(self):
        """The query narrowing it gates only makes sense when email triage is
        actually running and newsletter grading is there to narrow to."""
        no_newsletter = daemon.FunctionHalts(email_enabled=True, newsletter_enabled=False)
        no_newsletter.email.trip("cloud provider out of funds")
        assert no_newsletter.email_only_halted is False

        no_email = daemon.FunctionHalts(email_enabled=False, newsletter_enabled=True)
        no_email.email.trip("cloud provider out of funds")
        assert no_email.email_only_halted is False

    def _run_cycle(
        self, halts, mock_proxy, mock_classifier, mock_label_manager,
        mock_newsletter_classifier, cloud_sem, local_sem, output_file,
    ):
        """One miniature poll cycle: a newsletter thread and an email thread."""

        async def cycle():
            return [
                await process_single_thread(
                    tid, [tid], mock_proxy, mock_classifier, mock_label_manager,
                    cloud_sem, local_sem, max_thread_chars=50000,
                    newsletter_classifier=mock_newsletter_classifier,
                    newsletter_recipient="newsletters@dm.org",
                    newsletter_output_file=str(output_file),
                    halts=halts,
                )
                for tid in ("thread_nl", "thread_001")
            ]

        return cycle()

    async def test_newsletter_balance_fault_halts_newsletter_only(
        self, mock_proxy, mock_classifier, mock_label_manager,
        mock_newsletter_classifier, cloud_sem, local_sem,
        mock_thread_response, newsletter_thread_response, tmp_path,
    ):
        """The newsletter provider running out of funds stops newsletter grading
        only: the email sibling still classifies, in this cycle and the next."""
        threads = {
            "thread_nl": newsletter_thread_response,
            "thread_001": mock_thread_response,
        }
        mock_proxy.get_thread.side_effect = lambda thread_id: threads[thread_id]
        mock_newsletter_classifier.classify_newsletter.side_effect = LLMBalanceError(
            "newsletter provider out of funds"
        )
        halts = daemon.FunctionHalts(newsletter_enabled=True)
        args = (
            mock_proxy, mock_classifier, mock_label_manager,
            mock_newsletter_classifier, cloud_sem, local_sem,
            tmp_path / "assessments.jsonl",
        )

        # Three consecutive faults trip the slot (D22); the fourth cycle defers.
        cycles = [await self._run_cycle(halts, *args) for _ in range(4)]

        assert cycles == [[False, True]] * 4
        assert halts.newsletter.tripped is True
        assert halts.email.tripped is False
        # ...and the slot knows which client to re-probe.
        assert halts.newsletter.probe_client is mock_newsletter_classifier.cloud_llm
        # Email triage kept running across every cycle...
        assert mock_classifier.classify.await_count == 4
        assert mock_label_manager.apply_classification.await_count == 4
        # ...while the halted newsletter function stopped calling its dead
        # provider after the trip and committed nothing.
        assert mock_newsletter_classifier.classify_newsletter.await_count == 3
        mock_label_manager.apply_newsletter_classification.assert_not_called()
        assert not (tmp_path / "assessments.jsonl").exists()

    async def test_newsletter_success_resets_its_own_balance_count(
        self, mock_proxy, mock_classifier, mock_label_manager,
        mock_newsletter_classifier, cloud_sem, local_sem,
        newsletter_thread_response, tmp_path,
    ):
        """The consecutive count is per function: a newsletter grading that
        succeeds between two newsletter balance faults resets the NEWSLETTER
        count, so faults separated by a success never accumulate to a halt."""
        mock_proxy.get_thread.return_value = newsletter_thread_response
        good = [
            StoryResult(
                text="Content",
                scores={"simple": 3, "concrete": 3, "personal": 3, "dynamic": 3},
                average_score=3.0,
                tier=NewsletterTier.EXCELLENT,
                themes={"scripture": "emphasized"},
            )
        ]
        mock_newsletter_classifier.classify_newsletter.side_effect = [
            LLMBalanceError("nl out of funds"), LLMBalanceError("nl out of funds"),
            good,
            LLMBalanceError("nl out of funds"), LLMBalanceError("nl out of funds"),
        ]
        halts = daemon.FunctionHalts(newsletter_enabled=True)
        results = []
        for _ in range(5):
            results.append(await process_single_thread(
                "thread_nl", ["thread_nl"], mock_proxy, mock_classifier, mock_label_manager,
                cloud_sem, local_sem, max_thread_chars=50000,
                newsletter_classifier=mock_newsletter_classifier,
                newsletter_recipient="newsletters@dm.org",
                newsletter_output_file=str(tmp_path / "assessments.jsonl"),
                halts=halts,
            ))
        assert results == [False, False, True, False, False]
        assert halts.newsletter.tripped is False

    async def test_email_balance_fault_leaves_newsletter_running(
        self, mock_proxy, mock_classifier, mock_label_manager,
        mock_newsletter_classifier, cloud_sem, local_sem,
        mock_thread_response, newsletter_thread_response, tmp_path,
    ):
        """The mirror: an out-of-funds email tier halts email triage only —
        newsletter grading keeps grading and labeling."""
        threads = {
            "thread_nl": newsletter_thread_response,
            "thread_001": mock_thread_response,
        }
        mock_proxy.get_thread.side_effect = lambda thread_id: threads[thread_id]
        mock_classifier.classify_sender.side_effect = LLMBalanceError(
            "cloud provider out of funds"
        )
        mock_newsletter_classifier.classify_newsletter.return_value = [
            StoryResult(
                text="Content",
                scores={"simple": 3, "concrete": 3, "personal": 3, "dynamic": 3},
                average_score=3.0,
                tier=NewsletterTier.EXCELLENT,
                themes={"scripture": "emphasized"},
            )
        ]
        halts = daemon.FunctionHalts(newsletter_enabled=True)
        args = (
            mock_proxy, mock_classifier, mock_label_manager,
            mock_newsletter_classifier, cloud_sem, local_sem,
            tmp_path / "assessments.jsonl",
        )

        # Three consecutive faults trip the slot (D22); the fourth cycle defers.
        cycles = [await self._run_cycle(halts, *args) for _ in range(4)]

        assert cycles == [[True, False]] * 4
        assert halts.email.tripped is True
        assert halts.newsletter.tripped is False
        # Newsletter grading kept running across every cycle...
        assert mock_newsletter_classifier.classify_newsletter.await_count == 4
        assert mock_label_manager.apply_newsletter_classification.await_count == 4
        # ...while the halted email function stopped calling its dead provider
        # and committed nothing at all (no labels, no marker).
        assert mock_classifier.classify_sender.await_count == 3
        mock_label_manager.apply_classification.assert_not_called()
        mock_label_manager.mark_processed.assert_not_called()
        mock_label_manager.mark_attempted.assert_not_called()

    async def test_halted_email_function_narrows_the_poll_query(
        self, monkeypatch, tmp_path
    ):
        """While only email triage is halted, the poll query is narrowed to the
        newsletter recipient (the NEWSLETTER_ONLY precedent) so the halted
        function's backlog stops costing a get_thread per thread per cycle and
        can't crowd newsletter threads out of the max_results page. Halts are
        restart-reset, so the narrowing holds."""
        recipient = load_config()["newsletter"]["recipient"]

        async def halt_email(*args, **kwargs):
            kwargs["halts"].email.trip("cloud provider out of funds")
            return False

        proxy = await run_poll_cycles(
            monkeypatch, tmp_path,
            [
                {"messages": [{"id": "m1", "threadId": "t1"}]},
                {"messages": []},
                {"messages": []},
            ],
            process_mock=halt_email,
            keep_newsletter=True,
            newsletter_output_file=tmp_path / "assessments.jsonl",
        )

        # A partial halt keeps polling — all three cycles ran.
        queries = [c.kwargs["q"] for c in proxy.list_messages.call_args_list]
        assert len(queries) == 3
        assert f"to:{recipient}" not in queries[0]
        assert all(f"to:{recipient}" in q for q in queries[1:])

    async def test_halt_deferred_threads_record_no_cycle_failure(
        self, mock_proxy, mock_classifier, mock_label_manager,
        mock_newsletter_classifier, cloud_sem, local_sem,
        mock_thread_response, newsletter_thread_response, tmp_path,
    ):
        """A halt is a DEFERRAL, not a failure (D5/T9): neither function-aware
        skip may record a CycleFailure. One would be poison for the attribution
        step — a provider-shaped entry every cycle would either look like a
        single-thread masquerade (escalating a false suspect forever) or, as a
        candidate, strike the thread all the way to agent/attempted for the sole
        crime of belonging to the halted function. Both directions are checked
        against a REAL collector list, one halt at a time, so the partial-halt
        path runs rather than the all_halted short-circuit."""
        threads = {
            "thread_nl": newsletter_thread_response,
            "thread_001": mock_thread_response,
        }
        mock_proxy.get_thread.side_effect = lambda thread_id: threads[thread_id]
        failures: list[daemon.CycleFailure] = []

        def process(tid, halts):
            return process_single_thread(
                tid, [tid], mock_proxy, mock_classifier, mock_label_manager,
                cloud_sem, local_sem, max_thread_chars=50000,
                newsletter_classifier=mock_newsletter_classifier,
                newsletter_recipient="newsletters@dm.org",
                newsletter_output_file=str(tmp_path / "assessments.jsonl"),
                halts=halts, cycle_failures=failures,
            )

        nl_halts = daemon.FunctionHalts(newsletter_enabled=True)
        nl_halts.newsletter.trip("provider account balance exhausted")
        email_halts = daemon.FunctionHalts(newsletter_enabled=True)
        email_halts.email.trip("provider account balance exhausted")

        assert await process("thread_nl", nl_halts) is False
        assert await process("thread_001", email_halts) is False
        assert failures == []

    async def test_email_halt_commits_nothing_even_at_max_priority(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response,
    ):
        """The email-halt skip sits ABOVE the already-at-max-priority branch, so
        a halted email function commits NOTHING — not even the agent/processed
        marker that branch would otherwise write (D5 Rule 1: only successes
        commit outcomes). Marking under a halt would drop the thread out of
        gmail_query for good without it ever being classified.

        Only email is halted (newsletter grading is enabled and running), so the
        all_halted short-circuit at the top does not fire and the thread really
        does reach the priority check."""
        mock_proxy.get_thread.return_value = mock_thread_response
        mock_label_manager.get_existing_priority = MagicMock(
            return_value=_get_priority(EmailLabel.NEEDS_RESPONSE)
        )
        halts = daemon.FunctionHalts(newsletter_enabled=True)
        halts.email.trip("cloud provider out of funds")

        result = await process_single_thread(
            "thread_001", ["msg_001"], mock_proxy, mock_classifier, mock_label_manager,
            cloud_sem, local_sem, max_thread_chars=50000, halts=halts,
        )

        assert result is False
        mock_label_manager.mark_processed.assert_not_called()
        mock_label_manager.apply_classification.assert_not_called()

    async def test_query_is_narrowed_exactly_once(self, monkeypatch, tmp_path):
        """The email-only-halt narrowing appends `to:recipient` ONCE. Halts are
        restart-reset, so the partial-halt branch runs every cycle for the rest of
        the session: re-appending would grow the query without bound (and re-log
        the narrowing line) for as long as the daemon lives."""
        recipient = load_config()["newsletter"]["recipient"]

        async def halt_email(*args, **kwargs):
            kwargs["halts"].email.trip("cloud provider out of funds")
            return False

        proxy = await run_poll_cycles(
            monkeypatch, tmp_path,
            [
                {"messages": [{"id": "m1", "threadId": "t1"}]},
                {"messages": []},
                {"messages": []},
                {"messages": []},
            ],
            process_mock=halt_email,
            keep_newsletter=True,
            newsletter_output_file=tmp_path / "assessments.jsonl",
        )

        queries = [c.kwargs["q"] for c in proxy.list_messages.call_args_list]
        assert [q.count(f"to:{recipient}") for q in queries] == [0, 1, 1, 1]

    async def test_partial_halt_keeps_polling_and_names_the_halted_function(
        self, monkeypatch, tmp_path, caplog
    ):
        """A halted newsletter function must not stop the poll loop, and must not
        go quiet either: one ERROR per cycle names it and repeats the
        add-funds-and-restart instruction."""

        async def halt_newsletter(*args, **kwargs):
            kwargs["halts"].newsletter.trip("newsletter provider out of funds")
            return False

        with caplog.at_level(logging.ERROR, logger="email-labeler"):
            proxy = await run_poll_cycles(
                monkeypatch, tmp_path,
                [
                    {"messages": [{"id": "m1", "threadId": "t1"}]},
                    {"messages": []},
                    {"messages": []},
                ],
                process_mock=halt_newsletter,
                keep_newsletter=True,
                newsletter_output_file=tmp_path / "assessments.jsonl",
            )

        assert proxy.list_messages.call_count == 3
        halted = [r for r in caplog.records if "resumes on its own" in r.getMessage()]
        # Cycles 2 and 3 (the halt trips during cycle 1's processing).
        assert len(halted) == 2
        assert all(r.levelno == logging.ERROR for r in halted)
        assert all("newsletter" in r.getMessage() for r in halted)
        assert all("add funds" in r.getMessage().lower() for r in halted)

    async def test_partial_halt_error_names_the_function_not_merely_the_reason(
        self, monkeypatch, tmp_path, caplog
    ):
        """Sharper than the test above, which trips with a reason that happens to
        contain the word "newsletter" and so cannot tell a named function from a
        bare reason. The trip reason here mentions neither function, so the ERROR
        can only satisfy this by carrying halted_summary()'s `function: reason`
        prefix — the operator's one clue about WHICH function needs funds while
        the other keeps running."""

        async def halt_newsletter(*args, **kwargs):
            kwargs["halts"].newsletter.trip("provider account balance exhausted")
            return False

        with caplog.at_level(logging.ERROR, logger="email-labeler"):
            await run_poll_cycles(
                monkeypatch, tmp_path,
                [
                    {"messages": [{"id": "m1", "threadId": "t1"}]},
                    {"messages": []},
                    {"messages": []},
                ],
                process_mock=halt_newsletter,
                keep_newsletter=True,
                newsletter_output_file=tmp_path / "assessments.jsonl",
            )

        halted = [r for r in caplog.records if "resumes on its own" in r.getMessage()]
        assert len(halted) == 2  # cycles 2 and 3
        assert all(
            "newsletter grading: provider account balance exhausted" in r.getMessage()
            for r in halted
        )
        # ...and it must not smear the halt across the function still running.
        assert all("email triage" not in r.getMessage() for r in halted)

    async def test_both_functions_halted_stands_down(self, monkeypatch, tmp_path, caplog):
        """Only when EVERY enabled function is halted does the daemon stand down —
        and the recurring line names both."""

        async def halt_both(*args, **kwargs):
            kwargs["halts"].email.trip("cloud provider out of funds")
            kwargs["halts"].newsletter.trip("newsletter provider out of funds")
            return False

        with caplog.at_level(logging.ERROR, logger="email-labeler"):
            proxy = await run_poll_cycles(
                monkeypatch, tmp_path,
                [{"messages": [{"id": "m1", "threadId": "t1"}]}],
                process_mock=halt_both,
                cycles=3,
                keep_newsletter=True,
                newsletter_output_file=tmp_path / "assessments.jsonl",
            )

        # Cycle 1 polls and trips both halts; cycles 2–3 must not poll again.
        assert proxy.list_messages.call_count == 1
        halted = [r for r in caplog.records if "resumes on its own" in r.getMessage()]
        assert len(halted) == 2
        assert all("email" in r.getMessage() for r in halted)
        assert all("newsletter" in r.getMessage() for r in halted)

    async def test_newsletter_only_stands_down_on_a_newsletter_halt(
        self, monkeypatch, tmp_path, caplog
    ):
        """Under NEWSLETTER_ONLY email triage is not an enabled function — the
        skip sits above every email-tier call, so its halt slot can never trip.
        A newsletter-provider halt is therefore ALL of them: the daemon must
        stand down, not keep polling while logging the partial-halt line's false
        "the other function keeps running"."""
        monkeypatch.setenv("NEWSLETTER_ONLY", "1")

        async def halt_newsletter(*args, **kwargs):
            kwargs["halts"].newsletter.trip("newsletter provider out of funds")
            return False

        with caplog.at_level(logging.ERROR, logger="email-labeler"):
            proxy = await run_poll_cycles(
                monkeypatch, tmp_path,
                # Two spare outcomes so a daemon that wrongly keeps polling fails
                # on the assertion below rather than on an exhausted side_effect.
                [
                    {"messages": [{"id": "m1", "threadId": "t1"}]},
                    {"messages": []},
                    {"messages": []},
                ],
                process_mock=halt_newsletter,
                cycles=3,
                keep_newsletter=True,
                newsletter_output_file=tmp_path / "assessments.jsonl",
            )

        # Cycle 1 polls and trips the halt; cycles 2–3 must not poll again.
        assert proxy.list_messages.call_count == 1
        stood_down = [
            r for r in caplog.records if "every enabled function stopped" in r.getMessage()
        ]
        assert len(stood_down) == 2
        # ...and never the partial-halt line, which would claim email triage is
        # still working through a backlog it skips unconditionally.
        assert not any("the other function keeps running" in r.getMessage() for r in caplog.records)


class TestNewsletterOutputPathLogging:
    async def test_startup_logs_resolved_assessments_path(self, monkeypatch, tmp_path, caplog):
        """Startup must log the RESOLVED absolute assessments path: output_file is
        relative to the process working directory, so in Docker a missing bind
        mount silently strands records in the container layer (lost on recreate).
        The destination must be visible in the first log lines."""
        with caplog.at_level(logging.INFO, logger="email-labeler"):
            await run_poll_cycles(
                monkeypatch, tmp_path, [{"messages": []}], keep_newsletter=True
            )
        expected = str(Path(load_config()["newsletter"]["output_file"]).resolve())
        assert any(expected in r.getMessage() for r in caplog.records), (
            f"no startup log line contains the resolved assessments path {expected}"
        )

    async def test_startup_reports_how_many_records_the_sink_already_holds(
        self, monkeypatch, tmp_path, caplog
    ):
        """The count is the tell for a misdirected sink: a daemon that has been
        grading for weeks against a path holding 0 records is not appending to
        the file the operator reviews."""
        sink = tmp_path / "data" / "assessments.jsonl"
        sink.parent.mkdir()
        sink.write_text('{"thread_id": "a"}\n{"thread_id": "b"}\n')

        with caplog.at_level(logging.INFO, logger="email-labeler"):
            await run_poll_cycles(
                monkeypatch, tmp_path, [{"messages": []}],
                keep_newsletter=True, newsletter_output_file=sink,
            )

        assert any(
            str(sink) in r.getMessage() and "2 existing" in r.getMessage()
            for r in caplog.records
        ), "startup line did not report the sink's existing record count"

    async def test_startup_errors_when_the_sink_is_not_persisted(
        self, monkeypatch, tmp_path, caplog
    ):
        """The silent failure this guards: in a container with no volume over the
        assessments directory, every write succeeds and every record is lost on
        the next recreate. It must be an ERROR at startup, not a surprise later."""
        sink = tmp_path / "data" / "assessments.jsonl"
        monkeypatch.setattr(daemon, "running_in_container", lambda: True)
        monkeypatch.setattr(
            daemon, "read_mountinfo",
            lambda: "1450 1449 0:118 / / rw,relatime - overlay overlay rw,lowerdir=/l/A\n",
        )

        with caplog.at_level(logging.INFO, logger="email-labeler"):
            await run_poll_cycles(
                monkeypatch, tmp_path, [{"messages": []}],
                keep_newsletter=True, newsletter_output_file=sink,
            )

        errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert any(str(sink) in m and "volume" in m.lower() for m in errors), (
            f"no startup ERROR warned that {sink} is not persisted; errors={errors}"
        )

    async def test_startup_names_the_host_directory_behind_the_mount(
        self, monkeypatch, tmp_path, caplog
    ):
        """A mount pointing at a host directory other than the one being reviewed
        fails exactly like no mount at all — records accumulate somewhere nobody
        looks — but no check can know which directory the operator *meant*. So
        name the source: the one line that makes it verifiable at a glance."""
        sink = tmp_path / "data" / "assessments.jsonl"
        monkeypatch.setattr(daemon, "running_in_container", lambda: True)
        monkeypatch.setattr(
            daemon, "read_mountinfo",
            lambda: (
                "1450 1449 0:118 / / rw,relatime - overlay overlay rw,lowerdir=/l/A\n"
                f"1470 1450 259:2 /srv/elsewhere/data {tmp_path / 'data'} rw,relatime "
                "- ext4 /dev/sda1 rw\n"
            ),
        )

        with caplog.at_level(logging.INFO, logger="email-labeler"):
            await run_poll_cycles(
                monkeypatch, tmp_path, [{"messages": []}],
                keep_newsletter=True, newsletter_output_file=sink,
            )

        assert any(
            "/srv/elsewhere/data" in r.getMessage() for r in caplog.records
        ), "startup did not name the host source of the mount holding the sink"
        # A real mount covers it, so the container-layer ERROR must stay quiet.
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    async def test_no_persistence_error_outside_a_container(
        self, monkeypatch, tmp_path, caplog
    ):
        """On a host, a path on the root filesystem is perfectly durable — the
        check must not fire there."""
        monkeypatch.setattr(daemon, "running_in_container", lambda: False)
        monkeypatch.setattr(
            daemon, "read_mountinfo",
            lambda: "1450 1449 0:118 / / rw,relatime - overlay overlay rw,lowerdir=/l/A\n",
        )

        with caplog.at_level(logging.INFO, logger="email-labeler"):
            await run_poll_cycles(
                monkeypatch, tmp_path, [{"messages": []}],
                keep_newsletter=True,
                newsletter_output_file=tmp_path / "data" / "assessments.jsonl",
            )

        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    async def test_startup_errors_when_no_sink_is_configured(
        self, monkeypatch, tmp_path, caplog
    ):
        """The third silent way assessments go missing: [newsletter] with no
        output_file at all. Grading runs, labels apply, the per-thread INFO
        summary prints — and nothing is ever recorded. Startup must say so."""
        with caplog.at_level(logging.INFO, logger="email-labeler"):
            await run_poll_cycles(
                monkeypatch, tmp_path, [{"messages": []}],
                keep_newsletter=True, newsletter_output_file="",
            )

        errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert any("output_file" in m for m in errors), (
            f"startup did not flag the missing assessments sink; errors={errors}"
        )

    async def test_startup_errors_when_the_sink_cannot_be_read(
        self, monkeypatch, tmp_path, caplog
    ):
        """An unreadable sink is reported, not shrugged at. The realistic cause is
        an ``output_file`` naming a directory: ``os.access`` calls a directory
        writable, so without this the one guaranteed-fatal misconfiguration
        produces nothing louder than an INFO — and then every newsletter is graded
        and abandoned to agent/attempted."""
        sink = tmp_path / "data" / "assessments.jsonl"
        sink.mkdir(parents=True)

        with caplog.at_level(logging.INFO, logger="email-labeler"):
            await run_poll_cycles(
                monkeypatch, tmp_path, [{"messages": []}],
                keep_newsletter=True, newsletter_output_file=sink,
            )

        errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert any(str(sink) in m for m in errors), (
            f"an unreadable/misshapen sink produced no startup ERROR; errors={errors}"
        )

    async def test_startup_errors_when_the_sink_is_not_writable(
        self, monkeypatch, tmp_path, caplog
    ):
        sink = tmp_path / "data" / "assessments.jsonl"
        monkeypatch.setattr(daemon, "sink_writability_warning", lambda _p: "sink is read-only")

        with caplog.at_level(logging.INFO, logger="email-labeler"):
            await run_poll_cycles(
                monkeypatch, tmp_path, [{"messages": []}],
                keep_newsletter=True, newsletter_output_file=sink,
            )

        assert any(
            "sink is read-only" in r.getMessage()
            for r in caplog.records if r.levelno >= logging.ERROR
        )


class TestLoadConfig:
    def test_loads_config_toml(self):
        config = load_config()
        assert "daemon" in config
        assert "labels" in config
        assert "llm" in config
        assert "prompts" in config

    def test_config_has_required_daemon_keys(self):
        config = load_config()
        assert "poll_interval_seconds" in config["daemon"]
        assert "max_emails_per_cycle" in config["daemon"]
        assert "gmail_query" in config["daemon"]
        assert "healthcheck_file" in config["daemon"]

    def test_config_has_all_labels(self):
        config = load_config()
        for key in (
            "needs_response",
            "fyi",
            "low_priority",
            "processed",
            "personal",
            "non_personal",
        ):
            assert key in config["labels"]

    def test_config_has_prompts(self):
        config = load_config()
        assert "system" in config["prompts"]["sender_classification"]
        assert "user_template" in config["prompts"]["sender_classification"]
        email_config = config["prompts"]["email_classification"]
        assert "preamble" in email_config
        assert "postamble" in email_config
        assert "categories" in email_config
        assert "user_template" in email_config

    def test_config_has_vip_senders(self):
        config = load_config()
        assert "vip_senders" in config
        assert "categories" in config["vip_senders"]

    def test_config_has_max_thread_chars(self):
        config = load_config()
        assert "max_thread_chars" in config["daemon"]
        assert isinstance(config["daemon"]["max_thread_chars"], int)

    def test_config_has_parallel_settings(self):
        config = load_config()
        assert "cloud_parallel" in config["daemon"]
        assert "local_parallel" in config["daemon"]
        assert config["daemon"]["cloud_parallel"] >= 1
        assert config["daemon"]["local_parallel"] >= 1

    def test_config_disables_native_thinking_on_local_classifier(self):
        # Issue #10: the eval showed native thinking on the local person-email
        # classifier is strictly worse (budget-split failures where reasoning
        # overruns max_tokens and no label is emitted). The disable dialect is
        # backend-specific: this test guards the chat_template_kwargs form
        # (honored by mlx_lm.server / LM Studio, ignored by Ollama); the Ollama
        # form is reasoning_effort = "none", guarded by
        # test_config_local_extra_body_disables_thinking_on_ollama (issue #64).
        # The cloud classifier is unaffected.
        config = load_config()
        local = config["llm"]["local"]
        # Layout guard: the flag must be in the nested chat_template_kwargs form.
        # mlx_lm.server honors this form and ignores a top-level enable_thinking,
        # so this specific nesting is load-bearing — a refactor to the top-level
        # form would silently stop disabling thinking on mlx_lm.server even
        # though llm_client treats both as no-think.
        ctk = local["extra_body"]["chat_template_kwargs"]
        assert ctk["enable_thinking"] is False
        # Behavior guard: the LLMClient the daemon builds from this config must
        # actually treat the request as thinking-disabled.
        client = LLMClient(
            base_url="", api_key="", model=local["model"],
            extra_body=local.get("extra_body"),
        )
        assert client._extra_body_disables_thinking() is True

    def test_config_local_extra_body_disables_thinking_on_ollama(self):
        # Issue #64: Ollama's OpenAI-compat endpoint ignores the
        # chat_template_kwargs.enable_thinking form entirely (measured on
        # 0.32.5), so native thinking was silently re-enabled and reasoning
        # consumed the whole max_tokens budget before any content was emitted.
        # The one field that Ollama honors is a top-level
        # reasoning_effort = "none" ("low" is a silent no-op — only "none"
        # disables). llm_client merges extra_body at the top level of the
        # request body, so this is pure config data.
        config = load_config()
        extra_body = config["llm"]["local"]["extra_body"]
        assert extra_body["reasoning_effort"] == "none"

    def test_config_local_max_tokens_covers_thinking_off_decode(self):
        # Issue #64: with thinking off, the prompts still elicit the full
        # step-by-step scaffold as untagged content, and observed demand ran up
        # to 1,293 completion tokens — beyond the old 1024 budget. An
        # over-budget decode truncates content before any label; llm_client now
        # raises LLMContentError on finish_reason "length" (it used to parse to
        # a silent LOW_PRIORITY default), so an undersized budget means loud
        # give-ups rather than mislabels — still lost mail. The budget raise is
        # therefore coupled to the thinking disable, not optional insurance.
        # 2048 is the measured floor; shipped value 4096.
        config = load_config()
        assert config["llm"]["local"]["max_tokens"] >= 2048

    def test_config_newsletter_max_tokens_covers_multi_story_extraction(self):
        # Issue #64 collateral: story_extraction re-emits the FULL text of every
        # story in one response, so its output scales with newsletter size — the
        # one call whose demand 1024 demonstrably cannot bound. With llm_client
        # now raising on finish_reason "length" (instead of silently truncating
        # mid-story), an undersized budget would turn every long newsletter into
        # a deterministic give-up (agent/attempted). Sized by that reasoning
        # (issue #64's Fix 4), not by live measurement — the newsletter tier was
        # never probed.
        config = load_config()
        assert config["newsletter"]["llm"]["max_tokens"] >= 2048

    def test_config_has_newsletter_section(self):
        config = load_config()
        assert "newsletter" in config
        assert "recipient" in config["newsletter"]
        assert "output_file" in config["newsletter"]
        assert "labels" in config["newsletter"]
        assert "prompts" in config["newsletter"]

    def test_config_has_newsletter_labels(self):
        config = load_config()
        nl = config["newsletter"]["labels"]
        assert "newsletter" in nl
        assert "excellent" in nl
        assert "good" in nl
        assert "fair" in nl
        assert "poor" in nl
        assert "no_stories" in nl
        assert "themes" in nl
        assert len(nl["themes"]) == 5

    def test_config_has_newsletter_prompts(self):
        config = load_config()
        prompts = config["newsletter"]["prompts"]
        assert "story_extraction" in prompts
        assert "quality_assessment" in prompts
        assert "theme_classification" in prompts
        for key in ("story_extraction", "quality_assessment", "theme_classification"):
            assert "system" in prompts[key]
            assert "user_template" in prompts[key]


class TestResolveIntEnv:
    def test_returns_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("LOCAL_PARALLEL", raising=False)
        assert resolve_int_env("LOCAL_PARALLEL", 4) == 4

    def test_env_overrides_default(self, monkeypatch):
        monkeypatch.setenv("LOCAL_PARALLEL", "6")
        assert resolve_int_env("LOCAL_PARALLEL", 4) == 6

    def test_blank_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("LOCAL_PARALLEL", "   ")
        assert resolve_int_env("LOCAL_PARALLEL", 4) == 4

    def test_invalid_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MAX_EMAILS_PER_CYCLE", "lots")
        assert resolve_int_env("MAX_EMAILS_PER_CYCLE", 10) == 10

    def test_strips_whitespace_around_value(self, monkeypatch):
        monkeypatch.setenv("LOCAL_PARALLEL", "  2 ")
        assert resolve_int_env("LOCAL_PARALLEL", 4) == 2

    def test_zero_falls_back_to_default(self, monkeypatch):
        # 0 parses fine but is out of range: Semaphore(0) deadlocks the daemon.
        monkeypatch.setenv("LOCAL_PARALLEL", "0")
        assert resolve_int_env("LOCAL_PARALLEL", 4) == 4

    def test_negative_falls_back_to_default(self, monkeypatch):
        # -1 parses fine but is out of range: Semaphore(-1) crashes at startup.
        monkeypatch.setenv("LOCAL_PARALLEL", "-1")
        assert resolve_int_env("LOCAL_PARALLEL", 4) == 4

    def test_value_at_minimum_is_allowed(self, monkeypatch):
        monkeypatch.setenv("LOCAL_PARALLEL", "1")
        assert resolve_int_env("LOCAL_PARALLEL", 4) == 1

    def test_max_emails_zero_falls_back_to_default(self, monkeypatch):
        # The lower-bound guard protects every numeric override, not just concurrency:
        # max_results=0 would make the daemon process nothing each cycle.
        monkeypatch.setenv("MAX_EMAILS_PER_CYCLE", "0")
        assert resolve_int_env("MAX_EMAILS_PER_CYCLE", 10) == 10


@pytest.fixture
def mock_newsletter_classifier():
    classifier = AsyncMock()
    # A real model string, not an AsyncMock attribute: the assessment record is
    # JSON-serialized before the labels commit, so a non-serializable model would
    # fail the write (and, by design, block labeling) in every newsletter test.
    classifier.cloud_llm.model = "claude-sonnet-4-6"
    return classifier


@pytest.fixture
def newsletter_thread_response():
    body = "This month's campus update features Sarah's journey..."
    return {
        "id": "thread_nl",
        "snippet": "This month's campus update...",
        "messages": [
            {
                "id": "msg_nl_001",
                "threadId": "thread_nl",
                "internalDate": "1704067200000",
                "labelIds": ["INBOX", "UNREAD"],
                "payload": {
                    "headers": [
                        {"name": "From", "value": "John Staff <john@dm.org>"},
                        {"name": "To", "value": "newsletters@dm.org"},
                        {"name": "Subject", "value": "February Campus Update"},
                        {"name": "Date", "value": "Mon, 1 Jan 2024 12:00:00 +0000"},
                    ],
                    "body": {
                        "data": base64.urlsafe_b64encode(body.encode()).decode(),
                    },
                },
            },
        ],
    }


class TestNewsletterRouting:
    async def test_newsletter_skips_priority_classification(
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        mock_newsletter_classifier,
        cloud_sem,
        local_sem,
        newsletter_thread_response,
        tmp_path,
    ):
        mock_proxy.get_thread.return_value = newsletter_thread_response
        mock_newsletter_classifier.classify_newsletter.return_value = [
            StoryResult(
                text="Content",
                scores={"simple": 3, "concrete": 3, "personal": 3, "dynamic": 3},
                average_score=3.0,
                tier=NewsletterTier.EXCELLENT,
                themes={"scripture": "emphasized"},
            )
        ]

        result = await process_single_thread(
            "thread_nl",
            ["msg_nl_001"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
            newsletter_classifier=mock_newsletter_classifier,
            newsletter_recipient="newsletters@dm.org",
            newsletter_output_file=str(tmp_path / "assessments.jsonl"),
        )

        assert result is True
        mock_classifier.classify_sender.assert_not_called()
        mock_classifier.classify.assert_not_called()
        mock_newsletter_classifier.classify_newsletter.assert_called_once()
        mock_label_manager.apply_newsletter_classification.assert_called_once()

    async def test_assessment_records_send_date_and_model(
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        mock_newsletter_classifier,
        cloud_sem,
        local_sem,
        newsletter_thread_response,
        tmp_path,
    ):
        """The assessment record persists the email send-date (from the Date header,
        ISO-8601 UTC) and the classifier model, distinct from the processed timestamp
        (shared enabler for #35/#36)."""
        mock_proxy.get_thread.return_value = newsletter_thread_response
        mock_newsletter_classifier.classify_newsletter.return_value = [
            StoryResult(
                text="Content",
                scores={"simple": 3, "concrete": 3, "personal": 3, "dynamic": 3},
                average_score=3.0,
                tier=NewsletterTier.EXCELLENT,
                themes={"scripture": "emphasized"},
            )
        ]
        # The classifier's cloud LLM reports the model that actually ran.
        mock_newsletter_classifier.cloud_llm.model = "claude-sonnet-4-6"
        out = tmp_path / "assessments.jsonl"

        result = await process_single_thread(
            "thread_nl",
            ["msg_nl_001"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
            newsletter_classifier=mock_newsletter_classifier,
            newsletter_recipient="newsletters@dm.org",
            newsletter_output_file=str(out),
        )

        assert result is True
        record = json.loads(out.read_text().strip())
        # newsletter_thread_response's Date header is "Mon, 1 Jan 2024 12:00:00 +0000".
        assert record["send_date"] == "2024-01-01T12:00:00+00:00"
        assert record["model"] == "claude-sonnet-4-6"
        # send-date (email-intrinsic) is distinct from the processed timestamp.
        assert record["timestamp"] != record["send_date"]

    async def test_content_error_is_a_strike_candidate_not_an_empty_commit(
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        cloud_sem,
        local_sem,
        newsletter_thread_response,
        tmp_path,
    ):
        """#30 end-to-end: a content-less grade error must route the newsletter to
        the strike-candidate path — recorded as a CycleFailure for the poll loop's
        correlation attribution (D5), which decides whether it strikes — and must
        NOT commit an empty no-stories label/assessment (which would be
        indistinguishable from a genuine NO_STORIES newsletter)."""
        from llm_client import LLMContentError
        from newsletter import NewsletterClassifier

        mock_proxy.get_thread.return_value = newsletter_thread_response
        fake_llm = AsyncMock()
        fake_llm.model = "test-model"
        fake_llm.complete.side_effect = [
            ("STORY: A real story about ministry work.", ""),  # extract_stories
            LLMContentError("model returned no content"),      # assess_quality
        ]
        nl_config = {
            "newsletter": {
                "prompts": {
                    "story_extraction": {"system": "s", "user_template": "{body}"},
                    "quality_assessment": {"system": "s", "user_template": "{text}"},
                    "theme_classification": {"system": "s", "user_template": "{text}"},
                }
            }
        }
        classifier = NewsletterClassifier(cloud_llm=fake_llm, config=nl_config)
        out = tmp_path / "assessments.jsonl"
        failures: list[daemon.CycleFailure] = []

        result = await process_single_thread(
            "thread_nl",
            ["msg_nl_001"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
            newsletter_classifier=classifier,
            newsletter_recipient="newsletters@dm.org",
            newsletter_output_file=str(out),
            cycle_failures=failures,
        )

        # Routed to the strike-candidate path (recorded for the poll loop's
        # attribution — D5, Wave 2 T8), NOT committed as a (false) no-stories
        # outcome.
        assert result is False
        mock_label_manager.apply_newsletter_classification.assert_not_called()
        assert not out.exists()  # no empty assessment record written
        assert [f.signature for f in failures] == ["LLMContentError"]
        assert failures[0].provider_shaped is False

    @staticmethod
    def _real_newsletter_classifier(replies):
        """A real NewsletterClassifier over a fake LLM returning ``replies`` in order."""
        from newsletter import NewsletterClassifier

        fake_llm = AsyncMock()
        fake_llm.model = "test-model"
        fake_llm.complete.side_effect = replies
        nl_config = {
            "newsletter": {
                "prompts": {
                    "story_extraction": {"system": "s", "user_template": "{body}"},
                    "quality_assessment": {"system": "s", "user_template": "{text}"},
                    "theme_classification": {"system": "s", "user_template": "{text}"},
                }
            }
        }
        return NewsletterClassifier(cloud_llm=fake_llm, config=nl_config)

    async def test_unparseable_extraction_commits_nothing(
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        cloud_sem,
        local_sem,
        newsletter_thread_response,
        tmp_path,
    ):
        """D5/D20: an unparseable extraction reply is a failure, not a `no-stories`
        outcome — no labels, no assessment record, a strike candidate instead."""
        mock_proxy.get_thread.return_value = newsletter_thread_response
        classifier = self._real_newsletter_classifier(
            [("I could not find any stories, sorry!", "")]  # extract_stories
        )
        out = tmp_path / "assessments.jsonl"
        failures: list[daemon.CycleFailure] = []

        result = await process_single_thread(
            "thread_nl",
            ["msg_nl_001"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
            newsletter_classifier=classifier,
            newsletter_recipient="newsletters@dm.org",
            newsletter_output_file=str(out),
            cycle_failures=failures,
        )

        assert result is False
        mock_label_manager.apply_newsletter_classification.assert_not_called()
        assert not out.exists()
        assert [f.signature for f in failures] == ["LLMContentError"]

    async def test_all_grades_unparseable_commits_nothing(
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        cloud_sem,
        local_sem,
        newsletter_thread_response,
        tmp_path,
    ):
        """D5/D20: stories extracted but not one gradable — no `no-stories` label and
        no tier-less assessment record; the thread defers as a strike candidate."""
        mock_proxy.get_thread.return_value = newsletter_thread_response
        classifier = self._real_newsletter_classifier(
            [
                ("STORY: A real story about ministry work.", ""),  # extract_stories
                ("garbled quality output", ""),                    # assess_quality
                ("SCRIPTURE: PRESENT", ""),                        # classify_themes
            ]
        )
        out = tmp_path / "assessments.jsonl"
        failures: list[daemon.CycleFailure] = []

        result = await process_single_thread(
            "thread_nl",
            ["msg_nl_001"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
            newsletter_classifier=classifier,
            newsletter_recipient="newsletters@dm.org",
            newsletter_output_file=str(out),
            cycle_failures=failures,
        )

        assert result is False
        mock_label_manager.apply_newsletter_classification.assert_not_called()
        assert not out.exists()
        assert [f.signature for f in failures] == ["LLMContentError"]

    async def test_non_newsletter_uses_priority_pipeline(
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        mock_newsletter_classifier,
        cloud_sem,
        local_sem,
        mock_thread_response,
        tmp_path,
    ):
        mock_proxy.get_thread.return_value = mock_thread_response

        result = await process_single_thread(
            "thread_001",
            ["msg_001", "msg_002"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
            newsletter_classifier=mock_newsletter_classifier,
            newsletter_recipient="newsletters@dm.org",
            newsletter_output_file=str(tmp_path / "assessments.jsonl"),
        )

        assert result is True
        mock_classifier.classify_sender.assert_called_once()
        mock_classifier.classify.assert_called_once()
        mock_newsletter_classifier.classify_newsletter.assert_not_called()

    async def test_newsletter_no_stories(
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        mock_newsletter_classifier,
        cloud_sem,
        local_sem,
        newsletter_thread_response,
        tmp_path,
    ):
        mock_proxy.get_thread.return_value = newsletter_thread_response
        mock_newsletter_classifier.classify_newsletter.return_value = []

        result = await process_single_thread(
            "thread_nl",
            ["msg_nl_001"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
            newsletter_classifier=mock_newsletter_classifier,
            newsletter_recipient="newsletters@dm.org",
            newsletter_output_file=str(tmp_path / "assessments.jsonl"),
        )

        assert result is True
        call_kwargs = mock_label_manager.apply_newsletter_classification.call_args.kwargs
        assert call_kwargs["tier"] is None
        assert call_kwargs["themes"] == {}

    async def test_newsletter_only_skips_non_newsletter(
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        mock_newsletter_classifier,
        cloud_sem,
        local_sem,
        mock_thread_response,
        tmp_path,
    ):
        mock_proxy.get_thread.return_value = mock_thread_response

        result = await process_single_thread(
            "thread_001",
            ["msg_001", "msg_002"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
            newsletter_classifier=mock_newsletter_classifier,
            newsletter_recipient="newsletters@dm.org",
            newsletter_output_file=str(tmp_path / "assessments.jsonl"),
            newsletter_only=True,
        )

        assert result is False
        mock_classifier.classify_sender.assert_not_called()
        mock_classifier.classify.assert_not_called()
        mock_newsletter_classifier.classify_newsletter.assert_not_called()

    async def test_newsletter_only_still_processes_newsletters(
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        mock_newsletter_classifier,
        cloud_sem,
        local_sem,
        newsletter_thread_response,
        tmp_path,
    ):
        mock_proxy.get_thread.return_value = newsletter_thread_response
        mock_newsletter_classifier.classify_newsletter.return_value = [
            StoryResult(
                text="Content",
                scores={"simple": 3, "concrete": 3, "personal": 3, "dynamic": 3},
                average_score=3.0,
                tier=NewsletterTier.EXCELLENT,
                themes={"scripture": "emphasized"},
            )
        ]

        result = await process_single_thread(
            "thread_nl",
            ["msg_nl_001"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
            newsletter_classifier=mock_newsletter_classifier,
            newsletter_recipient="newsletters@dm.org",
            newsletter_output_file=str(tmp_path / "assessments.jsonl"),
            newsletter_only=True,
        )

        assert result is True
        mock_newsletter_classifier.classify_newsletter.assert_called_once()
        mock_classifier.classify_sender.assert_not_called()

    async def test_newsletter_without_classifier_falls_through(
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        cloud_sem,
        local_sem,
        newsletter_thread_response,
    ):
        mock_proxy.get_thread.return_value = newsletter_thread_response

        result = await process_single_thread(
            "thread_nl",
            ["msg_nl_001"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
        )

        assert result is True
        mock_classifier.classify_sender.assert_called_once()


class TestNewsletterAssessmentDurability:
    """The assessment JSONL is the only place a newsletter's grading survives —
    Gmail keeps just the coarse tier/theme labels, and the labels include
    ``agent/processed``, which drops the thread out of ``gmail_query`` forever.

    So the record must be persisted BEFORE the labels commit. Committing first
    and writing after makes any sink failure (missing bind mount made
    read-only, full disk, bad permissions) permanently lose that newsletter's
    grade: the thread is already marked processed, so it is never re-graded.
    """

    @staticmethod
    def _story():
        return StoryResult(
            text="Content",
            scores={"simple": 3, "concrete": 3, "personal": 3, "dynamic": 3},
            average_score=3.0,
            tier=NewsletterTier.EXCELLENT,
            themes={"scripture": "emphasized"},
        )

    async def test_record_is_on_disk_before_labels_are_applied(
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        mock_newsletter_classifier,
        cloud_sem,
        local_sem,
        newsletter_thread_response,
        tmp_path,
    ):
        mock_proxy.get_thread.return_value = newsletter_thread_response
        mock_newsletter_classifier.classify_newsletter.return_value = [self._story()]
        out = tmp_path / "assessments.jsonl"

        seen_at_label_time = {}

        async def _capture(**_kwargs):
            seen_at_label_time["text"] = out.read_text() if out.exists() else ""

        mock_label_manager.apply_newsletter_classification.side_effect = _capture

        result = await process_single_thread(
            "thread_nl",
            ["msg_nl_001"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
            newsletter_classifier=mock_newsletter_classifier,
            newsletter_recipient="newsletters@dm.org",
            newsletter_output_file=str(out),
        )

        assert result is True
        assert seen_at_label_time["text"].strip(), (
            "labels were applied while the assessment file was still empty — a "
            "sink failure at that point loses the grading permanently"
        )

    async def test_sink_failure_leaves_thread_unlabeled_for_retry(
        self,
        monkeypatch,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        mock_newsletter_classifier,
        cloud_sem,
        local_sem,
        newsletter_thread_response,
    ):
        """An unwritable sink must NOT be swallowed: no labels, no
        agent/processed, and a False return so the thread is retried next
        cycle — forever.

        A sink fault is shared-cause (disk), so it is never counted toward
        give-up (decision D5's sink corollary, Wave 2 T12): the newsletter is
        retried every cycle and is never abandoned to agent/attempted. Before
        T12 the bare OSError walked into the generic Exception arm as a strike
        candidate, so a read-only mount abandoned graded newsletters after
        max_failures cycles — this drives one cycle past the threshold to pin
        the reversal.
        """
        mock_proxy.get_thread.return_value = newsletter_thread_response
        mock_newsletter_classifier.classify_newsletter.return_value = [self._story()]

        def _boom(**_kwargs):
            raise OSError(30, "Read-only file system")

        monkeypatch.setattr(daemon, "write_assessment", _boom)
        tracker = FailureTracker(max_failures=3)

        for _ in range(tracker.max_failures + 1):
            results = await drive_attribution_cycle(
                [("thread_nl", ["msg_nl_001"])],
                mock_proxy,
                mock_classifier,
                mock_label_manager,
                cloud_sem,
                local_sem,
                tracker=tracker,
                newsletter_classifier=mock_newsletter_classifier,
                newsletter_recipient="newsletters@dm.org",
                newsletter_output_file="/nonexistent/assessments.jsonl",
            )
            assert results == [False]

        mock_label_manager.apply_newsletter_classification.assert_not_called()
        mock_label_manager.mark_processed.assert_not_called()
        # Never counted: no strike accrued, so no marker and no give-up.
        mock_label_manager.mark_attempted.assert_not_called()
        assert tracker.should_give_up("thread_nl") is False
        assert tracker.take_given_up() == []

    async def test_sink_failure_names_the_path_at_error_level(
        self,
        monkeypatch,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        mock_newsletter_classifier,
        cloud_sem,
        local_sem,
        newsletter_thread_response,
        caplog,
    ):
        """The operator has to be able to tell a sink fault from a grading fault,
        so the failing path is named at ERROR."""
        mock_proxy.get_thread.return_value = newsletter_thread_response
        mock_newsletter_classifier.classify_newsletter.return_value = [self._story()]

        def _boom(**_kwargs):
            raise OSError(30, "Read-only file system")

        monkeypatch.setattr(daemon, "write_assessment", _boom)

        with caplog.at_level(logging.ERROR, logger="email-labeler"):
            await process_single_thread(
                "thread_nl",
                ["msg_nl_001"],
                mock_proxy,
                mock_classifier,
                mock_label_manager,
                cloud_sem,
                local_sem,
                max_thread_chars=50000,
                newsletter_classifier=mock_newsletter_classifier,
                newsletter_recipient="newsletters@dm.org",
                newsletter_output_file="/nonexistent/assessments.jsonl",
            )

        assert any(
            "/nonexistent/assessments.jsonl" in r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.ERROR
        ), "no ERROR log line named the assessments path that failed to write"


class TestResultReuse:
    """Issue #29 (Wave 2 T6): a transient label-write fault must not discard the
    finished classification.

    The session ``ResultCache`` keeps the classified result keyed by the
    thread's message-id fingerprint, so a later cycle re-attempts only the
    write — no Stage 1/Stage 2 re-run (the scarce local GPU pass for person
    threads), no newsletter re-extraction/re-grading, and no duplicate JSONL
    record for the same thread content.
    """

    @staticmethod
    def _story():
        return StoryResult(
            text="Content",
            scores={"simple": 3, "concrete": 3, "personal": 3, "dynamic": 3},
            average_score=3.0,
            tier=NewsletterTier.EXCELLENT,
            themes={"scripture": "emphasized"},
        )

    async def test_write_failure_then_retry_does_not_reclassify(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response,
    ):
        """First cycle classifies but the label write fails; the second cycle
        must land the labels WITHOUT re-running Stage 1/Stage 2."""
        mock_proxy.get_thread.return_value = mock_thread_response
        mock_label_manager.apply_classification.side_effect = [
            ProxyUnavailableError("proxy 503 on write"),
            None,
        ]
        cache = daemon.ResultCache()

        results = [
            await process_single_thread(
                "thread_001", ["msg_001", "msg_002"], mock_proxy, mock_classifier,
                mock_label_manager, cloud_sem, local_sem, max_thread_chars=16000,
                result_cache=cache,
            )
            for _ in range(2)
        ]

        assert results == [False, True]
        # Classified exactly once across both cycles — the retry reused the cache.
        assert mock_classifier.classify_sender.call_count == 1
        assert mock_classifier.classify.call_count == 1
        # And the labels landed on the second attempt, on the full thread.
        assert mock_label_manager.apply_classification.call_count == 2
        applied = mock_label_manager.apply_classification.call_args
        assert applied.args[0] == ["msg_001", "msg_002"]
        assert applied.args[1] == EmailLabel.NEEDS_RESPONSE

    async def test_newsletter_write_failure_reuses_grading_and_does_not_reappend(
        self, mock_proxy, mock_classifier, mock_label_manager, mock_newsletter_classifier,
        cloud_sem, local_sem, newsletter_thread_response, tmp_path,
    ):
        """Same shape on the newsletter path: the grading is reused and the JSONL
        append is not repeated — exactly one record for the thread."""
        mock_proxy.get_thread.return_value = newsletter_thread_response
        mock_newsletter_classifier.classify_newsletter.return_value = [self._story()]
        mock_label_manager.apply_newsletter_classification.side_effect = [
            ProxyUnavailableError("proxy 503 on write"),
            None,
        ]
        out = tmp_path / "assessments.jsonl"
        cache = daemon.ResultCache()

        results = [
            await process_single_thread(
                "thread_nl", ["msg_nl_001"], mock_proxy, mock_classifier,
                mock_label_manager, cloud_sem, local_sem, max_thread_chars=16000,
                newsletter_classifier=mock_newsletter_classifier,
                newsletter_recipient="newsletters@dm.org",
                newsletter_output_file=str(out),
                result_cache=cache,
            )
            for _ in range(2)
        ]

        assert results == [False, True]
        # Graded exactly once across both cycles.
        mock_newsletter_classifier.classify_newsletter.assert_called_once()
        assert mock_label_manager.apply_newsletter_classification.call_count == 2
        # Exactly one JSONL record — the write-retry cycle did not re-append.
        records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        assert len(records) == 1
        assert records[0]["thread_id"] == "thread_nl"

    async def test_new_message_invalidates_cached_result(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response,
    ):
        """A new message changes the thread's fingerprint: the cached result is
        dropped and the thread is classified fresh (staleness answer)."""
        grown = copy.deepcopy(mock_thread_response)
        grown["messages"].append(
            {
                "id": "msg_003",
                "threadId": "thread_001",
                "internalDate": "1704074400000",
                "labelIds": ["INBOX", "UNREAD"],
                "payload": {
                    "headers": [
                        {"name": "From", "value": "John Doe <john@example.com>"},
                        {"name": "Subject", "value": "Re: Meeting tomorrow"},
                        {"name": "Date", "value": "Mon, 1 Jan 2024 14:00:00 +0000"},
                    ],
                    "body": {"data": base64.urlsafe_b64encode(b"New reply").decode()},
                },
            }
        )
        mock_proxy.get_thread.side_effect = [mock_thread_response, grown]
        mock_label_manager.apply_classification.side_effect = [
            ProxyUnavailableError("proxy 503 on write"),
            None,
        ]
        cache = daemon.ResultCache()

        results = [
            await process_single_thread(
                "thread_001", ["msg_001", "msg_002"], mock_proxy, mock_classifier,
                mock_label_manager, cloud_sem, local_sem, max_thread_chars=16000,
                result_cache=cache,
            )
            for _ in range(2)
        ]

        assert results == [False, True]
        # The grown thread was classified fresh, not served from the cache.
        assert mock_classifier.classify.call_count == 2
        # The second write covered the new message too.
        applied = mock_label_manager.apply_classification.call_args
        assert applied.args[0] == ["msg_001", "msg_002", "msg_003"]

    async def test_poll_loop_hands_every_cycle_the_same_cache(self, monkeypatch, tmp_path):
        """The wiring the tests above assume: run_daemon builds ONE session cache
        and passes it to every thread of every cycle. Dropping the kwarg reverts
        the daemon to pre-T6 behaviour silently — Stage 1 + Stage 2 (the scarce
        local GPU pass) re-run every cycle while a write keeps failing — which
        matters because T8 removed the strike bound on write-phase faults on the
        strength of this reuse."""
        seen = []

        async def process(tid, msg_ids, *args, **kwargs):
            seen.append(kwargs.get("result_cache"))
            return True

        await run_poll_cycles(
            monkeypatch, tmp_path,
            [
                {"messages": [{"id": "m1", "threadId": "t1"}]},
                {"messages": [{"id": "m2", "threadId": "t2"}]},
            ],
            process_mock=process,
        )

        assert len(seen) == 2
        assert all(isinstance(c, daemon.ResultCache) for c in seen)
        assert seen[0] is seen[1]  # one session cache, not one per cycle

    async def test_poll_loop_prunes_cached_results_for_threads_that_left(
        self, monkeypatch, tmp_path
    ):
        """The per-cycle prune is the ONLY eviction path for a thread that never
        gets a successful write and then leaves the query — given up to
        agent/attempted (the give-up path does not clear the cache), archived, or
        relabeled externally. Without it the payload (for a newsletter, the whole
        story_results list) leaks for the daemon's lifetime."""
        caches = []

        async def process(tid, msg_ids, *args, **kwargs):
            cache = kwargs.get("result_cache")
            caches.append(cache)
            cache.put(tid, tuple(msg_ids), f"payload-{tid}")
            return False

        await run_poll_cycles(
            monkeypatch, tmp_path,
            [
                {"messages": [{"id": "m1", "threadId": "t1"}]},
                {"messages": [{"id": "m2", "threadId": "t2"}]},
            ],
            process_mock=process,
        )

        assert len(caches) == 2
        cache = caches[0]
        # t1 was absent from cycle 2's page, so its entry is gone...
        assert cache.get("t1", ("m1",)) is None
        # ...while the thread this cycle did see kept its cached result.
        assert cache.get("t2", ("m2",)) == "payload-t2"


class TestVerifyLabelsWithRetry:
    """Startup label verification must survive a transiently-unreachable api-proxy.

    Regression guard for the daemon crash-loop. Two boot-time conditions are
    transient and must be waited out rather than crashed on:
      * the proxy is slow/down — a transport fault (ConnectError/ConnectTimeout/
        read timeout/dropped connection); and
      * the proxy is up but its Gmail backend is still warming, answering 5xx,
        which surfaces as proxy_client.ProxyError (NOT an httpx.TransportError).
    Permanent failures (a misconfigured PROXY_URL → httpx.UnsupportedProtocol, a
    bad key → ProxyAuthError, a programming error) must surface immediately so a
    real misconfiguration is not masked as a silent, endless retry.
    """

    @staticmethod
    def _label_manager(side_effect):
        label_manager = AsyncMock()
        label_manager.proxy.proxy_url = "http://proxy:8000"
        label_manager.verify_labels.side_effect = side_effect
        return label_manager

    async def test_retries_on_connect_timeout(self):
        """ConnectTimeout (proxy slow/unreachable at startup) is retried, not propagated."""
        label_manager = self._label_manager([httpx.ConnectTimeout("connect timed out"), []])

        missing = await daemon.verify_labels_with_retry(
            label_manager, initial_backoff=0, max_backoff=0,
        )

        assert missing == []
        assert label_manager.verify_labels.call_count == 2

    async def test_retries_on_connect_error(self):
        """ConnectError stays retryable (existing transient-outage behavior preserved)."""
        label_manager = self._label_manager(
            [httpx.ConnectError("connection refused"), ["agent/processed"]]
        )

        missing = await daemon.verify_labels_with_retry(
            label_manager, initial_backoff=0, max_backoff=0,
        )

        assert missing == ["agent/processed"]
        assert label_manager.verify_labels.call_count == 2

    async def test_retries_on_proxy_5xx_error(self):
        """A warming proxy answers 5xx → ProxyError; this is transient and must retry.

        Regression for the original crash-loop: ProxyError is not an
        httpx.TransportError, so a TransportError-only catch let it propagate and
        the daemon exited — the exact failure this helper exists to prevent.
        """
        label_manager = self._label_manager([ProxyError("Proxy error: 503"), []])

        missing = await daemon.verify_labels_with_retry(
            label_manager, initial_backoff=0, max_backoff=0,
        )

        assert missing == []
        assert label_manager.verify_labels.call_count == 2

    async def test_propagates_unsupported_protocol(self):
        """A misconfigured PROXY_URL (UnsupportedProtocol) is permanent — fail fast.

        UnsupportedProtocol is an httpx.TransportError subclass, so a base-class
        catch would retry it forever and mask the misconfiguration as a hang.
        """
        label_manager = self._label_manager(
            httpx.UnsupportedProtocol("Request URL has no scheme")
        )

        with pytest.raises(httpx.UnsupportedProtocol):
            await daemon.verify_labels_with_retry(
                label_manager, initial_backoff=0, max_backoff=0,
            )

        assert label_manager.verify_labels.call_count == 1

    async def test_propagates_auth_error(self):
        """A bad PROXY_API_KEY (ProxyAuthError) is permanent — surface immediately."""
        label_manager = self._label_manager(ProxyAuthError("Unauthorized"))

        with pytest.raises(ProxyAuthError):
            await daemon.verify_labels_with_retry(
                label_manager, initial_backoff=0, max_backoff=0,
            )

        assert label_manager.verify_labels.call_count == 1

    async def test_propagates_programming_error(self):
        """A non-transient programming error must surface immediately, not retry forever."""
        label_manager = self._label_manager(RuntimeError("boom"))

        with pytest.raises(RuntimeError, match="boom"):
            await daemon.verify_labels_with_retry(
                label_manager, initial_backoff=0, max_backoff=0,
            )

        assert label_manager.verify_labels.call_count == 1


class TestNewsletterLLMEndpoint:
    """The newsletter grader can use a different provider than the cloud classifier.

    Newsletter quality grading is configured for a Claude model
    (config.toml [newsletter.llm] model = "claude-sonnet-4-6"), but the cloud
    classification endpoint (CLOUD_LLM_URL) points at a provider that doesn't
    serve Claude (e.g. Novita) — so requesting that model there 404s. These env
    vars let the newsletter LLM target its own Claude-serving endpoint.
    """

    def test_defaults_to_cloud_endpoint(self, monkeypatch):
        """Without NEWSLETTER_LLM_*, the newsletter LLM shares the cloud endpoint."""
        monkeypatch.setenv("CLOUD_LLM_URL", "https://novita.example/v1/chat/completions")
        monkeypatch.setenv("CLOUD_LLM_API_KEY", "novita-key")
        monkeypatch.delenv("NEWSLETTER_LLM_URL", raising=False)
        monkeypatch.delenv("NEWSLETTER_LLM_API_KEY", raising=False)

        url, key = daemon.resolve_newsletter_llm_endpoint()

        assert url == "https://novita.example/v1/chat/completions"
        assert key == "novita-key"

    def test_overrides_with_newsletter_env(self, monkeypatch):
        """NEWSLETTER_LLM_* point the newsletter LLM at its own provider (e.g. Anthropic)."""
        monkeypatch.setenv("CLOUD_LLM_URL", "https://novita.example/v1/chat/completions")
        monkeypatch.setenv("CLOUD_LLM_API_KEY", "novita-key")
        monkeypatch.setenv("NEWSLETTER_LLM_URL", "https://api.anthropic.com/v1/chat/completions")
        monkeypatch.setenv("NEWSLETTER_LLM_API_KEY", "sk-ant-newsletter")

        url, key = daemon.resolve_newsletter_llm_endpoint()

        assert url == "https://api.anthropic.com/v1/chat/completions"
        assert key == "sk-ant-newsletter"

    def test_partial_override_does_not_borrow_cloud_key(self, monkeypatch):
        """An override URL must never silently pair with the cloud provider's key.

        The override is atomic: setting NEWSLETTER_LLM_URL alone targets the new
        endpoint with an empty key (auth fails clearly) rather than the cloud key
        (which would authenticate against the wrong provider and 401 confusingly).
        """
        monkeypatch.setenv("CLOUD_LLM_URL", "https://novita.example/v1/chat/completions")
        monkeypatch.setenv("CLOUD_LLM_API_KEY", "novita-key")
        monkeypatch.setenv("NEWSLETTER_LLM_URL", "https://api.anthropic.com/v1/chat/completions")
        monkeypatch.delenv("NEWSLETTER_LLM_API_KEY", raising=False)

        url, key = daemon.resolve_newsletter_llm_endpoint()

        assert url == "https://api.anthropic.com/v1/chat/completions"
        assert key != "novita-key"
        assert key == ""
