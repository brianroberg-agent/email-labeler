"""Tests for notify.py — ntfy pushes on daemon halt and resume (decision D22, issue #73)."""

import logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from llm_client import LLMBalanceError
from notify import HaltNotifier, halt_message, resume_message

URL = "https://ntfy.example.test/labeler-alerts"


def _patched_client(response=None, error=None):
    """Patch notify's httpx.AsyncClient; returns the mock so posts can be asserted."""
    mock_client = AsyncMock()
    if error is not None:
        mock_client.post.side_effect = error
    else:
        mock_client.post.return_value = response
    cm = patch("notify.httpx.AsyncClient")
    mock_cls = cm.start()
    mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
    return cm, mock_client


def _response(status):
    return httpx.Response(status, request=httpx.Request("POST", URL))


class TestFromEnv:
    def test_both_vars_set_enables(self, monkeypatch, caplog):
        monkeypatch.setenv("NTFY_URL", URL)
        monkeypatch.setenv("NTFY_TOKEN", "tk_secret")
        with caplog.at_level(logging.INFO, logger="email-labeler"):
            notifier = HaltNotifier.from_env()
        assert notifier.enabled is True
        assert not any("disabled" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize("missing", ["NTFY_URL", "NTFY_TOKEN"])
    def test_a_missing_var_disables_with_one_warning(self, monkeypatch, caplog, missing):
        monkeypatch.setenv("NTFY_URL", URL)
        monkeypatch.setenv("NTFY_TOKEN", "tk_secret")
        monkeypatch.delenv(missing)
        with caplog.at_level(logging.WARNING, logger="email-labeler"):
            notifier = HaltNotifier.from_env()
        assert notifier.enabled is False
        warnings = [r for r in caplog.records if "halt notifications disabled" in r.getMessage()]
        assert len(warnings) == 1
        assert warnings[0].levelno == logging.WARNING
        assert "NTFY_URL/NTFY_TOKEN not set" in warnings[0].getMessage()
        # The token is a secret: the warning must not echo any env value.
        assert "tk_secret" not in warnings[0].getMessage()

    def test_blank_values_count_as_unset(self, monkeypatch):
        monkeypatch.setenv("NTFY_URL", "  ")
        monkeypatch.setenv("NTFY_TOKEN", "tk_secret")
        assert HaltNotifier.from_env().enabled is False


class TestSend:
    async def test_posts_title_and_body_with_bearer_token(self):
        cm, client = _patched_client(response=_response(200))
        try:
            ok = await HaltNotifier(URL, "tk_secret").send("a title", "a body")
        finally:
            cm.stop()
        assert ok is True
        client.post.assert_awaited_once()
        args, kwargs = client.post.call_args
        assert args[0] == URL
        assert kwargs["headers"]["Authorization"] == "Bearer tk_secret"
        assert kwargs["headers"]["Title"] == "a title"
        assert kwargs["content"] == "a body"

    async def test_disabled_notifier_sends_nothing(self):
        cm, client = _patched_client(response=_response(200))
        try:
            ok = await HaltNotifier(None, None).send("t", "b")
        finally:
            cm.stop()
        assert ok is False
        client.post.assert_not_awaited()

    async def test_transport_error_logs_a_warning_and_does_not_raise(self, caplog):
        cm, _client = _patched_client(error=httpx.ConnectError("refused"))
        try:
            with caplog.at_level(logging.WARNING, logger="email-labeler"):
                ok = await HaltNotifier(URL, "tk_secret").send("t", "b")
        finally:
            cm.stop()
        assert ok is False
        assert any("notification" in r.getMessage().lower() for r in caplog.records)
        assert not any("tk_secret" in r.getMessage() for r in caplog.records)

    async def test_non_2xx_logs_a_warning_and_does_not_raise(self, caplog):
        cm, _client = _patched_client(response=_response(403))
        try:
            with caplog.at_level(logging.WARNING, logger="email-labeler"):
                ok = await HaltNotifier(URL, "tk_secret").send("t", "b")
        finally:
            cm.stop()
        assert ok is False
        assert any("403" in r.getMessage() for r in caplog.records)

    async def test_unexpected_exception_does_not_raise(self, caplog):
        cm, _client = _patched_client(error=ValueError("boom"))
        try:
            with caplog.at_level(logging.WARNING, logger="email-labeler"):
                ok = await HaltNotifier(URL, "tk_secret").send("t", "b")
        finally:
            cm.stop()
        assert ok is False


class TestMessages:
    def test_halt_message_names_function_provider_status_reason_and_time(self):
        fault = LLMBalanceError(
            "LLM provider out of funds — status 403 [tier=cloud model=zai-org/glm-5]: ...",
            tier="cloud", model="zai-org/glm-5", status_code=403,
            detail='{"code":403,"reason":"NOT_ENOUGH_BALANCE","message":"not enough balance"}',
            signature="NOT_ENOUGH_BALANCE",
        )
        title, body = halt_message(
            "email triage", fault, tripped_wall=1_757_419_200.0, probe_interval=3600
        )
        assert title == "email-labeler halted: email triage"
        assert "cloud" in body
        assert "zai-org/glm-5" in body
        assert "403" in body
        assert "NOT_ENOUGH_BALANCE" in body
        assert "2025-09-09" in body  # the trip time, wall clock
        assert "3600" in body or "60 min" in body or "1h" in body

    def test_halt_message_without_provenance_still_reads(self):
        title, body = halt_message("newsletter grading", None, tripped_wall=None, probe_interval=3600)
        assert title == "email-labeler halted: newsletter grading"
        assert "newsletter grading" in body

    def test_halt_message_forwards_the_signature_not_the_provider_body(self):
        """Review of #81 (Opus F4): a 400 with a balance signature may echo the
        request it rejected. The push carries the matched phrase and the HTTP
        status; the response body stays in the daemon log."""
        fault = LLMBalanceError(
            "x", tier="cloud", model="m", status_code=400,
            detail='{"error":"insufficient_quota","request":"Subject: Re: your invoice"}',
            signature="insufficient_quota",
        )
        _title, body = halt_message("email triage", fault, tripped_wall=0.0, probe_interval=3600)
        assert "insufficient_quota" in body
        assert "400" in body
        assert "your invoice" not in body
        assert "request" not in body

    def test_halt_message_for_a_402_names_the_status_only(self):
        fault = LLMBalanceError(
            "x", tier="cloud", model="m", status_code=402,
            detail='{"error":"payment required","echo":"Subject: hello"}',
        )
        _title, body = halt_message("email triage", fault, tripped_wall=0.0, probe_interval=3600)
        assert "402" in body
        assert "hello" not in body

    def test_resume_message_names_function_and_downtime(self):
        title, body = resume_message("email triage", 5400.0)
        assert title == "email-labeler resumed: email triage"
        assert "1h 30m" in body
