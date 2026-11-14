"""Tests for the Claude comparison eval (evals/compare.py, issue #92).

No test reaches a real service. The Anthropic arm runs through the real SDK
over an ``httpx2.MockTransport``, so request shapes and response parsing are the
SDK's own; the local arm and the cloud tier are plain fake classes. Every
fixture is synthetic mail (``example.test`` addresses, invented names).
"""

import argparse
import asyncio
import base64
import functools
import getpass
import io
import json
import logging
import sys
import warnings

import anthropic
import httpx2
import pytest

from classifier import EmailClassifier
from daemon import load_config
from evals import compare, report
from evals.schemas import GoldenThread, PredictionResult
from llm_client import AvailabilityResult, LLMContentError, LLMUnavailableError

SENTINEL_KEY = "sk-ant-test-SENTINEL-0000"


def _message(text="<think>it asks for a date</think>\nNEEDS_RESPONSE", *, stop_reason="end_turn",
             input_tokens=120, output_tokens=30, thinking=None, stop_details=None):
    content = []
    if thinking is not None:
        content.append({"type": "thinking", "thinking": thinking, "signature": "sig"})
    if text is not None:
        content.append({"type": "text", "text": text})
    message = {
        "id": "msg_900001", "type": "message", "role": "assistant",
        "model": "claude-opus-5-5", "content": content,
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }
    if stop_details is not None:
        message["stop_details"] = stop_details
    return message


def _refusal(category="reasoning_extraction"):
    """A declined reply, as the API returns it: HTTP 200, stop_reason "refusal"."""
    return _message("", stop_reason="refusal", input_tokens=900, output_tokens=7, stop_details={
        "type": "refusal", "category": category, "explanation": "EXPLANATION-MARKER free text"})


class _Transport:
    """Records each request and answers from a queue of (status, json, headers)."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        status, body, headers = self.responses.pop(0)
        return httpx2.Response(status, json=body, headers=headers or {})

    def bodies(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests]


def _sdk_client(transport: _Transport, **kw) -> anthropic.AsyncAnthropic:
    return compare.build_anthropic_client(
        SENTINEL_KEY,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(transport)),
        **kw,
    )


class TestAnthropicLLMClient:
    async def test_returns_stripped_answer_and_inline_thinking(self):
        transport = _Transport((200, _message(), None))
        llm = compare.AnthropicLLMClient(_sdk_client(transport), model="claude-opus-5-5")

        answer, thinking = await llm.complete("SYSTEM PROMPT", "USER CONTENT", include_thinking=True)

        assert answer == "NEEDS_RESPONSE"
        assert thinking == "it asks for a date"

    async def test_request_carries_prompt_model_budget_and_effort_but_no_sampling(self):
        transport = _Transport((200, _message(), None))
        llm = compare.AnthropicLLMClient(_sdk_client(transport), model="claude-haiku-5-5")

        await llm.complete("SYSTEM PROMPT", "USER CONTENT")

        body = transport.bodies()[0]
        assert body["model"] == "claude-haiku-5-5"
        assert body["system"] == "SYSTEM PROMPT"
        assert body["messages"] == [{"role": "user", "content": "USER CONTENT"}]
        assert body["max_tokens"] == compare.ANTHROPIC_MAX_TOKENS
        assert body["output_config"] == {"effort": compare.ANTHROPIC_EFFORT}
        # Per the claude-api skill's model notes, current models reject a
        # temperature and Opus 5.5 rejects disabled thinking, so thinking is
        # left at its adaptive default; fallbacks would change which model
        # answered, which an eval must not do.
        for absent in ("temperature", "thinking", "fallbacks"):
            assert absent not in body

    @pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal"])
    async def test_truncated_or_refused_reply_is_an_error_not_an_answer(self, stop_reason):
        # A label read out of a cut-off or declined reply is an artifact, so it
        # must count as an arm error (LLMContentError, as llm_client raises on
        # finish_reason "length"), never as a prediction.
        transport = _Transport((200, _message("FYI", stop_reason=stop_reason), None))
        llm = compare.AnthropicLLMClient(_sdk_client(transport), model="claude-opus-5-5")

        with pytest.raises(LLMContentError, match=stop_reason):
            await llm.complete("S", "U", include_thinking=True)

    @pytest.mark.parametrize("reply, kind", [
        (_refusal("reasoning_extraction"), "refusal: reasoning_extraction"),
        (_refusal(None), "refusal: uncategorised"),
        (_message("FYI", stop_reason="refusal"), "refusal: uncategorised"),  # no stop_details
        (_message("FY", stop_reason="max_tokens"), "max_tokens"),
    ])
    async def test_an_unusable_reply_records_its_kind(self, reply, kind):
        # The row keeps only the error text, so the refusal's category (the
        # prompt's <think> instruction is a documented reasoning_extraction
        # trigger) must be in it. The explanation is free text and stays out.
        transport = _Transport((200, reply, None))
        llm = compare.AnthropicLLMClient(_sdk_client(transport), model="claude-opus-5-5")

        with pytest.raises(LLMContentError) as raised:
            await llm.complete("S", "U", include_thinking=True)

        assert compare.error_kind(str(raised.value)) == kind
        assert "EXPLANATION-MARKER" not in str(raised.value)

    @pytest.mark.parametrize("error, kind", [
        ("[refusal: reasoning_extraction] synthetic", "refusal: reasoning_extraction"),
        ("[not sent] stopped after refusals", "not sent"),
        ("[Errno 61] Connection refused", "other"),  # an OS error's own brackets
        ("No email label keyword in LLM output: synthetic", "other"),
        (None, None),
    ])
    def test_error_kind_reads_only_the_comparisons_own_tags(self, error, kind):
        assert compare.error_kind(error) == kind

    async def test_a_timeout_records_its_kind(self):
        def no_reply(request):
            raise httpx2.ReadTimeout("synthetic read timeout", request=request)

        http_client = anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(no_reply))
        # No retries, so the test does not sit through the SDK's backoff.
        client = compare.build_anthropic_client(SENTINEL_KEY, http_client=http_client)
        llm = compare.AnthropicLLMClient(client.with_options(max_retries=0), model="claude-opus-5-5")

        with pytest.raises(TimeoutError) as raised:
            await llm.complete("S", "U", include_thinking=True)

        assert compare.error_kind(str(raised.value)) == "timeout"

    @pytest.mark.parametrize("status, error_type", [
        (529, "overloaded_error"), (429, "rate_limit_error"), (500, "api_error")])
    async def test_an_api_outage_after_retries_records_its_kind(self, status, error_type):
        # An exhausted 429 or a 5xx means Anthropic could not serve anyone, as
        # LLMUnavailableError means for the local model: an outage, not a miss.
        transport = _Transport((status, _error(error_type), None))
        llm = compare.AnthropicLLMClient(_sdk_client(transport).with_options(max_retries=0),
                                         model="claude-opus-5-5")

        with pytest.raises(Exception) as raised:
            await llm.complete("S", "U", include_thinking=True)

        assert compare.error_kind(str(raised.value)) == "unavailable"

    @pytest.mark.parametrize("failure", [httpx2.ConnectError, httpx2.ConnectTimeout])
    async def test_a_connection_failure_records_its_kind(self, failure):
        # A connect timeout is an unreachable endpoint, as llm_client treats it
        # for the local model, not a slow reply.
        def unreachable(request):
            raise failure("synthetic connection failure", request=request)

        http_client = anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(unreachable))
        client = compare.build_anthropic_client(SENTINEL_KEY, http_client=http_client)
        llm = compare.AnthropicLLMClient(client.with_options(max_retries=0), model="claude-opus-5-5")

        with pytest.raises(Exception) as raised:
            await llm.complete("S", "U", include_thinking=True)

        assert compare.error_kind(str(raised.value)) == "unavailable"

    async def test_a_rejected_request_is_not_an_outage(self):
        transport = _Transport((400, _error("invalid_request_error"), None))
        llm = compare.AnthropicLLMClient(_sdk_client(transport), model="claude-opus-5-5")

        with pytest.raises(anthropic.BadRequestError) as raised:
            await llm.complete("S", "U", include_thinking=True)

        assert compare.error_kind(str(raised.value)) == "other"

    async def test_reply_with_only_reasoning_is_an_error(self):
        transport = _Transport((200, _message("<think>hmm</think>", thinking="native"), None))
        llm = compare.AnthropicLLMClient(_sdk_client(transport), model="claude-opus-5-5")

        with pytest.raises(LLMContentError):
            await llm.complete("S", "U", include_thinking=True)

    async def test_native_thinking_text_is_captured_with_inline_thinking(self):
        transport = _Transport((200, _message("<think>inline</think>FYI", thinking="native"), None))
        llm = compare.AnthropicLLMClient(_sdk_client(transport), model="claude-opus-5-5")

        answer, thinking = await llm.complete("S", "U", include_thinking=True)

        assert answer == "FYI"
        assert thinking == "native\n\ninline"


class TestRefusalEarlyStop:
    @pytest.mark.parametrize("refused, answered, next_is_sent", [
        (5, 0, False),  # five replies, all refused
        (3, 2, False),  # five replies, more than half refused
        (4, 0, True),   # all refused, but fewer than five replies
        (3, 3, True),   # six replies, exactly half refused
    ])
    async def test_arm_b_stops_once_five_replies_are_in_and_most_are_refusals(
        self, refused, answered, next_is_sent,
    ):
        # Answers first, so the rule is checked on the full count each time.
        transport = _Transport(*[(200, _message("FYI"), None)] * answered,
                               *[(200, _refusal(), None)] * refused, (200, _message("FYI"), None))
        llm = compare.AnthropicLLMClient(_sdk_client(transport), model="claude-opus-5-5")
        for _ in range(answered):
            await llm.complete("S", "U")
        for _ in range(refused):
            with pytest.raises(LLMContentError):
                await llm.complete("S", "U")

        if next_is_sent:
            assert await llm.complete("S", "U") == "FYI"
            assert len(transport.requests) == refused + answered + 1
        else:
            with pytest.raises(Exception) as raised:
                await llm.complete("S", "U")
            assert compare.error_kind(str(raised.value)) == "not sent"
            assert len(transport.requests) == refused + answered


    async def test_once_stopped_arm_b_stays_stopped_when_answers_in_flight_land(self):
        # A request in flight when the rule trips still lands. If it is an
        # answer, the ratio falls back to half (3 of 6), but nothing more may
        # be sent.
        arrived, release = asyncio.Event(), asyncio.Event()
        queue = [_refusal(), _refusal(), _refusal(), _message("FYI"), _message("FYI"), _message("FYI")]
        seen = []

        async def handler(request):
            seen.append(request)
            if len(seen) == 1:  # held in flight until after the stop
                arrived.set()
                await release.wait()
                return httpx2.Response(200, json=_message("FYI"))
            return httpx2.Response(200, json=queue.pop(0))

        http_client = anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler))
        llm = compare.AnthropicLLMClient(
            compare.build_anthropic_client(SENTINEL_KEY, http_client=http_client), model="claude-opus-5-5")
        held = asyncio.create_task(llm.complete("S", "U"))
        await arrived.wait()
        for _ in range(3):
            with pytest.raises(LLMContentError):
                await llm.complete("S", "U")
        for _ in range(2):
            await llm.complete("S", "U")
        with pytest.raises(compare.NotSentError):
            await llm.complete("S", "U")

        release.set()
        assert await held == "FYI"
        with pytest.raises(compare.NotSentError):
            await llm.complete("S", "U")
        assert len(seen) == 6


def _error(kind):
    return {"type": "error", "error": {"type": kind, "message": "synthetic"}}


class TestBuildAnthropicClient:
    async def test_uses_only_the_given_key_whatever_the_environment_holds(self, monkeypatch):
        # daemon.py loads .env at import, and the eval README symlinks .env to
        # agent-stack's, so every variable the SDK would read is set here.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env-file")
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "bearer-from-env-file")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example.test")
        monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Leak: from-env")
        transport = _Transport((200, _message(), None))
        llm = compare.AnthropicLLMClient(_sdk_client(transport), model="claude-opus-5-5")

        await llm.complete("S", "U")

        request = transport.requests[0]
        # SDK 2.x sends an API key as a bearer credential.
        assert request.headers.get_list("authorization") == [f"Bearer {SENTINEL_KEY}"]
        assert "x-api-key" not in request.headers
        assert "x-leak" not in request.headers
        assert request.url.host == "api.anthropic.com"

    async def test_rate_limit_and_overload_replies_are_retried(self):
        # Three retries: one more than the SDK's default budget of two, so this
        # passes only with the builder's own retry budget.
        transport = _Transport(
            (429, _error("rate_limit_error"), {"retry-after-ms": "1"}),
            (529, _error("overloaded_error"), {"retry-after-ms": "1"}),
            (429, _error("rate_limit_error"), {"retry-after-ms": "1"}),
            (200, _message("FYI"), None),
        )
        llm = compare.AnthropicLLMClient(_sdk_client(transport), model="claude-opus-5-5")

        assert await llm.complete("S", "U") == "FYI"
        assert len(transport.requests) == 4

    async def test_retry_budget_and_per_request_timeout_are_set(self):
        transport = _Transport((200, _message(), None))
        client = _sdk_client(transport)

        await compare.AnthropicLLMClient(client, model="claude-opus-5-5").complete("S", "U")

        assert client.max_retries == compare.ANTHROPIC_MAX_RETRIES
        sent = transport.requests[0].extensions["timeout"]
        # Each attempt gets the SDK's own non-streaming read timeout, which covers
        # the SDK's expected time for a reply that runs to max_tokens (3600 s x
        # max_tokens / 128,000). A shorter one cuts a slow reply and retries it,
        # and no usage is counted for the attempts it cut.
        assert sent["read"] == anthropic.DEFAULT_TIMEOUT.read == 600
        assert sent["read"] >= 3600 * compare.ANTHROPIC_MAX_TOKENS / 128_000
        # Connecting keeps the SDK's own 5 s: a blackholed endpoint must not
        # hold each attempt for the read budget.
        assert sent["connect"] == anthropic.DEFAULT_TIMEOUT.connect == 5.0


class TestUsageAndCost:
    async def test_usage_counts_every_billed_reply_including_unusable_ones(self):
        transport = _Transport(
            (200, _message("FYI", input_tokens=1000, output_tokens=200), None),
            (200, _message("FY", stop_reason="max_tokens", input_tokens=3000, output_tokens=16000), None),
        )
        llm = compare.AnthropicLLMClient(_sdk_client(transport), model="claude-opus-5-5")

        await llm.complete("S", "U")
        with pytest.raises(LLMContentError):
            await llm.complete("S", "U")

        assert (llm.calls, llm.input_tokens, llm.output_tokens) == (2, 4000, 16200)

    def test_prices_cover_the_selectable_models(self):
        assert set(compare.PRICES_PER_MTOK) == {
            "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-2",
        }
        assert compare.PRICES_AS_OF == "2026-11-03"

    @pytest.mark.parametrize("model, expected", [
        ("claude-opus-5-5", 4.00 + 20.00),
        ("claude-haiku-5-5", 0.10 + 0.50),
        ("claude-sonnet-5-5", 2.00 + 10.00),
        ("claude-fable-5-2", 8.00 + 40.00),
    ])
    def test_cost_is_tokens_times_published_price(self, model, expected):
        assert compare.cost_usd(model, 1_000_000, 1_000_000) == pytest.approx(expected)


class TestObtainKey:
    def test_key_file_is_read_and_stripped(self, tmp_path):
        key_file = tmp_path / "key"
        key_file.write_text(SENTINEL_KEY + "\n")
        key_file.chmod(0o600)

        assert compare.obtain_key(key_file, getpass_fn=_no_prompt) == SENTINEL_KEY

    @pytest.mark.parametrize("mode", [0o640, 0o604, 0o644])
    def test_key_file_readable_by_others_is_refused(self, tmp_path, mode):
        key_file = tmp_path / "key"
        key_file.write_text(SENTINEL_KEY)
        key_file.chmod(mode)

        with pytest.raises(compare.ApiKeyError, match="chmod 600"):
            compare.obtain_key(key_file, getpass_fn=_no_prompt)

    def test_empty_key_file_is_refused(self, tmp_path):
        key_file = tmp_path / "key"
        key_file.write_text("\n")
        key_file.chmod(0o600)

        with pytest.raises(compare.ApiKeyError, match="empty"):
            compare.obtain_key(key_file, getpass_fn=_no_prompt)

    def test_missing_key_file_is_refused_without_a_traceback(self, tmp_path):
        with pytest.raises(compare.ApiKeyError, match="cannot read"):
            compare.obtain_key(tmp_path / "absent", getpass_fn=_no_prompt)

    def test_without_a_key_file_the_key_is_typed_not_taken_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env-file")
        prompts = []

        def fake_getpass(prompt):
            prompts.append(prompt)
            return f"  {SENTINEL_KEY}  "

        assert compare.obtain_key(None, getpass_fn=fake_getpass) == SENTINEL_KEY
        assert len(prompts) == 1

    def test_an_empty_typed_key_is_refused(self):
        with pytest.raises(compare.ApiKeyError, match="empty"):
            compare.obtain_key(None, getpass_fn=lambda prompt: "   ")

    def test_a_prompt_that_cannot_hide_the_key_is_refused_before_it_is_read(self):
        # Without a terminal (ssh host 'cmd' without -t), getpass warns with
        # GetPassWarning and then reads the key with echo on, so it can land in
        # scrollback. The warning must stop the read and point to the key file.
        read = []

        def no_terminal_getpass(prompt):
            warnings.warn("Can not control echo on the terminal.", getpass.GetPassWarning, stacklevel=2)
            read.append(prompt)
            return SENTINEL_KEY

        with pytest.raises(compare.ApiKeyError, match="--anthropic-key-file"):
            compare.obtain_key(None, getpass_fn=no_terminal_getpass)
        assert read == []

    def test_getpass_fallback_leaves_the_key_unread(self, monkeypatch):
        # The standard library's own no-terminal path, reading from stdin.
        stdin = io.StringIO(SENTINEL_KEY + "\n")
        monkeypatch.setattr(sys, "stdin", stdin)
        fallback = functools.partial(getpass.fallback_getpass, stream=io.StringIO())

        with pytest.raises(compare.ApiKeyError, match="--anthropic-key-file"):
            compare.obtain_key(None, getpass_fn=fallback)
        assert stdin.read() == SENTINEL_KEY + "\n"


def _no_prompt(prompt):
    raise AssertionError("getpass must not be called when a key file is given")


def _gmail_message(sender, subject, body, internal_date="1700000000000"):
    data = base64.urlsafe_b64encode(body.encode()).decode()
    return {
        "id": "m" + internal_date, "threadId": "t", "internalDate": internal_date,
        "snippet": body[:40],
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "From", "value": sender},
                {"name": "Subject", "value": subject},
                {"name": "Date", "value": "Mon, 6 Nov 2023 12:00:00 +0000"},
            ],
            "body": {"data": data, "size": len(body)},
        },
    }


def _thread(thread_id, sender_type="person", label="fyi", *, subject=None, body=None,
            sender="Pat Example <pat@example.test>", reviewed=True):
    subject = subject or f"Synthetic subject {thread_id}"
    body = body or f"Synthetic body text for {thread_id}."
    return GoldenThread(
        thread_id=thread_id, messages=[_gmail_message(sender, subject, body)],
        senders=[sender], subject=subject, snippet=body[:40],
        expected_sender_type=sender_type, expected_label=label, reviewed=reviewed,
    )


class TestCapturePrompts:
    async def test_captures_the_classifier_prompt_for_each_thread(self):
        config = load_config()
        threads = [
            _thread("900001", subject="Lunch on Thursday?", body="Could you join us Thursday?"),
            _thread("900002", subject="Minutes attached", body="Notes from the meeting."),
        ]

        prompts = await compare.capture_prompts(threads, config)

        categories = list(config["prompts"]["email_classification"]["categories"])
        expected_system = EmailClassifier(None, None, config)._build_email_prompt(categories)
        assert [system for system, _ in prompts] == [expected_system, expected_system]
        assert "Lunch on Thursday?" in prompts[0][1]
        assert "Could you join us Thursday?" in prompts[0][1]
        assert "pat@example.test" in prompts[0][1]
        assert "Notes from the meeting." in prompts[1][1]


class TestEstimate:
    def test_estimate_from_prompt_characters_and_assumed_output(self):
        prompts = [("s" * 1000, "u" * 750), ("s" * 1000, "u" * 750)]  # 3500 chars

        est = compare.estimate("claude-opus-5-5", prompts)

        assert est.calls == 2
        assert est.input_chars == 3500
        assert est.input_tokens == round(3500 / compare.CHARS_PER_TOKEN)
        assert est.output_tokens == 2 * compare.ASSUMED_OUTPUT_TOKENS_PER_CALL
        assert est.cost == pytest.approx(
            compare.cost_usd("claude-opus-5-5", est.input_tokens, est.output_tokens))
        assert est.cost_at_max_tokens == pytest.approx(
            compare.cost_usd("claude-opus-5-5", est.input_tokens, 2 * compare.ANTHROPIC_MAX_TOKENS))


def _plan(**kw):
    prompts = [("s" * 2000, "u" * 1500)] * 5
    base = dict(
        golden_set_path="evals/golden_set.jsonl", reviewed_threads=12, person_threads=5,
        max_thread_chars=16000, local_model="qwen-synthetic:8b",
        local_url="http://ollama.example.test:11434/v1/chat/completions", local_error=None,
        local_stand_in=False,
        local_settings="temperature 0, max_tokens 4096, timeout 180 s per request, extra_body {}",
        model="claude-opus-5-5", estimate=compare.estimate("claude-opus-5-5", prompts),
    )
    base.update(kw)
    return compare.Plan(**base)


class TestPlan:
    def test_states_sets_models_calls_tokens_and_cost(self):
        text = compare.format_plan(_plan())

        assert "evals/golden_set.jsonl" in text
        assert "12 reviewed threads" in text and "5 person threads" in text
        assert "qwen-synthetic:8b" in text and "ollama.example.test" in text
        assert "reachable" in text
        assert "claude-opus-5-5" in text
        assert f"effort {compare.ANTHROPIC_EFFORT}" in text
        assert "5 message requests" in text
        assert "1 model lookup that sends no mail" in text
        assert "5,000 input tokens" in text  # 17,500 chars at 3.5 chars/token
        assert "5,000 output tokens" in text  # 1,000 assumed per call
        assert "$0.12" in text  # 5,000 x $4 + 5,000 x $20 per million
        assert "$4.00 / $20.00 per million" in text and "2026-11-03" in text
        assert "estimate" in text.lower()

    def test_cost_figures_are_estimates_that_name_what_they_leave_out(self):
        # The max_tokens figure prices an estimated input count and one attempt
        # per thread, so it is not a ceiling and must not be worded as one.
        text = compare.format_plan(_plan())

        assert "at most" not in text
        assert "if every reply ran to max_tokens" in text
        assert "input tokens are estimated" in text
        assert "retried attempts are not counted" in text

    def test_states_the_refusal_early_stop(self):
        text = compare.format_plan(_plan())

        assert ("stops sending once at least 5 replies are in and more than half of them are "
                "refusals") in text

    def test_says_plainly_that_real_mail_goes_to_anthropic(self):
        text = compare.format_plan(_plan())

        assert "The golden sets are real mail, and arm B sends them to Anthropic" in text
        assert ("inputs are deleted within 30 days, and zero retention is available "
                "under an agreement") in text

    def test_privacy_names_arm_a_destination_host(self):
        text = compare.format_plan(_plan())
        privacy = text[text.index("Privacy:"):]

        assert "Arm A sends the same threads to ollama.example.test" in privacy
        assert "D4" not in privacy

    def test_privacy_flags_an_api_stand_in_for_arm_a(self):
        text = compare.format_plan(_plan(
            local_url="https://standin.example.test/v1/chat/completions", local_stand_in=True))
        privacy = text[text.index("Privacy:"):]

        assert "Arm A sends the same threads to standin.example.test" in privacy
        assert "API stand-in (decision D4)" in privacy

    def test_an_unreachable_local_model_is_stated_and_the_run_refused(self):
        text = compare.format_plan(_plan(local_error="ConnectError: connection refused"))

        assert "NOT reachable" in text
        assert "ConnectError: connection refused" in text
        assert "will not start" in text


class TestConfirmation:
    @pytest.mark.parametrize("answer", ["yes", "YES", "  yes\n"])
    def test_yes_proceeds(self, answer):
        assert compare.confirmed(answer)

    @pytest.mark.parametrize("answer", ["", "y", "no", "yes please", "sure"])
    def test_anything_else_stops(self, answer):
        assert not compare.confirmed(answer)


def _pred(tid, expected, predicted, seconds=1.0, error=None):
    return PredictionResult(
        thread_id=tid, expected_sender_type="person", expected_label=expected,
        predicted_sender_type="person", predicted_label=None if error else predicted,
        label_correct=None if error else expected == predicted,
        duration_seconds=seconds, error=error,
    )


def _arms():
    nr, fyi, low = "needs_response", "fyi", "low_priority"
    arm_a = compare.ArmRun(name="Arm A: local qwen-synthetic:8b", parallelism=1, results=[
        _pred("900001", nr, low, 1.0),   # person mail archived
        _pred("900002", fyi, low, 2.0),  # person mail archived
        _pred("900003", nr, fyi, 3.0),   # needs_response read as FYI
        _pred("900004", fyi, nr, 4.0),   # FYI read as needs_response
        _pred("900005", low, fyi, 5.0),  # cold pitch kept
        _pred("900006", nr, nr, 6.0),
        _pred("900007", nr, None, 9.0, error="LLM endpoint unavailable"),
        _pred("900008", fyi, fyi, 7.0),
    ])
    arm_b = compare.ArmRun(
        name="Arm B: claude-opus-5-5", parallelism=4, priced_model="claude-opus-5-5",
        input_tokens=10_000, output_tokens=2_000, results=[
            _pred("900001", nr, nr, 0.5), _pred("900002", fyi, fyi, 0.5),
            _pred("900003", nr, nr, 0.5), _pred("900004", fyi, fyi, 0.5),
            _pred("900005", low, low, 0.5), _pred("900006", nr, nr, 0.5),
            _pred("900007", nr, nr, 0.5),
            _pred("900008", fyi, None, 0.5, error="stop_reason='refusal'"),
        ])
    return arm_a, arm_b


def _report():
    arm_a, arm_b = _arms()
    return compare.build_report(
        arm_a, arm_b, golden_set_path="evals/golden_set.jsonl", reviewed_threads=20,
        timestamp="2026-11-13T12:00:00+00:00",
    )


def _rows(report):
    rows = {}
    order = []
    for line in report.splitlines():
        if line.startswith("| ") and not line.startswith("| Metric") and not line.startswith("|---"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            rows[cells[0]] = cells[1:]
            order.append(cells[0])
    return rows, order


class TestReport:
    def test_scores_both_arms_on_the_threads_both_were_sent(self):
        rows, _ = _rows(_report())

        # 900007 failed in arm A and 900008 in arm B; each stays in its arm's
        # figures as a miss, so all eight threads are scored for both arms.
        assert rows["needs_response recall"] == ["25.0% (1/4)", "100.0% (4/4)"]
        assert rows["Errors (no usable answer)"] == ["1/8 (needs_response 1)", "1/8 (fyi 1)"]
        assert "Scored on the 8 threads both arms were sent" in _report()

    def test_needs_response_recall_leads_and_error_classes_follow_the_cost_order(self):
        rows, order = _rows(_report())

        assert order[0] == "needs_response recall"
        cost_order = [
            "1. Person mail archived (needs_response/fyi → low_priority)",
            "2. needs_response read as FYI",
            "3. FYI read as needs_response",
            "4. Cold pitch kept (low_priority → needs_response/fyi)",
        ]
        assert [label for label in order if label[:2] in ("1.", "2.", "3.", "4.")] == cost_order
        assert [rows[label][0] for label in cost_order] == ["2", "1", "1", "1"]
        assert [rows[label][1] for label in cost_order] == ["0", "0", "0", "0"]

    def test_per_category_precision_and_recall(self):
        rows, _ = _rows(_report())

        assert rows["needs_response precision"] == ["50.0% (1/2)", "100.0% (4/4)"]
        assert rows["fyi recall"] == ["33.3% (1/3)", "66.7% (2/3)"]
        assert rows["fyi precision"] == ["33.3% (1/3)", "100.0% (2/2)"]
        assert rows["low_priority recall"] == ["0.0% (0/1)", "100.0% (1/1)"]
        assert rows["low_priority precision"] == ["0.0% (0/2)", "100.0% (1/1)"]

    def test_latency_median_and_p90_per_answered_message(self):
        rows, _ = _rows(_report())

        # Arm A answered seven threads in 1..7 s; its failed 900007 (9 s) is
        # not an answer, so it does not count toward latency.
        assert rows["Latency per message, median"] == ["4.0 s", "0.5 s"]
        assert rows["Latency per message, p90"] == ["7.0 s", "0.5 s"]

    def test_cost_per_message_and_total(self):
        rows, _ = _rows(_report())

        # 10,000 x $4 + 2,000 x $20 per million = $0.08 over 8 threads sent.
        assert rows["Cost per message"] == ["$0 (local)", "$0.0100"]
        assert rows["Cost, total"][0] == "$0 (local)"
        assert rows["Cost, total"][1].startswith("$0.08")
        assert "2026-11-03" in rows["Cost, total"][1]

    def test_an_arms_error_counts_as_a_miss_for_its_expected_label(self):
        # In production a failed thread is deferred, so its action is late: an
        # error on a needs_response thread must lower that arm's recall, not
        # drop the thread from both arms' figures.
        nr, fyi = "needs_response", "fyi"
        arm_a = compare.ArmRun(name="Arm A", parallelism=1, results=[
            _pred("900401", nr, nr), _pred("900402", nr, nr), _pred("900403", fyi, fyi)])
        arm_b = compare.ArmRun(name="Arm B", parallelism=4, priced_model="claude-opus-5-5", results=[
            _pred("900401", nr, nr), _pred("900402", nr, None, error="synthetic failure"),
            _pred("900403", fyi, fyi)])

        text = compare.build_report(arm_a, arm_b, golden_set_path="evals/golden_set.jsonl",
                                    reviewed_threads=3, timestamp="2026-11-13T12:00:00+00:00")

        rows, _ = _rows(text)
        assert rows["needs_response recall"] == ["100.0% (2/2)", "50.0% (1/2)"]
        assert rows["Errors (no usable answer)"] == ["0/3", "1/3 (needs_response 1)"]
        assert "Scored on the 3 threads both arms were sent" in text

    def test_threads_not_sent_are_neither_errors_nor_billed_messages(self):
        nr, fyi = "needs_response", "fyi"
        arm_a = compare.ArmRun(name="Arm A", parallelism=1, results=[
            _pred("900501", nr, nr), _pred("900502", fyi, fyi), _pred("900503", nr, nr)])
        arm_b = compare.ArmRun(
            name="Arm B", parallelism=4, priced_model="claude-opus-5-5",
            input_tokens=10_000, output_tokens=2_000, results=[
                _pred("900501", nr, nr), _pred("900502", fyi, fyi),
                _pred("900503", nr, None, error=compare.NOT_SENT_LOCAL_UNAVAILABLE)])

        rows, _ = _rows(compare.build_report(
            arm_a, arm_b, golden_set_path="evals/golden_set.jsonl", reviewed_threads=3,
            timestamp="2026-11-13T12:00:00+00:00"))

        assert rows["Errors (no usable answer)"] == ["0/2", "0/2"]
        assert rows["needs_response recall"] == ["100.0% (1/1)", "100.0% (1/1)"]
        assert rows["Threads not sent"] == ["0", "1 (local model unavailable 1)"]
        # $0.08 over the two messages sent, not over three threads.
        assert rows["Cost per message"][1] == "$0.0400"

    def test_a_thread_either_arm_lost_to_an_outage_is_scored_for_neither(self):
        nr = "needs_response"
        arm_a = compare.ArmRun(name="Arm A", parallelism=1, results=[
            _pred("900521", nr, nr),
            _pred("900522", nr, None, error="[unavailable] synthetic local outage"),
            _pred("900523", nr, nr)])
        arm_b = compare.ArmRun(name="Arm B", parallelism=4, priced_model="claude-opus-5-5", results=[
            _pred("900521", nr, None, error="[unavailable] synthetic API overload"),
            _pred("900522", nr, None, error=compare.NOT_SENT_LOCAL_UNAVAILABLE),
            _pred("900523", nr, nr)])

        text = compare.build_report(arm_a, arm_b, golden_set_path="evals/golden_set.jsonl",
                                    reviewed_threads=3, timestamp="2026-11-13T12:00:00+00:00")

        rows, _ = _rows(text)
        assert rows["needs_response recall"] == ["100.0% (1/1)", "100.0% (1/1)"]
        assert rows["Errors (no usable answer)"] == ["0/1", "0/1"]
        assert rows["Lost to an outage"] == ["1", "1"]
        assert "Scored on the 1 threads both arms were sent and neither lost to an outage" in text

    def test_the_header_says_arm_b_ran_only_the_threads_it_was_sent(self):
        nr, fyi = "needs_response", "fyi"
        arm_a = compare.ArmRun(name="Arm A", parallelism=1, results=[
            _pred("900511", nr, nr), _pred("900512", fyi, fyi), _pred("900513", nr, nr)])
        arm_b = compare.ArmRun(name="Arm B", parallelism=4, priced_model="claude-opus-5-5", results=[
            _pred("900511", nr, nr), _pred("900512", fyi, fyi),
            _pred("900513", nr, None, error=compare.NOT_SENT_REFUSALS)])

        text = compare.build_report(arm_a, arm_b, golden_set_path="evals/golden_set.jsonl",
                                    reviewed_threads=3, timestamp="2026-11-13T12:00:00+00:00")

        assert "arm A ran Stage 2 on all 3 of them and arm B on the 2 it was sent" in text
        assert "both arms ran" not in text

    def test_errors_are_counted_by_kind_and_the_kinds_sum_to_the_errors(self):
        nr, fyi = "needs_response", "fyi"
        unparseable = "No email label keyword in LLM output: synthetic"
        arm_a = compare.ArmRun(name="Arm A", parallelism=1, results=[
            *[_pred(f"90070{i}", nr, nr) for i in range(5)], _pred("900705", fyi, None, error=unparseable)])
        arm_b = compare.ArmRun(name="Arm B", parallelism=4, priced_model="claude-opus-5-5", results=[
            _pred("900700", nr, None, error="[refusal: reasoning_extraction] synthetic"),
            _pred("900701", nr, None, error="[refusal: reasoning_extraction] synthetic"),
            _pred("900702", nr, None, error="[max_tokens] synthetic"),
            _pred("900703", nr, None, error="[timeout] synthetic"),
            _pred("900704", nr, nr),
            _pred("900705", fyi, None, error=unparseable),
        ])

        rows, _ = _rows(compare.build_report(
            arm_a, arm_b, golden_set_path="evals/golden_set.jsonl", reviewed_threads=6,
            timestamp="2026-11-13T12:00:00+00:00"))

        assert rows["Errors by kind"] == [
            "other 1", "max_tokens 1, refusal: reasoning_extraction 2, timeout 1, other 1"]
        for errors, kinds in zip(rows["Errors (no usable answer)"], rows["Errors by kind"]):
            assert int(errors.split("/")[0]) == sum(int(kind.rsplit(" ", 1)[1]) for kind in kinds.split(", "))

    @pytest.mark.parametrize("replies, refusals, marked", [(3, 3, True), (6, 4, True), (6, 3, False)])
    def test_a_refusal_dominated_arm_b_is_marked_not_meaningful_above_the_table(
        self, replies, refusals, marked,
    ):
        arm_a, arm_b = _arms()
        arm_b.replies, arm_b.refusals = replies, refusals

        text = compare.build_report(arm_a, arm_b, golden_set_path="evals/golden_set.jsonl",
                                    reviewed_threads=20, timestamp="2026-11-13T12:00:00+00:00")

        if marked:
            assert text.index("Not a meaningful comparison") < text.index("| Metric")
            assert f"{refusals} of its {replies} replies" in text
        else:
            assert "Not a meaningful comparison" not in text

    def test_holds_counts_only_no_thread_ids(self):
        report = _report()

        for tid in ("900001", "900005", "900008"):
            assert tid not in report


class TestPercentile:
    @pytest.mark.parametrize("values, expected", [
        ([5.0], 5.0),
        ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 9.0),
        ([3.0, 1.0, 2.0], 3.0),
    ])
    def test_p90_is_nearest_rank(self, values, expected):
        assert compare.percentile(values, 90) == expected


class _FakeLocal:
    """The local slot for orchestration tests: a plain class, as evaluate_single
    probes its clients with hasattr."""

    model = "qwen-synthetic:8b"
    base_url = "http://ollama.example.test:11434/v1/chat/completions"

    def __init__(self, *, reachable=True, fail=False, outage_on=None, content_failure_on=None):
        self.reachable = reachable
        self.fail = fail
        # Fail only the thread whose prompt holds this text: an outage
        # (LLMUnavailableError) or an unusable reply (LLMContentError).
        self.outage_on = outage_on
        self.content_failure_on = content_failure_on
        self.calls = 0
        self.probes = 0

    async def probe(self, timeout=None):
        self.probes += 1
        if self.reachable:
            return AvailabilityResult(ok=True, status_code=200)
        return AvailabilityResult(ok=False, error="ConnectError: connection refused")

    async def complete(self, system_prompt, user_content, include_thinking=False):
        self.calls += 1
        if self.fail or (self.outage_on and self.outage_on in user_content):
            raise LLMUnavailableError("synthetic outage", tier="local")
        if self.content_failure_on and self.content_failure_on in user_content:
            # Worded to catch a rule that matches on text instead of type.
            raise LLMContentError("synthetic unusable reply; endpoint unavailable wording")
        return ("FYI", "") if include_thinking else "FYI"


class _Router:
    """Answers the SDK by path: the model lookup, then message requests."""

    def __init__(self, model_status=200, reply=None, fail_on=None, failure=None):
        self.model_status = model_status
        self.reply = reply or _message("<think>reasoning</think>\nNEEDS_RESPONSE")
        # The message request whose body holds fail_on gets failure: an HTTP
        # status, or an exception class the transport raises.
        self.fail_on = fail_on
        self.failure = failure
        self.requests: list[httpx2.Request] = []

    def __call__(self, request):
        self.requests.append(request)
        if request.url.path.startswith("/v1/models/"):
            if self.model_status != 200:
                return httpx2.Response(self.model_status, json=_error("authentication_error"))
            return httpx2.Response(200, json={
                "type": "model", "id": "claude-opus-5-5", "display_name": "Claude Opus 5.5",
                "created_at": "2026-01-01T00:00:00Z",
            })
        if self.fail_on and self.fail_on in request.content.decode():
            if isinstance(self.failure, int):
                return httpx2.Response(self.failure, json=_error("overloaded_error"))
            raise self.failure("synthetic failure", request=request)
        return httpx2.Response(200, json=self.reply)

    def message_requests(self):
        return [r for r in self.requests if r.url.path == "/v1/messages"]


def _never(*args, **kwargs):
    raise AssertionError("must not be called on this path")


@pytest.fixture
def golden_file(tmp_path):
    path = tmp_path / "golden.jsonl"
    threads = [
        _thread("900101", label="needs_response", subject="Synthetic subject alpha",
                body="Synthetic body alpha, can you reply?"),
        _thread("900102", label="fyi", subject="Synthetic subject beta", body="Synthetic body beta."),
        _thread("900103", label="low_priority", subject="Synthetic subject gamma",
                body="Synthetic body gamma, a pitch."),
        _thread("900104", sender_type="service", label="low_priority",
                sender="Example Shop <news@shop.example.test>"),
        _thread("900105", reviewed=False),
    ]
    path.write_text("".join(json.dumps(t.to_dict()) + "\n" for t in threads))
    return path


def _args(golden_file, tmp_path, **kw):
    base = dict(
        golden_set=str(golden_file), config=None, output_dir=str(tmp_path / "results"),
        model="claude-opus-5-5", anthropic_key_file=None, dry_run=False, max_threads=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def _factory(router, keys):
    def make(key):
        keys.append(key)
        return _sdk_client(router)
    return make


class TestRunComparison:
    async def test_dry_run_states_the_plan_and_sends_nothing(self, golden_file, tmp_path, capsys):
        local = _FakeLocal()

        code = await compare.run_comparison(
            _args(golden_file, tmp_path, dry_run=True), local_llm=local,
            client_factory=_never, getpass_fn=_never, input_fn=_never,
        )

        out = capsys.readouterr().out
        assert code == 0
        assert "3 person threads" in out
        assert "The golden sets are real mail, and arm B sends them to Anthropic" in out
        assert "reachable" in out
        assert local.calls == 0
        # The probe is the one request a dry run makes, and it carries no mail.
        assert local.probes == 1
        assert "Dry run: no mail was sent; the local model received one probe." in out
        assert "nothing was sent anywhere" not in out
        assert not (tmp_path / "results").exists()
        for content in ("Synthetic subject alpha", "Synthetic body alpha", "pat@example.test"):
            assert content not in out

    async def test_dry_run_flags_an_api_stand_in_when_mlx_api_key_is_set(self, golden_file, tmp_path, capsys):
        local = _FakeLocal()
        local.api_key = "local-synthetic"  # build_local_llm puts MLX_API_KEY here

        await compare.run_comparison(
            _args(golden_file, tmp_path, dry_run=True), local_llm=local,
            client_factory=_never, getpass_fn=_never, input_fn=_never,
        )

        assert "API stand-in (decision D4)" in capsys.readouterr().out

    async def test_unreachable_local_model_refuses_before_anything_is_sent(
        self, golden_file, tmp_path, capsys,
    ):
        code = await compare.run_comparison(
            _args(golden_file, tmp_path), local_llm=_FakeLocal(reachable=False),
            client_factory=_never, getpass_fn=_never, input_fn=_never,
        )

        assert code == 1
        assert "NOT reachable" in capsys.readouterr().out

    async def test_anything_but_yes_stops_before_the_key_is_asked(self, golden_file, tmp_path, capsys):
        local = _FakeLocal()

        code = await compare.run_comparison(
            _args(golden_file, tmp_path), local_llm=local,
            client_factory=_never, getpass_fn=_never, input_fn=lambda prompt: "no",
        )

        assert code == 1
        assert local.calls == 0
        assert "Nothing was sent to Anthropic" in capsys.readouterr().out

    async def test_rejected_key_or_model_stops_before_arm_a(self, golden_file, tmp_path, capsys):
        router, keys, local = _Router(model_status=401), [], _FakeLocal()

        code = await compare.run_comparison(
            _args(golden_file, tmp_path), local_llm=local, client_factory=_factory(router, keys),
            getpass_fn=lambda prompt: SENTINEL_KEY, input_fn=lambda prompt: "yes",
        )

        assert code == 1
        assert local.calls == 0
        assert router.message_requests() == []
        assert "AuthenticationError" in capsys.readouterr().out

    async def test_arm_b_is_not_run_when_arm_a_produced_nothing(self, golden_file, tmp_path, capsys):
        router, keys = _Router(), []

        code = await compare.run_comparison(
            _args(golden_file, tmp_path), local_llm=_FakeLocal(fail=True),
            client_factory=_factory(router, keys),
            getpass_fn=lambda prompt: SENTINEL_KEY, input_fn=lambda prompt: "yes",
        )

        assert code == 1
        assert router.message_requests() == []
        out = capsys.readouterr().out
        # The model lookup did reach Anthropic; what was not sent is mail.
        assert "arm B was not run and no mail was sent to Anthropic" in out

    async def test_a_thread_arm_a_lost_to_an_outage_is_not_sent_but_a_content_failure_is(
        self, golden_file, tmp_path,
    ):
        # The MacBook sleeps partway through arm A: the thread it lost to the
        # outage buys no comparison, so its mail must not go to Anthropic. A
        # thread arm A answered unusably is a measured outcome and is sent.
        router = _Router()
        local = _FakeLocal(outage_on="Synthetic subject alpha", content_failure_on="Synthetic subject beta")

        code = await compare.run_comparison(
            _args(golden_file, tmp_path), local_llm=local, client_factory=_factory(router, []),
            getpass_fn=lambda prompt: SENTINEL_KEY, input_fn=lambda prompt: "yes",
        )

        sent = [json.loads(r.content)["messages"][0]["content"] for r in router.message_requests()]
        assert len(sent) == 2
        assert not any("Synthetic subject alpha" in content for content in sent)
        assert any("Synthetic subject beta" in content for content in sent)
        assert code == 0
        [path_b] = (tmp_path / "results").glob("*compare-B-*.jsonl")
        rows_b = {r.thread_id: r for r in report.load_results(path_b)[1]}
        assert rows_b["900101"].error == compare.NOT_SENT_LOCAL_UNAVAILABLE
        [report_file] = (tmp_path / "results").glob("*.md")
        text = report_file.read_text()
        rows, _ = _rows(text)
        assert rows["Threads not sent"] == ["0", "1 (local model unavailable 1)"]
        assert "Scored on the 2 threads both arms were sent" in text

    async def test_a_run_whose_replies_are_all_refusals_is_marked_not_meaningful(self, golden_file, tmp_path):
        router = _Router(reply=_refusal("reasoning_extraction"))

        await compare.run_comparison(
            _args(golden_file, tmp_path), local_llm=_FakeLocal(), client_factory=_factory(router, []),
            getpass_fn=lambda prompt: SENTINEL_KEY, input_fn=lambda prompt: "yes",
        )

        # Three replies are below the early stop's minimum, so all were sent.
        assert len(router.message_requests()) == 3
        [report_file] = (tmp_path / "results").glob("*.md")
        text = report_file.read_text()
        assert text.index("Not a meaningful comparison") < text.index("| Metric")
        rows, _ = _rows(text)
        assert rows["Errors by kind"][1] == "refusal: reasoning_extraction 3"

    async def test_arm_b_sends_nothing_more_once_refusals_dominate(self, tmp_path):
        golden = tmp_path / "golden.jsonl"
        golden.write_text("".join(json.dumps(_thread(f"9008{i:02d}").to_dict()) + "\n" for i in range(12)))
        router = _Router(reply=_refusal())
        asked = []

        await compare.run_comparison(
            _args(golden, tmp_path), local_llm=_FakeLocal(), client_factory=_factory(router, []),
            getpass_fn=lambda prompt: SENTINEL_KEY, input_fn=lambda prompt: asked.append(prompt) or "yes",
        )

        assert len(asked) == 1  # the one typed yes; the stop asks nothing
        sent = len(router.message_requests())
        # Up to four requests are in flight when the fifth refusal lands, and
        # they still land; nothing is sent after that.
        assert 5 <= sent <= 8
        [path_b] = (tmp_path / "results").glob("*compare-B-*.jsonl")
        rows_b = report.load_results(path_b)[1]
        assert sum(1 for r in rows_b if compare.error_kind(r.error) == "not sent") == 12 - sent
        [report_file] = (tmp_path / "results").glob("*.md")
        text = report_file.read_text()
        rows, _ = _rows(text)
        assert rows["Threads not sent"][1] == f"{12 - sent} (stopped after refusals {12 - sent})"
        assert text.index("Arm B stopped sending") < text.index("| Metric")

    @pytest.mark.parametrize("failure", [529, httpx2.ConnectError], ids=["overloaded", "connection"])
    async def test_a_thread_arm_b_lost_to_an_outage_is_scored_for_neither_arm(
        self, golden_file, tmp_path, failure,
    ):
        # Alpha is the golden file's one needs_response thread. Anthropic is
        # overloaded or unreachable for it: an outage, as a local one is for arm A.
        router = _Router(fail_on="Synthetic subject alpha", failure=failure)

        code = await compare.run_comparison(
            _args(golden_file, tmp_path), local_llm=_FakeLocal(),
            client_factory=lambda key: _sdk_client(router).with_options(max_retries=0),
            getpass_fn=lambda prompt: SENTINEL_KEY, input_fn=lambda prompt: "yes",
        )

        assert code == 0
        [path_b] = (tmp_path / "results").glob("*compare-B-*.jsonl")
        rows_b = {r.thread_id: r for r in report.load_results(path_b)[1]}
        assert compare.error_kind(rows_b["900101"].error) == "unavailable"
        [report_file] = (tmp_path / "results").glob("*.md")
        rows, _ = _rows(report_file.read_text())
        assert rows["needs_response recall"] == ["N/A (0/0)", "N/A (0/0)"]
        assert rows["Errors (no usable answer)"] == ["0/2", "0/2"]
        assert rows["Lost to an outage"] == ["0", "1"]

    async def test_full_run_compares_both_arms_on_person_threads(self, golden_file, tmp_path):
        router, keys, local = _Router(), [], _FakeLocal()

        code = await compare.run_comparison(
            _args(golden_file, tmp_path), local_llm=local, client_factory=_factory(router, keys),
            getpass_fn=lambda prompt: SENTINEL_KEY, input_fn=lambda prompt: "yes",
        )

        assert code == 0
        assert keys == [SENTINEL_KEY]
        assert local.calls == 3
        assert len(router.message_requests()) == 3
        results_dir = tmp_path / "results"
        [report_file] = list(results_dir.glob("*.md"))
        text = report_file.read_text()
        rows, _ = _rows(text)
        assert rows["Errors (no usable answer)"] == ["0/3", "0/3"]  # the cloud slot was never called
        assert rows["needs_response recall"] == ["0.0% (0/1)", "100.0% (1/1)"]
        runs = sorted(results_dir.glob("*.jsonl"))
        assert len(runs) == 2
        metas = [report.load_results(path)[0] for path in runs]
        assert {m.local_model for m in metas} == {"qwen-synthetic:8b", "claude-opus-5-5"}
        assert {m.stages for m in metas} == {"stage2_only"}

    async def test_report_holds_no_message_content(self, golden_file, tmp_path):
        router = _Router()

        await compare.run_comparison(
            _args(golden_file, tmp_path), local_llm=_FakeLocal(), client_factory=_factory(router, []),
            getpass_fn=lambda prompt: SENTINEL_KEY, input_fn=lambda prompt: "yes",
        )

        [report_file] = list((tmp_path / "results").glob("*.md"))
        text = report_file.read_text()
        for content in ("Synthetic subject", "Synthetic body", "pat@example.test", "Pat Example",
                        "900101", "900102", "900103"):
            assert content not in text

    async def test_key_is_never_printed_or_written(self, golden_file, tmp_path, capsys):
        router = _Router()

        await compare.run_comparison(
            _args(golden_file, tmp_path), local_llm=_FakeLocal(), client_factory=_factory(router, []),
            getpass_fn=lambda prompt: SENTINEL_KEY, input_fn=lambda prompt: "yes",
        )

        captured = capsys.readouterr()
        assert SENTINEL_KEY not in captured.out + captured.err
        written = list((tmp_path / "results").iterdir())
        assert written
        for path in written:
            assert SENTINEL_KEY not in path.read_text()

    async def test_arm_a_runs_at_the_daemons_local_concurrency_env_override_included(
        self, golden_file, tmp_path, capsys, monkeypatch,
    ):
        # The daemon reads LOCAL_PARALLEL over [daemon] local_parallel
        # (README-technical documents it), so arm A must too.
        monkeypatch.setenv("LOCAL_PARALLEL", "3")

        await compare.run_comparison(
            _args(golden_file, tmp_path), local_llm=_FakeLocal(), client_factory=_factory(_Router(), []),
            getpass_fn=lambda prompt: SENTINEL_KEY, input_fn=lambda prompt: "yes",
        )

        [path_a] = (tmp_path / "results").glob("*compare-A-*.jsonl")
        assert report.load_results(path_a)[0].parallelism == 3
        assert "3 at a time" in capsys.readouterr().out

    async def test_max_threads_caps_what_is_sent(self, golden_file, tmp_path):
        router, local = _Router(), _FakeLocal()

        await compare.run_comparison(
            _args(golden_file, tmp_path, max_threads=2), local_llm=local,
            client_factory=_factory(router, []),
            getpass_fn=lambda prompt: SENTINEL_KEY, input_fn=lambda prompt: "yes",
        )

        assert local.calls == 2
        assert len(router.message_requests()) == 2


class TestParser:
    def _options(self):
        return {opt for action in compare.build_parser()._actions for opt in action.option_strings}

    def test_flags(self):
        assert self._options() == {
            "-h", "--help", "--golden-set", "--config", "--output-dir", "--model",
            "--anthropic-key-file", "--dry-run", "--max-threads",
        }

    def test_no_flag_takes_the_key_itself(self):
        # The key comes from a file or a hidden prompt, never argv: the only
        # key-related flag names a file.
        assert [opt for opt in self._options() if "key" in opt] == ["--anthropic-key-file"]

    def test_model_defaults_to_opus_and_accepts_only_priced_models(self):
        parser = compare.build_parser()
        assert parser.parse_args([]).model == "claude-opus-5-5"
        assert parser.parse_args(["--model", "claude-haiku-5-5"]).model == "claude-haiku-5-5"
        with pytest.raises(SystemExit):
            parser.parse_args(["--model", "claude-unpriced-9"])

    def test_defaults_match_run_eval_paths(self):
        args = compare.build_parser().parse_args([])
        assert args.golden_set == "evals/golden_set.jsonl"
        assert args.output_dir == "evals/results/"
        assert args.anthropic_key_file is None
        assert args.dry_run is False


class TestEntryPoint:
    def test_sdk_debug_logging_does_not_write_the_mail(self, monkeypatch, caplog):
        # ANTHROPIC_LOG=debug sets the anthropic logger to DEBUG when the SDK is
        # imported, and the SDK then logs each request's options, body
        # included. The entry point must pin that logger before anything is sent.
        caplog.set_level(logging.DEBUG, logger="anthropic")

        async def one_request(args):
            transport = _Transport((200, _message("FYI"), None))
            llm = compare.AnthropicLLMClient(_sdk_client(transport), model="claude-opus-5-5")
            await llm.complete("SYSTEM-MARKER-900301", "BODY-MARKER-900301")
            return 0

        monkeypatch.setattr(compare, "run_comparison", one_request)
        monkeypatch.setattr(sys, "argv", ["evals.compare"])
        with pytest.raises(SystemExit):
            compare.cli()

        assert "BODY-MARKER-900301" not in caplog.text
        assert "SYSTEM-MARKER-900301" not in caplog.text
        assert logging.getLogger("anthropic").level == logging.INFO


class TestBuildLocalLlm:
    def test_built_from_mlx_env_and_the_local_config_section(self, monkeypatch):
        # Arm A must be the local tier as run_eval builds it: a dropped
        # extra_body would turn the local model's thinking back on and skew
        # the comparison without any error.
        monkeypatch.setenv("MLX_URL", "http://ollama.example.test:11434/v1/chat/completions")
        monkeypatch.setenv("MLX_API_KEY", "local-synthetic")
        extra = {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
        config = {"llm": {"local": {
            "model": "qwen-synthetic:8b", "max_tokens": 4096, "temperature": 0, "timeout": 180,
            "extra_body": extra,
        }}}

        llm = compare.build_local_llm(config)

        assert llm.base_url == "http://ollama.example.test:11434/v1/chat/completions"
        assert llm.api_key == "local-synthetic"
        assert (llm.model, llm.max_tokens, llm.temperature, llm.timeout) == (
            "qwen-synthetic:8b", 4096, 0, 180)
        assert llm.extra_body == extra


class TestArmSettingsDisclosed:
    """The arms differ in more than the model, and the settings are disclosed
    rather than aligned, so the plan and the report state what each arm was
    sent."""

    @pytest.fixture
    def unusual_local_config(self, monkeypatch):
        # Values unlike the shipped config's, so only text built from the
        # config that ran can pass.
        config, path = compare.load_config(None)
        config["llm"]["local"].update(
            temperature=0.35, max_tokens=2345, timeout=123,
            extra_body={"reasoning_effort": "synthetic-effort"})
        monkeypatch.setattr(compare, "load_config", lambda _path: (config, path))

    @staticmethod
    def _assert_settings(text):
        for local in ("temperature 0.35", "max_tokens 2345", "timeout 123 s",
                      '{"reasoning_effort": "synthetic-effort"}'):
            assert local in text
        for claude in (f"effort {compare.ANTHROPIC_EFFORT}", f"max_tokens {compare.ANTHROPIC_MAX_TOKENS}",
                       "adaptive thinking", "default sampling", "no temperature",
                       f"timeout {compare.ANTHROPIC_TIMEOUT_SECONDS:.0f} s"):
            assert claude in text
        assert "one run" in text.lower() and "a rerun can differ" in text
        assert "deterministic" not in text.lower()

    async def test_the_plan_states_each_arms_settings_and_one_run(
        self, golden_file, tmp_path, capsys, unusual_local_config,
    ):
        await compare.run_comparison(
            _args(golden_file, tmp_path, dry_run=True), local_llm=_FakeLocal(),
            client_factory=_never, getpass_fn=_never, input_fn=_never,
        )

        self._assert_settings(capsys.readouterr().out)

    async def test_the_report_states_each_arms_settings_above_the_table(
        self, golden_file, tmp_path, unusual_local_config,
    ):
        await compare.run_comparison(
            _args(golden_file, tmp_path), local_llm=_FakeLocal(), client_factory=_factory(_Router(), []),
            getpass_fn=lambda prompt: SENTINEL_KEY, input_fn=lambda prompt: "yes",
        )

        [report_file] = (tmp_path / "results").glob("*.md")
        text = report_file.read_text()
        self._assert_settings(text)
        assert text.index("max_tokens 2345") < text.index("| Metric")
        assert "arm A ran at temperature 0.35" in text


class TestRunComparisonExits:
    async def test_a_refused_key_file_stops_before_any_call(self, golden_file, tmp_path, capsys):
        key_file = tmp_path / "key"
        key_file.write_text(SENTINEL_KEY)
        key_file.chmod(0o644)
        local = _FakeLocal()

        code = await compare.run_comparison(
            _args(golden_file, tmp_path, anthropic_key_file=key_file), local_llm=local,
            client_factory=_never, getpass_fn=_never, input_fn=lambda prompt: "yes",
        )

        assert code == 1
        assert local.calls == 0
        out = capsys.readouterr().out
        assert "chmod 600" in out
        assert SENTINEL_KEY not in out

    async def test_no_reviewed_person_threads_stops_at_once(self, tmp_path, capsys):
        golden = tmp_path / "golden.jsonl"
        golden.write_text(json.dumps(_thread("900201", sender_type="service").to_dict()) + "\n")

        code = await compare.run_comparison(
            _args(golden, tmp_path), local_llm=_FakeLocal(reachable=False),
            client_factory=_never, getpass_fn=_never, input_fn=_never,
        )

        assert code == 1
        assert "No reviewed person threads" in capsys.readouterr().out
