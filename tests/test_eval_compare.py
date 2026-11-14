"""Tests for the Claude comparison eval (evals/compare.py, issue #92).

No test reaches a real service. The Anthropic arm runs through the real SDK
over an ``httpx2.MockTransport``, so request shapes and response parsing are the
SDK's own; the local arm and the cloud tier are plain fake classes. Every
fixture is synthetic mail (``example.test`` addresses, invented names).
"""

import argparse
import base64
import json

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
             input_tokens=120, output_tokens=30, thinking=None):
    content = []
    if thinking is not None:
        content.append({"type": "thinking", "thinking": thinking, "signature": "sig"})
    if text is not None:
        content.append({"type": "text", "text": text})
    return {
        "id": "msg_900001", "type": "message", "role": "assistant",
        "model": "claude-opus-5-5", "content": content,
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


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

    def test_retry_budget_and_per_request_timeout_are_set(self):
        client = compare.build_anthropic_client(SENTINEL_KEY)

        assert client.max_retries == compare.ANTHROPIC_MAX_RETRIES
        assert client.timeout == compare.ANTHROPIC_TIMEOUT_SECONDS


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
        assert est.ceiling == pytest.approx(
            compare.cost_usd("claude-opus-5-5", est.input_tokens, 2 * compare.ANTHROPIC_MAX_TOKENS))


def _plan(**kw):
    prompts = [("s" * 2000, "u" * 1500)] * 5
    base = dict(
        golden_set_path="evals/golden_set.jsonl", reviewed_threads=12, person_threads=5,
        max_thread_chars=16000, local_model="qwen-synthetic:8b",
        local_url="http://ollama.example.test:11434/v1/chat/completions", local_error=None,
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

    def test_says_plainly_that_real_mail_goes_to_anthropic(self):
        text = compare.format_plan(_plan())

        assert "The golden sets are real mail, and arm B sends them to Anthropic" in text
        assert ("inputs are deleted within 30 days, and zero retention is available "
                "under an agreement") in text

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
    def test_scores_both_arms_on_the_threads_both_answered(self):
        rows, _ = _rows(_report())

        # 900007 failed in arm A and 900008 in arm B, so six threads are scored.
        assert rows["needs_response recall"] == ["33.3% (1/3)", "100.0% (3/3)"]
        assert rows["Errors (no usable answer)"] == ["1/8", "1/8"]
        assert "Scored on the 6 threads both arms answered" in _report()

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

        assert rows["needs_response precision"] == ["50.0% (1/2)", "100.0% (3/3)"]
        assert rows["fyi recall"] == ["0.0% (0/2)", "100.0% (2/2)"]
        assert rows["fyi precision"] == ["0.0% (0/2)", "100.0% (2/2)"]
        assert rows["low_priority recall"] == ["0.0% (0/1)", "100.0% (1/1)"]
        assert rows["low_priority precision"] == ["0.0% (0/2)", "100.0% (1/1)"]

    def test_latency_median_and_p90_per_message(self):
        rows, _ = _rows(_report())

        assert rows["Latency per message, median"] == ["3.5 s", "0.5 s"]
        assert rows["Latency per message, p90"] == ["6.0 s", "0.5 s"]

    def test_cost_per_message_and_total(self):
        rows, _ = _rows(_report())

        # 10,000 x $4 + 2,000 x $20 per million = $0.08 over 8 threads sent.
        assert rows["Cost per message"] == ["$0 (local)", "$0.0100"]
        assert rows["Cost, total"][0] == "$0 (local)"
        assert rows["Cost, total"][1].startswith("$0.08")
        assert "2026-11-03" in rows["Cost, total"][1]

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

    def __init__(self, *, reachable=True, fail=False):
        self.reachable = reachable
        self.fail = fail
        self.calls = 0

    async def probe(self, timeout=None):
        if self.reachable:
            return AvailabilityResult(ok=True, status_code=200)
        return AvailabilityResult(ok=False, error="ConnectError: connection refused")

    async def complete(self, system_prompt, user_content, include_thinking=False):
        self.calls += 1
        if self.fail:
            raise LLMUnavailableError("synthetic outage", tier="local")
        return ("FYI", "") if include_thinking else "FYI"


class _Router:
    """Answers the SDK by path: the model lookup, then message requests."""

    def __init__(self, model_status=200):
        self.model_status = model_status
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
        return httpx2.Response(200, json=_message("<think>reasoning</think>\nNEEDS_RESPONSE"))

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
        assert not (tmp_path / "results").exists()
        for content in ("Synthetic subject alpha", "Synthetic body alpha", "pat@example.test"):
            assert content not in out

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
        assert "arm B was not run" in capsys.readouterr().out

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
