"""Tests for the Claude comparison eval (evals/compare.py, issue #92).

No test reaches a real service. The Anthropic arm runs through the real SDK
over an ``httpx2.MockTransport``, so request shapes and response parsing are the
SDK's own; the local arm and the cloud tier are plain fake classes. Every
fixture is synthetic mail (``example.test`` addresses, invented names).
"""

import json

import anthropic
import httpx2
import pytest

from evals import compare
from llm_client import LLMContentError

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
        # temperature is rejected by current models; thinking cannot be disabled
        # on them and is left to its adaptive default; fallbacks would change
        # which model answered, which an eval must not do.
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
        transport = _Transport(
            (429, _error("rate_limit_error"), {"retry-after-ms": "1"}),
            (529, _error("overloaded_error"), {"retry-after-ms": "1"}),
            (200, _message("FYI"), None),
        )
        llm = compare.AnthropicLLMClient(_sdk_client(transport), model="claude-opus-5-5")

        assert await llm.complete("S", "U") == "FYI"
        assert len(transport.requests) == 3

    def test_retry_budget_and_per_request_timeout_are_set(self):
        client = compare.build_anthropic_client(SENTINEL_KEY)

        assert client.max_retries == compare.ANTHROPIC_MAX_RETRIES
        assert client.timeout == compare.ANTHROPIC_TIMEOUT_SECONDS
