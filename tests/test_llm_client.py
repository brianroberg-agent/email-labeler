"""Tests for LLM client."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from llm_client import (
    DEFAULT_AVAILABILITY_TIMEOUT,
    AvailabilityResult,
    LLMBalanceError,
    LLMClient,
    LLMContentError,
    LLMUnavailableError,
)


@pytest.fixture
def cloud_client():
    return LLMClient(
        base_url="https://api.cloud.example.com/v1/chat/completions",
        api_key="sk-test-key",
        model="test-cloud-model",
        max_tokens=8096,
        temperature=0.2,
        timeout=60,
    )


@pytest.fixture
def local_client():
    return LLMClient(
        base_url="http://localhost:8080/v1/chat/completions",
        api_key="",
        model="mlx-community/Qwen3-14B-4bit",
        max_tokens=8096,
        temperature=0.2,
        timeout=180,
    )


def _mock_response(status_code=200, json_data=None):
    """Create a mock httpx Response."""
    response = httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("POST", "https://example.com"),
    )
    return response


class TestComplete:
    async def test_sends_correct_request_format(self, cloud_client):
        """Verify request body matches OpenAI chat completion format."""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"content": "PERSON"}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await cloud_client.complete("You are a classifier.", "Classify this email")

            mock_client.post.assert_called_once()
            call_kwargs = mock_client.post.call_args
            assert call_kwargs.args[0] == "https://api.cloud.example.com/v1/chat/completions"

            body = call_kwargs.kwargs["json"]
            assert body["model"] == "test-cloud-model"
            assert body["max_tokens"] == 8096
            assert body["temperature"] == 0.2
            assert len(body["messages"]) == 2
            assert body["messages"][0] == {"role": "system", "content": "You are a classifier."}
            assert body["messages"][1] == {"role": "user", "content": "Classify this email"}

    async def test_sends_auth_header_when_api_key_set(self, cloud_client):
        """Verify Authorization header is sent when api_key is provided."""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"content": "SERVICE"}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await cloud_client.complete("sys", "user")

            call_kwargs = mock_client.post.call_args
            headers = call_kwargs.kwargs["headers"]
            assert headers["Authorization"] == "Bearer sk-test-key"

    async def test_no_auth_header_when_no_api_key(self, local_client):
        """Verify no Authorization header when api_key is empty."""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"content": "NEEDS_RESPONSE"}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await local_client.complete("sys", "user")

            call_kwargs = mock_client.post.call_args
            headers = call_kwargs.kwargs["headers"]
            assert "Authorization" not in headers

    async def test_strips_think_tags(self, cloud_client):
        """Verify <think>...</think> tags are stripped from response."""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"content":
                "<think>The sender is a company sending a shipping notification.</think>\nSERVICE"}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await cloud_client.complete("sys", "user")
            assert result == "SERVICE"

    async def test_strips_double_bracket_think_tags(self, cloud_client):
        """Verify <<think>>...</<think>> tags (DeepSeek variant) are stripped."""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"content":
                "The sender is a company\n\n<<think>>\nreasoning here\n</<think>>\n\nSERVICE"}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await cloud_client.complete("sys", "user")
            assert "SERVICE" in result
            assert "<<think>>" not in result
            assert "reasoning here" not in result

    async def test_extracts_double_bracket_thinking(self, cloud_client):
        """Verify <<think>> content is captured in include_thinking mode."""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"content":
                "preamble\n<<think>>\ndeep reasoning\n</<think>>\nSERVICE"}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            response, thinking = await cloud_client.complete("sys", "user", include_thinking=True)
            assert response == "preamble\n\nSERVICE"
            assert "deep reasoning" in thinking

    async def test_strips_multiple_think_tags(self, cloud_client):
        """Verify multiple think blocks are stripped."""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"content":
                "<think>first thought</think>\n<think>second thought</think>\nPERSON"}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await cloud_client.complete("sys", "user")
            assert result == "PERSON"

    async def test_returns_raw_content_without_think_tags(self, cloud_client):
        """Content without think tags is returned as-is (stripped)."""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"content": "  PERSON  "}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await cloud_client.complete("sys", "user")
            assert result == "PERSON"

    async def test_raises_on_http_error(self, cloud_client):
        """A 5xx raises LLMUnavailableError — provider-shaped, never a strike.

        Reworked for decision D5 (Wave 2 T8): this test used to pin 500 →
        RuntimeError, the give-up-eligible path; a 5xx means the provider can't
        serve anyone and is deferred instead. The RuntimeError pin for
        request-specific non-200s lives on the 400 in
        test_http_error_message_identifies_provider below."""
        mock_response = _mock_response(status_code=500, json_data={"error": "Internal Server Error"})

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(LLMUnavailableError, match="status 500"):
                await cloud_client.complete("sys", "user")

    async def test_http_error_message_identifies_provider(self, cloud_client):
        """Any non-200 error message names the model and endpoint (3 clients exist)."""
        with pytest.raises(RuntimeError) as exc_info:
            await _post_canned(cloud_client, _mock_response(400, {"error": "bad request"}))

        msg = str(exc_info.value)
        assert "test-cloud-model" in msg
        assert "api.cloud.example.com" in msg

    async def test_connect_error_surfaces_as_unavailable(self, cloud_client):
        """A refused connection (server down) surfaces as LLMUnavailableError (transient)."""
        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(LLMUnavailableError):
                await cloud_client.complete("sys", "user")

    async def test_connect_timeout_surfaces_as_unavailable_not_timeout(self, cloud_client):
        """A connect timeout is endpoint unavailability, NOT a request-specific timeout.

        Regression (review finding #1): httpx.ConnectTimeout is a subclass of
        TimeoutException, so it used to be mapped to TimeoutError and counted toward
        give-up. It must surface as LLMUnavailableError (transient) instead.
        """
        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectTimeout("connect timed out")
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(LLMUnavailableError):
                await cloud_client.complete("sys", "user")

    async def test_pool_timeout_surfaces_as_unavailable(self, cloud_client):
        """A pool timeout (connection pool exhausted) is transient unavailability."""
        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.PoolTimeout("pool exhausted")
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(LLMUnavailableError):
                await cloud_client.complete("sys", "user")

    async def test_read_timeout_stays_request_specific_timeout_error(self, cloud_client):
        """A read timeout (slow prefill on large input) stays TimeoutError — request-specific,
        so the daemon may give up on one oversized thread rather than retry forever."""
        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ReadTimeout("timed out")
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(TimeoutError, match="timed out after 60s"):
                await cloud_client.complete("sys", "user")

    async def test_surfaces_read_error_as_informative_unavailable_error(self, cloud_client):
        """A dropped/reset connection (httpx.ReadError) surfaces as a non-empty LLMUnavailableError.

        Regression: a bare reset previously propagated as an exception whose str() was
        empty, so callers recorded it as an empty error / no result instead of a failure.
        It is treated as transient unavailability (server crashed/restarted/flapped).
        """
        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ReadError("")  # empty msg, like a bare reset
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(LLMUnavailableError) as exc_info:
                await cloud_client.complete("sys", "user")
            msg = str(exc_info.value)
            assert msg  # non-empty even though the underlying error message was empty
            assert "ReadError" in msg
            assert "test-cloud-model" in msg

    async def test_surfaces_remote_protocol_error_as_unavailable(self, cloud_client):
        """A server disconnect (httpx.RemoteProtocolError) surfaces as LLMUnavailableError."""
        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.RemoteProtocolError("Server disconnected")
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(LLMUnavailableError, match="RemoteProtocolError"):
                await cloud_client.complete("sys", "user")

    async def test_uses_configured_timeout(self, cloud_client):
        """Verify timeout is passed to httpx client."""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"content": "SERVICE"}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await cloud_client.complete("sys", "user")

            mock_client_cls.assert_called_once_with(timeout=60)


class TestTier:
    """LLMUnavailableError carries the raising client's tier (issue #24).

    Cloud and local clients raise the same exception type; the tier attribute is
    what lets the daemon log a routine local outage (laptop offline for hours)
    at a lower level than a surprising cloud outage.
    """

    def _down_client(self, exc, **client_kwargs):
        client = LLMClient(
            base_url="http://localhost:8080/v1/chat/completions",
            api_key="", model="test-model", timeout=60, **client_kwargs,
        )
        patcher = patch("llm_client.httpx.AsyncClient")
        mock_client_cls = patcher.start()
        mock_client = AsyncMock()
        mock_client.post.side_effect = exc
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        return client, patcher

    async def test_connect_error_carries_client_tier(self):
        client, patcher = self._down_client(httpx.ConnectError("refused"), tier="local")
        try:
            with pytest.raises(LLMUnavailableError) as exc_info:
                await client.complete("sys", "user")
        finally:
            patcher.stop()
        assert exc_info.value.tier == "local"

    async def test_mid_request_drop_carries_client_tier(self):
        client, patcher = self._down_client(httpx.ReadError("reset"), tier="local")
        try:
            with pytest.raises(LLMUnavailableError) as exc_info:
                await client.complete("sys", "user")
        finally:
            patcher.stop()
        assert exc_info.value.tier == "local"

    async def test_tier_defaults_to_none(self, cloud_client):
        """Clients that don't declare a tier (evals, older call sites) stay tier-less."""
        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("refused")
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(LLMUnavailableError) as exc_info:
                await cloud_client.complete("sys", "user")
        assert exc_info.value.tier is None


async def _post_canned(client, response):
    """Drive complete() against a canned httpx response."""
    with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = response
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        return await client.complete("sys", "user")


# Verbatim shape of the Novita out-of-funds response observed in production.
NOVITA_BALANCE_BODY = {
    "code": 403,
    "reason": "NOT_ENOUGH_BALANCE",
    "message": "not enough balance",
    "metadata": {},
}


class TestBalanceError:
    """Out-of-funds responses raise LLMBalanceError (account-wide: halts the
    function that provider serves — decision D5's scope rule, D19).

    Detection must be conservative enough that an ordinary 403 (bad key, forbidden
    route) stays a bare RuntimeError — only a payment-required status or a body
    carrying a known balance/quota signature may trip a restart-only halt.
    """

    async def test_403_with_balance_body_raises_balance_error(self, cloud_client):
        """The exact Novita NOT_ENOUGH_BALANCE response surfaces as LLMBalanceError."""
        with pytest.raises(LLMBalanceError):
            await _post_canned(cloud_client, _mock_response(403, NOVITA_BALANCE_BODY))

    async def test_402_raises_balance_error_regardless_of_body(self, cloud_client):
        """402 Payment Required is a balance error whatever the body says."""
        with pytest.raises(LLMBalanceError):
            await _post_canned(cloud_client, _mock_response(402, {"error": "payment required"}))

    async def test_plain_403_stays_bare_runtime_error(self, cloud_client):
        """A 403 without a balance signature (bad key etc.) must NOT halt anything."""
        with pytest.raises(RuntimeError) as exc_info:
            await _post_canned(cloud_client, _mock_response(403, {"error": "forbidden"}))
        assert not isinstance(exc_info.value, LLMBalanceError)

    @pytest.mark.parametrize(
        ("status", "message"),
        [
            (403, "Insufficient_Quota: request rejected"),
            (400, "Your credit balance is too low to access the Anthropic API."),
            (400, "NOT_ENOUGH_BALANCE"),
        ],
    )
    async def test_balance_signatures_matched_case_insensitively(
        self, cloud_client, status, message
    ):
        """Known provider phrasings (OpenAI-style, Anthropic) count on 400/403 too."""
        body = {"error": {"message": message, "code": "billing"}}
        with pytest.raises(LLMBalanceError):
            await _post_canned(cloud_client, _mock_response(status, body))

    async def test_429_quota_phrasing_never_halts(self, cloud_client):
        """A 429 must NEVER halt, even with quota phrasing (decision D19):
        Gemini-style per-minute rate limits use the same wording as hard quota
        exhaustion, and wrongly converting a transient rate limit into a
        restart-only halt is worse than retrying it as unavailability. Reworked
        for D5 (Wave 2 T8): an exhausted 429 is provider-shaped now, so it
        raises LLMUnavailableError instead of the old RuntimeError strike path.

        This body carries no balance signature, so it pins only the
        wording-collision rationale; the parametrized test below is the one that
        actually holds 429 out of the balance statuses."""
        body = {"error": {"message": "You exceeded your current quota, please check your plan."}}
        with patch("retry.asyncio.sleep", new=AsyncMock()):  # skip real retry backoff
            with pytest.raises(LLMUnavailableError) as exc_info:
                await _post_canned(cloud_client, _mock_response(429, body))
        assert not isinstance(exc_info.value, LLMBalanceError)

    @pytest.mark.parametrize(
        "message",
        [
            "Insufficient_Quota: request rejected",
            "Your credit balance is too low to access the Anthropic API.",
            "NOT_ENOUGH_BALANCE",
        ],
    )
    async def test_429_carrying_a_balance_signature_still_never_halts(
        self, cloud_client, message
    ):
        """The load-bearing D19 guard: the SAME phrasings that trip
        LLMBalanceError on 400/403 must not trip it on 429. The balance branch
        is a two-condition guard (status in the balance set AND the body matches
        the signature), so a 429 body with no signature — the test above —
        cannot tell whether 429 is in that set. These bodies satisfy the
        signature half, leaving the status exclusion as the only thing between a
        429 and a restart-only function halt.

        The live case: OpenAI returns 429 with `insufficient_quota` for an
        exhausted account, wording a per-minute throttle shares."""
        body = {"error": {"message": message, "code": "billing"}}
        with patch("retry.asyncio.sleep", new=AsyncMock()):  # skip real retry backoff
            with pytest.raises(LLMUnavailableError) as exc_info:
                await _post_canned(cloud_client, _mock_response(429, body))
        assert not isinstance(exc_info.value, LLMBalanceError)

    async def test_echoed_email_balance_phrase_does_not_trip(self, cloud_client):
        """An error body that quotes the email being classified (e.g. a bank
        'insufficient balance' alert echoed by a moderation/validation error)
        must not read as the PROVIDER being out of funds."""
        body = {
            "error": "invalid request",
            "input": "ALERT: your checking account has an insufficient balance for autopay.",
        }
        with pytest.raises(RuntimeError) as exc_info:
            await _post_canned(cloud_client, _mock_response(403, body))
        assert not isinstance(exc_info.value, LLMBalanceError)

    async def test_message_identifies_provider(self):
        """The error message names tier, model, and endpoint so logs show WHO is broke."""
        client = LLMClient(
            base_url="https://api.cloud.example.com/v1/chat/completions",
            api_key="sk-test-key",
            model="test-cloud-model",
            tier="cloud",
        )
        with pytest.raises(LLMBalanceError) as exc_info:
            await _post_canned(client, _mock_response(403, NOVITA_BALANCE_BODY))
        msg = str(exc_info.value)
        assert "test-cloud-model" in msg
        assert "api.cloud.example.com" in msg
        assert "cloud" in msg


class TestProviderShapedStatuses:
    """Exhausted 429s and 5xx responses are provider-shaped (decision D5, Wave 2
    T8): the provider can't serve anyone right now, so they surface as
    LLMUnavailableError — deferred and retried next cycle, never a strike
    candidate. Request-specific non-200s (400/401/404/422…) stay RuntimeError."""

    async def test_exhausted_429_raises_unavailable_not_runtime(self, cloud_client):
        """A 429 that survives retry.py's budget is a sustained throttle — the
        endpoint can't serve anyone, which is never one thread's blame (D5).
        It must surface as LLMUnavailableError carrying the tier (so the
        daemon's quiet-local handling still applies), not the RuntimeError
        strike-candidate path."""
        client = LLMClient(
            base_url="https://api.cloud.example.com/v1/chat/completions",
            api_key="sk-test-key",
            model="test-cloud-model",
            tier="cloud",
        )
        with patch("retry.asyncio.sleep", new=AsyncMock()):  # skip real retry backoff
            with pytest.raises(LLMUnavailableError) as exc_info:
                await _post_canned(client, _mock_response(429, {"error": "rate limited"}))
        assert exc_info.value.tier == "cloud"


class TestContentlessResponse:
    """A response message lacking usable content must raise a clear, non-KeyError error.

    A reasoning model (or GLM) that exhausts max_tokens mid-<think> returns a message
    with reasoning/reasoning_content but no `content` — previously a raw KeyError. It is
    request-specific/permanent (retrying as-is won't help), so it must surface as a
    RuntimeError — a strike candidate under the daemon's cycle-level attribution
    (D5) — NOT LLMUnavailableError (which would retry forever).
    """

    async def test_missing_content_key_raises_runtime_error(self, cloud_client):
        """A message with no `content` key raises RuntimeError, not KeyError."""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"role": "assistant"}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(RuntimeError, match="no content") as exc_info:
                await cloud_client.complete("sys", "user")
            msg = str(exc_info.value)
            assert "test-cloud-model" in msg
            assert "max_tokens" in msg  # names the likely cause

    async def test_content_less_raises_llm_content_error(self, cloud_client):
        """The content-less guard raises the dedicated LLMContentError type so the
        newsletter pipeline can re-raise it pipeline-wide (issue #30). It must
        subclass RuntimeError so the email pipeline's `except RuntimeError` arm
        still records it as a strike candidate for the cycle-level attribution
        (D5) — give-up is then decided post-gather by correlation."""
        assert issubclass(LLMContentError, RuntimeError)
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"role": "assistant", "content": None}}]
        })
        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            with pytest.raises(LLMContentError, match="no content"):
                await cloud_client.complete("sys", "user")

    async def test_none_content_raises_runtime_error(self, cloud_client):
        """An explicit `content: null` raises RuntimeError, not a downstream crash."""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"role": "assistant", "content": None}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(RuntimeError, match="no content"):
                await cloud_client.complete("sys", "user")

    async def test_empty_string_content_raises_runtime_error(self, cloud_client):
        """An empty-but-present `content: ""` is just as unusable as null/missing.

        A reasoning model that exhausts max_tokens mid-think can emit `content: ""`
        rather than null. That must raise the same no-content RuntimeError, not fall
        through to be parsed as an empty classification: Stage 1 would read it as
        SERVICE — the deliberate best-effort default (D2) — and route a person
        thread's body to the cloud. (The Stage-2 LOW_PRIORITY→archive default this
        docstring used to name is gone since D5 Rule 1, Wave 2 T10.)"""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"role": "assistant", "content": ""}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(RuntimeError, match="no content"):
                await cloud_client.complete("sys", "user")

    async def test_glm_reasoning_only_no_content_raises_runtime_error(self):
        """GLM returning only reasoning_content (max_tokens exhausted mid-think) raises
        RuntimeError, not KeyError — the cloud-tier case from issue #11's comment.

        There is no final answer, only reasoning, so even include_thinking mode must
        not silently treat a content-less reply as a classification.
        """
        glm_client = LLMClient(
            base_url="https://api.example.com/v1/chat/completions",
            api_key="sk-test",
            model="zai-org/glm-5",
        )
        mock_response = _mock_response(json_data={
            "choices": [{"message": {
                "role": "assistant",
                "reasoning_content": "Let me think about whether this is a person...",
            }}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(RuntimeError, match="no content"):
                await glm_client.complete("sys", "user", include_thinking=True)


class TestFinishReasonLength:
    """A finish_reason of "length" must be loud, never silently parsed (issue #64).

    Measured hazard (as it stood in #64): with thinking off, an over-budget
    decode returns non-empty but truncated content, cut before any label, and
    parse_email_label silently defaulted it to LOW_PRIORITY (action: ARCHIVE)
    with only a parse WARNING. That default is gone (D5 Rule 1, Wave 2 T10),
    but the guard here is still the right place to fail: it names the real
    cause — the budget — with the response diagnostics attached, instead of
    surfacing as an unexplained unparseable reply a cycle later, and it also
    covers Stage 1, whose SERVICE default deliberately stays (D2).
    And the empty-content variant's old error message hardcoded a
    guess ("max_tokens likely exhausted; reasoning_content ...") instead of
    reporting the response's actual finish_reason and which reasoning field was
    populated (Ollama's is `reasoning`, not `reasoning_content`).
    """

    async def test_length_with_empty_content_reports_real_diagnostics(self, local_client):
        """The exact production signature (issue #64): HTTP 200, finish_reason
        "length", content == "", reasoning in Ollama's `reasoning` field. The
        error must carry the REAL finish_reason, a prompt-size estimate, and
        name the actual populated reasoning field."""
        resp = _mock_response(json_data={
            "choices": [{
                "finish_reason": "length",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning": "Step 1: the sender appears to be a colleague...",
                },
            }]
        })
        with pytest.raises(LLMContentError) as exc_info:
            await _post_canned(local_client, resp)
        msg = str(exc_info.value)
        assert "finish_reason='length'" in msg
        assert "max_tokens" in msg
        assert "prompt ~" in msg  # prompt token estimate, not a hardcoded guess
        # Names the field that actually held the reasoning (`reasoning`), not
        # the hardcoded `reasoning_content` guess — and not the absent-field
        # text, which is what a diagnostics regression would fall back to.
        assert "reasoning in reasoning" in msg
        assert "reasoning_content" not in msg
        assert "no reasoning field" not in msg

    async def test_length_with_truncated_content_raises_not_parses(self, local_client):
        """finish_reason "length" with NON-empty content is a truncated answer:
        it must raise (a strike candidate under the daemon's cycle-level
        attribution, D5), never return for parsing — the truncation lands
        mid-scaffold before any label, so any label read out of it is an
        artifact of where the budget ran out."""
        resp = _mock_response(json_data={
            "choices": [{
                "finish_reason": "length",
                "message": {
                    "role": "assistant",
                    "content": "1. Who is this from? The sender appears to be",
                },
            }]
        })
        with pytest.raises(LLMContentError, match="truncated"):
            await _post_canned(local_client, resp)

    async def test_length_with_truncated_content_raises_in_include_thinking_mode(self, local_client):
        """The truncation guard must also cover the include_thinking path (evals)."""
        resp = _mock_response(json_data={
            "choices": [{
                "finish_reason": "length",
                "message": {"role": "assistant", "content": "1. Who is this"},
            }]
        })
        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = resp
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            with pytest.raises(LLMContentError, match="truncated"):
                await local_client.complete("sys", "user", include_thinking=True)

    async def test_stop_with_content_returns_normally(self, local_client):
        """A normal completion (finish_reason "stop") is unaffected."""
        resp = _mock_response(json_data={
            "choices": [{"finish_reason": "stop", "message": {"content": "FYI"}}]
        })
        assert await _post_canned(local_client, resp) == "FYI"

    async def test_missing_finish_reason_with_content_returns_normally(self, local_client):
        """Responses that omit finish_reason entirely (many mocks, some servers)
        keep working — absence is not "length"."""
        resp = _mock_response(json_data={
            "choices": [{"message": {"content": "NEEDS_RESPONSE"}}]
        })
        assert await _post_canned(local_client, resp) == "NEEDS_RESPONSE"

    async def test_empty_content_without_finish_reason_names_absent_reasoning(self, cloud_client):
        """Empty content stays loud even without finish_reason, and the error
        says no reasoning field was populated rather than guessing one was.
        The reported finish_reason must be the response's REAL (absent) value —
        this is the case that distinguishes real reporting from a hardcoded
        'length'."""
        resp = _mock_response(json_data={
            "choices": [{"message": {"role": "assistant", "content": ""}}]
        })
        with pytest.raises(LLMContentError) as exc_info:
            await _post_canned(cloud_client, resp)
        msg = str(exc_info.value)
        assert "no content" in msg
        assert "no reasoning field" in msg
        assert "finish_reason=None" in msg

    async def test_length_with_inline_think_content_names_inline_channel(self, local_client):
        """When the reasoning that ate the budget sits inline in content as an
        (unclosed) <think> block — the mlx_lm.server / LM Studio think-on
        truncation shape, the original #10/#30 failure — the diagnostics must
        say so instead of claiming "no reasoning field": an operator triaging
        logs would otherwise conclude the model emitted no reasoning at all."""
        resp = _mock_response(json_data={
            "choices": [{
                "finish_reason": "length",
                "message": {
                    "role": "assistant",
                    "content": "<think>step 1: the sender appears to be... step 2:",
                },
            }]
        })
        with pytest.raises(LLMContentError) as exc_info:
            await _post_canned(local_client, resp)
        msg = str(exc_info.value)
        assert "inline <think>" in msg
        assert "no reasoning field" not in msg


class TestThinkOnlyResponse:
    """Non-empty content that strips to NOTHING — a fully-closed think-only
    response (finish_reason "stop") — is as unusable as no content: returning
    "" would carry a non-answer downstream, where Stage 1 still reads it as
    SERVICE (D2) and Stage 2 raises a cause-less parse failure instead of
    naming the empty reply. Reachable
    whenever a backend leaves thinking effectively on (e.g. mlx_lm.server
    ignoring reasoning_effort) and the model closes its think block without an
    answer."""

    async def test_closed_think_only_content_raises(self, local_client):
        resp = _mock_response(json_data={
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "<think>the sender is a colleague... FYI probably</think>",
                },
            }]
        })
        with pytest.raises(LLMContentError, match="only reasoning"):
            await _post_canned(local_client, resp)

    async def test_closed_think_only_raises_in_include_thinking_mode(self, local_client):
        """The guard must also cover the include_thinking path (evals)."""
        resp = _mock_response(json_data={
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "<think>reasoning without an answer</think>",
                },
            }]
        })
        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = resp
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            with pytest.raises(LLMContentError, match="only reasoning"):
                await local_client.complete("sys", "user", include_thinking=True)


class TestProbe:
    """probe() carries status detail so preflight can distinguish a 404
    (model-name mismatch) from an unreachable endpoint (issue #41 item 7)."""

    async def test_probe_ok_on_200(self, local_client):
        mock_response = _mock_response(json_data={"choices": [{"message": {"content": "ok"}}]})
        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await local_client.probe()
            assert result.ok is True
            assert result.status_code == 200
            assert result.error is None

    async def test_probe_reports_404_status(self, local_client):
        mock_response = _mock_response(status_code=404, json_data={"error": "not found"})
        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await local_client.probe()
            assert result.ok is False
            assert result.status_code == 404

    async def test_probe_error_on_connection_failure(self, local_client):
        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await local_client.probe()
            assert result.ok is False
            assert result.status_code is None
            assert "ConnectError" in result.error

    async def test_probe_error_on_unsupported_protocol(self, local_client):
        """An unset/schemeless URL makes httpx raise UnsupportedProtocol; probe()
        relies on this being caught (ok=False), not propagated."""
        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.UnsupportedProtocol("missing scheme")
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await local_client.probe()
            assert result.ok is False

    async def test_probe_default_timeout_is_longer_than_10s(self, local_client):
        """The ping timeout defaults to DEFAULT_AVAILABILITY_TIMEOUT (>10s) so a
        cold on-demand model load is not mistaken for an unreachable server."""
        assert DEFAULT_AVAILABILITY_TIMEOUT > 10
        mock_response = _mock_response(json_data={"choices": [{"message": {"content": "ok"}}]})
        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await local_client.probe()

            assert mock_client_cls.call_args.kwargs["timeout"] == DEFAULT_AVAILABILITY_TIMEOUT

    async def test_probe_explicit_timeout_is_honored(self, local_client):
        mock_response = _mock_response(json_data={"choices": [{"message": {"content": "ok"}}]})
        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await local_client.probe(timeout=123)

            assert mock_client_cls.call_args.kwargs["timeout"] == 123


class TestAvailabilityDetail:
    """AvailabilityResult.detail() is the single place that composes a probe
    failure's diagnosis (Findings 1 & 2): the HTTP status (with the 404
    model-mismatch hint) or the exception text, so every preflight can surface it
    instead of discarding it."""

    def test_detail_empty_when_ok(self):
        assert AvailabilityResult(ok=True, status_code=200).detail() == ""

    def test_detail_includes_non_404_status_code(self):
        detail = AvailabilityResult(ok=False, status_code=401).detail()
        assert "401" in detail
        # A non-404 status is not a model-mismatch, so no such hint.
        assert "model name does not match" not in detail

    def test_detail_404_includes_model_mismatch_hint(self):
        detail = AvailabilityResult(ok=False, status_code=404).detail()
        assert "404" in detail
        assert "model name does not match" in detail

    def test_detail_includes_error_text(self):
        detail = AvailabilityResult(
            ok=False, error="ConnectError: Connection refused"
        ).detail()
        assert "ConnectError" in detail
        assert "Connection refused" in detail

    def test_detail_empty_when_no_status_or_error(self):
        # ok=False with neither a status nor an error (e.g. a bare down probe) has
        # nothing specific to say.
        assert AvailabilityResult(ok=False).detail() == ""


class TestExtraBody:
    async def test_extra_body_merged_into_request(self):
        """Extra body fields are merged into the API request."""
        client = LLMClient(
            base_url="http://localhost:8080/v1/chat/completions",
            api_key="",
            model="qwen3",
            extra_body={"enable_thinking": False},
        )
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"content": "SERVICE"}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await client.complete("sys", "user")

            body = mock_client.post.call_args.kwargs["json"]
            assert body["enable_thinking"] is False
            assert body["model"] == "qwen3"

    async def test_nested_extra_body(self):
        """Nested extra body (e.g. chat_template_kwargs) is preserved."""
        client = LLMClient(
            base_url="http://localhost:8080/v1/chat/completions",
            api_key="",
            model="qwen3",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"content": "PERSON"}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await client.complete("sys", "user")

            body = mock_client.post.call_args.kwargs["json"]
            assert body["chat_template_kwargs"] == {"enable_thinking": False}

    async def test_no_extra_body_by_default(self, cloud_client):
        """Without extra_body, request contains only standard fields."""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"content": "SERVICE"}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await cloud_client.complete("sys", "user")

            body = mock_client.post.call_args.kwargs["json"]
            assert set(body.keys()) == {"model", "max_tokens", "temperature", "messages"}

    async def test_extra_body_in_availability_check(self):
        """Extra body fields are included in the probe() ping."""
        client = LLMClient(
            base_url="http://localhost:8080/v1/chat/completions",
            api_key="",
            model="qwen3",
            extra_body={"enable_thinking": False},
        )
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"content": "ok"}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await client.probe()

            body = mock_client.post.call_args.kwargs["json"]
            assert body["enable_thinking"] is False


class TestGLMReasoningContent:
    """Tests for GLM-style reasoning_content extraction."""

    @pytest.fixture
    def glm_client(self):
        return LLMClient(
            base_url="https://api.example.com/v1/chat/completions",
            api_key="sk-test",
            model="zai-org/glm-5",
        )

    async def test_glm_model_detection(self):
        """GLM models are detected by name."""
        assert LLMClient(base_url="", api_key="", model="zai-org/glm-5")._is_glm_model()
        assert LLMClient(base_url="", api_key="", model="zai-org/glm-4.7-flash")._is_glm_model()
        assert LLMClient(base_url="", api_key="", model="GLM-4")._is_glm_model()
        assert not LLMClient(base_url="", api_key="", model="deepseek/deepseek-v3")._is_glm_model()
        assert not LLMClient(base_url="", api_key="", model="qwen/qwen3-32b")._is_glm_model()

    async def test_glm_injects_thinking_param(self, glm_client):
        """GLM models get thinking param injected when include_thinking=True."""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"content": "SERVICE", "reasoning_content": "This is a service email."}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await glm_client.complete("sys", "user", include_thinking=True)

            body = mock_client.post.call_args.kwargs["json"]
            assert body["thinking"] == {"type": "enabled"}

    async def test_glm_no_think_extra_body_disables_native_thinking(self):
        """--no-think (chat_template_kwargs.enable_thinking=False) must disable GLM's
        native thinking, not be contradicted by an unconditional thinking=enabled.

        Regression (review finding #4): GLM is the default cloud model, so the
        documented thinking on/off A/B run would otherwise send enable_thinking=false
        AND thinking={type:enabled} in the same body.
        """
        client = LLMClient(
            base_url="https://api.example.com/v1/chat/completions",
            api_key="sk-test",
            model="zai-org/glm-5",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"content": "SERVICE"}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await client.complete("sys", "user", include_thinking=True)

            body = mock_client.post.call_args.kwargs["json"]
            assert body["thinking"] == {"type": "disabled"}

    async def test_glm_top_level_enable_thinking_false_disables_native_thinking(self):
        """A top-level enable_thinking=False in extra_body also disables GLM thinking."""
        client = LLMClient(
            base_url="https://api.example.com/v1/chat/completions",
            api_key="sk-test",
            model="zai-org/glm-5",
            extra_body={"enable_thinking": False},
        )
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"content": "SERVICE"}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await client.complete("sys", "user", include_thinking=True)

            body = mock_client.post.call_args.kwargs["json"]
            assert body["thinking"] == {"type": "disabled"}

    async def test_glm_explicit_thinking_field_is_not_overridden(self):
        """An explicit `thinking` field in extra_body wins over the auto-injection."""
        client = LLMClient(
            base_url="https://api.example.com/v1/chat/completions",
            api_key="sk-test",
            model="zai-org/glm-5",
            extra_body={"thinking": {"type": "disabled"}},
        )
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"content": "SERVICE"}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await client.complete("sys", "user", include_thinking=True)

            body = mock_client.post.call_args.kwargs["json"]
            assert body["thinking"] == {"type": "disabled"}

    async def test_glm_no_thinking_param_without_include(self, glm_client):
        """GLM models don't get thinking param when include_thinking=False."""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"content": "SERVICE"}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await glm_client.complete("sys", "user")

            body = mock_client.post.call_args.kwargs["json"]
            assert "thinking" not in body

    async def test_glm_extracts_reasoning_content(self, glm_client):
        """GLM reasoning_content is extracted as thinking."""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {
                "content": "SERVICE",
                "reasoning_content": "The sender is automated — this is a shipping notification.",
            }}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            response, thinking = await glm_client.complete("sys", "user", include_thinking=True)
            assert response == "SERVICE"
            assert "shipping notification" in thinking

    async def test_glm_falls_back_to_inline_tags(self, glm_client):
        """GLM client falls back to inline tag extraction if reasoning_content is absent."""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"content": "<think>some reasoning</think>\nSERVICE"}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            response, thinking = await glm_client.complete("sys", "user", include_thinking=True)
            assert response == "SERVICE"
            assert "some reasoning" in thinking

    async def test_separate_and_inline_channels_both_captured_for_non_glm(self):
        """A populated separate reasoning field is read for ANY model, not just
        GLM (issue #64: capture used to be GLM-gated, so Ollama-served local
        models silently yielded thinking=""). Nothing "wins": the inline
        channel is captured alongside it — the inline block is stripped from
        the returned content, so dropping it here would lose it entirely."""
        client = LLMClient(
            base_url="https://api.example.com/v1/chat/completions",
            api_key="sk-test",
            model="deepseek/deepseek-v3",
        )
        mock_response = _mock_response(json_data={
            "choices": [{"message": {
                "content": "<think>inline reasoning</think>\nSERVICE",
                "reasoning_content": "separate-channel reasoning",
            }}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            response, thinking = await client.complete("sys", "user", include_thinking=True)
            assert response == "SERVICE"  # inline tags still stripped from content
            assert "separate-channel reasoning" in thinking
            assert "inline reasoning" in thinking

    async def test_glm_empty_reasoning_content_falls_back(self, glm_client):
        """GLM with empty reasoning_content falls back to inline tags."""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {
                "content": "<think>inline thought</think>\nSERVICE",
                "reasoning_content": "",
            }}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            response, thinking = await glm_client.complete("sys", "user", include_thinking=True)
            assert response == "SERVICE"
            assert "inline thought" in thinking


class TestSeparateReasoningFieldCapture:
    """include_thinking must read the separate reasoning field for ANY model
    (issue #64). It was GLM-gated, so under Ollama — which routes Qwen's
    reasoning to `msg.reasoning`, not `reasoning_content` — every local call
    returned thinking="" and the eval .cot.jsonl sidecar was silently empty,
    documented as if the model had emitted no reasoning.
    """

    def _qwen_ollama_client(self):
        return LLMClient(
            base_url="http://ollama.example.com/v1/chat/completions",
            api_key="",
            model="qwen3.6:35b-mlx",
        )

    async def test_ollama_reasoning_field_yields_thinking(self):
        """The Ollama response shape (reasoning in `msg.reasoning`, bare-prose
        content) must yield non-empty thinking for a non-GLM model."""
        mock_response = _mock_response(json_data={
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "1. The sender is a colleague.\n\nFYI",
                    "reasoning": "Okay, let me work through the three questions...",
                },
            }]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            response, thinking = await self._qwen_ollama_client().complete(
                "sys", "user", include_thinking=True,
            )
            assert "FYI" in response
            assert "three questions" in thinking

    async def test_no_separate_field_falls_back_to_inline_tags(self):
        """Without a separate field, inline <think> extraction still works
        (DeepSeek/Qwen served by mlx_lm.server with thinking on)."""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {
                "content": "<think>inline path</think>\nLOW_PRIORITY",
            }}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            response, thinking = await self._qwen_ollama_client().complete(
                "sys", "user", include_thinking=True,
            )
            assert response == "LOW_PRIORITY"
            assert "inline path" in thinking

    async def test_untagged_content_with_no_reasoning_yields_empty_thinking(self):
        """Thinking-off Ollama shape: no separate field, no tags — the CoT is
        the untagged content itself (preserved in the response / label_raw), so
        thinking is legitimately ""."""
        mock_response = _mock_response(json_data={
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "1. A vendor.\n2. No action.\n\nLOW_PRIORITY",
                },
            }]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            response, thinking = await self._qwen_ollama_client().complete(
                "sys", "user", include_thinking=True,
            )
            assert thinking == ""
            assert "LOW_PRIORITY" in response

    async def test_both_channels_captured_when_model_emits_both(self):
        """Issue #64 finding: with thinking ON and budget to spare, the model
        emitted BOTH native reasoning AND <think>-tagged content CoT. Both must
        land in thinking — the inline block is stripped from the returned
        content, so capture that takes only the separate field would lose the
        inline channel entirely (in neither thinking nor label_raw)."""
        mock_response = _mock_response(json_data={
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "content": "<think>inline channel</think>\nFYI",
                    "reasoning": "native channel",
                },
            }]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            response, thinking = await self._qwen_ollama_client().complete(
                "sys", "user", include_thinking=True,
            )
            assert response == "FYI"
            assert "native channel" in thinking
            assert "inline channel" in thinking

    async def test_reasoning_effort_none_recognized_as_thinking_disable(self):
        """reasoning_effort="none" (the Ollama disable dialect, now in the
        shipped local config) must read as thinking-disabled intent — so a GLM
        client carrying it injects thinking=disabled instead of contradicting
        it with enabled."""
        client = LLMClient(
            base_url="https://api.example.com/v1/chat/completions",
            api_key="sk-test",
            model="zai-org/glm-5",
            extra_body={"reasoning_effort": "none"},
        )
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"content": "SERVICE"}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await client.complete("sys", "user", include_thinking=True)

            body = mock_client.post.call_args.kwargs["json"]
            assert body["thinking"] == {"type": "disabled"}

    async def test_reasoning_effort_other_values_do_not_read_as_disable(self):
        """Only "none" disables (Ollama treats "low" as a silent no-op), so only
        "none" may read as disable intent."""
        client = LLMClient(
            base_url="", api_key="", model="qwen3",
            extra_body={"reasoning_effort": "low"},
        )
        assert client._extra_body_disables_thinking() is False

    async def test_reasoning_effort_none_recognized_case_insensitively(self):
        """A hand-edited config or --extra-body JSON may spell it "None"/"NONE";
        case must not silently flip a think-off A/B arm back to thinking-on."""
        client = LLMClient(
            base_url="https://api.example.com/v1/chat/completions",
            api_key="sk-test",
            model="zai-org/glm-5",
            extra_body={"reasoning_effort": "None"},
        )
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"content": "SERVICE"}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await client.complete("sys", "user", include_thinking=True)

            body = mock_client.post.call_args.kwargs["json"]
            assert body["thinking"] == {"type": "disabled"}

    async def test_non_string_reasoning_field_is_ignored(self):
        """A backend/proxy may surface the reasoning channel as structured data
        (e.g. an OpenRouter-style list of reasoning blocks). Non-string values
        must not crash the thinking join — fall back to the inline channel, as
        the old GLM-gated code did for fields it didn't read."""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {
                "content": "<think>inline fallback</think>\nFYI",
                "reasoning": [{"type": "reasoning.text", "text": "structured"}],
            }}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            response, thinking = await self._qwen_ollama_client().complete(
                "sys", "user", include_thinking=True,
            )
            assert response == "FYI"
            assert thinking == "inline fallback"

    async def test_both_separate_fields_captured_when_populated(self):
        """Some proxies populate BOTH reasoning_content and reasoning — the
        error-path diagnostics already treat them as simultaneously reportable.
        Neither channel may be silently dropped from the sidecar."""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {
                "content": "FYI",
                "reasoning_content": "CHANNEL_A",
                "reasoning": "CHANNEL_B",
            }}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            response, thinking = await self._qwen_ollama_client().complete(
                "sys", "user", include_thinking=True,
            )
            assert response == "FYI"
            assert "CHANNEL_A" in thinking
            assert "CHANNEL_B" in thinking

    async def test_mirrored_separate_fields_not_duplicated(self):
        """A proxy that mirrors the same text into both separate fields must
        not double it in the captured thinking."""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {
                "content": "FYI",
                "reasoning_content": "SAME TEXT",
                "reasoning": "SAME TEXT",
            }}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            _, thinking = await self._qwen_ollama_client().complete(
                "sys", "user", include_thinking=True,
            )
            assert thinking == "SAME TEXT"


class TestBalanceErrorProvenance:
    """LLMBalanceError carries structured provenance (issue #73, D22): the daemon
    composes the halt notification — provider tier, model, HTTP status, the
    provider's reason text — from attributes, not by parsing the message, and
    picks which client to re-probe by ``tier``."""

    async def test_balance_error_carries_tier_model_status_and_detail(self):
        client = LLMClient(
            base_url="https://api.cloud.example.com/v1/chat/completions",
            api_key="sk-test-key", model="zai-org/glm-5", tier="cloud",
        )
        with pytest.raises(LLMBalanceError) as exc_info:
            await _post_canned(client, _mock_response(403, NOVITA_BALANCE_BODY))
        exc = exc_info.value
        assert exc.tier == "cloud"
        assert exc.model == "zai-org/glm-5"
        assert exc.status_code == 403
        assert "NOT_ENOUGH_BALANCE" in exc.detail
        # The message text the logs already carry is unchanged.
        assert "status 403" in str(exc)

    def test_bare_construction_still_works(self):
        """Tests and evals raise it with a message alone; provenance defaults empty."""
        exc = LLMBalanceError("out of funds")
        assert exc.tier is None
        assert exc.model is None
        assert exc.status_code is None
        assert exc.detail == ""
