"""LLM client abstraction for cloud and local endpoints.

Supports any OpenAI-compatible chat completion API.
Handles thinking tag stripping for reasoning models.
"""

import re
from dataclasses import dataclass

import httpx

from retry import retry_with_backoff

# Default timeout (seconds) for the lightweight probe() ping. Longer than
# a typical liveness probe because a server that loads models on demand only
# loads the requested model on the first request — a cold load of a large model
# routinely exceeds 10s, and timing out would wrongly report it unreachable.
DEFAULT_AVAILABILITY_TIMEOUT = 60


@dataclass(frozen=True)
class AvailabilityResult:
    """Result of an endpoint availability probe (issue #41 item 7).

    ``ok`` is True only on HTTP 200. ``status_code`` is the HTTP status when a
    response arrived (None if none did); ``error`` is the exception detail when
    the request failed before a response (connect/timeout/unsupported-protocol).
    """

    ok: bool
    status_code: int | None = None
    error: str | None = None

    def detail(self) -> str:
        """One-line failure diagnosis for a preflight message (issue #41 item 7).

        The single place that composes a failed probe's specific reason so every
        consumer surfaces it rather than discarding it (Findings 1 & 2): the HTTP
        status when a response arrived — with the 404-means-model-name-mismatch
        hint, distinct from a merely-down endpoint — or the exception text when no
        response arrived at all (connect/timeout/UnsupportedProtocol). Returns ""
        when ``ok`` or when there is nothing specific to report.
        """
        if self.ok:
            return ""
        if self.status_code is not None:
            if self.status_code == 404:
                return "HTTP 404 — likely the model name does not match the served model"
            return f"HTTP {self.status_code}"
        if self.error:
            return self.error
        return ""


class LLMUnavailableError(Exception):
    """The LLM endpoint can't serve requests right now (provider-shaped, transient).

    Raised for transport-level unavailability — endpoint unreachable, connect/
    pool timeout, connection dropped mid-request — and for provider-shaped
    responses: an exhausted 429 (retry.py already spent its budget) or any 5xx.
    Either way the provider can't serve anyone, so the fault is never one
    thread's blame: the daemon defers and retries next cycle and never counts
    it toward give-up (decision D5). Request-specific failures stay distinct —
    a read/write timeout is TimeoutError and any other non-200 is RuntimeError,
    both strike candidates under the daemon's cycle-level attribution.

    ``tier`` identifies the raising client ("cloud" / "local" / None when the
    client didn't declare one). Both tiers raise this same type; the tier is what
    lets the daemon treat a routine local outage (the MLX laptop is offline for
    hours at a time) as INFO-grade while a cloud outage stays WARNING (issue #24),
    and what excludes local-tier outages from masquerade tracking (D5).
    """

    def __init__(self, message: str, *, tier: str | None = None):
        super().__init__(message)
        self.tier = tier


class LLMContentError(RuntimeError):
    """The model returned no usable ``content`` (issues #30, #64).

    A reasoning model that exhausts max_tokens mid-<think>, a GLM reply carrying
    only ``reasoning_content``, or an empty string all yield an unusable response.
    So does a response whose ``finish_reason`` is ``"length"`` even when content
    is non-empty: the answer was cut at the max_tokens budget, so any label read
    out of it is an artifact of the truncation (issue #64). Retrying as-is
    won't help, so it is request-specific/permanent — a strike candidate under
    the daemon's cycle-level attribution (decision D5). It subclasses
    ``RuntimeError`` so the email pipeline's ``except RuntimeError`` arm catches
    it unchanged, while giving the newsletter pipeline a dedicated type to
    re-raise (a *bare* per-story RuntimeError stays isolated; a content-less one
    propagates pipeline-wide).
    """


# Bodies that mean "the account is out of funds/credit", across providers:
# Novita (NOT_ENOUGH_BALANCE), OpenAI-style (insufficient_quota), Anthropic
# ("credit balance is too low", sent as HTTP 400). Deliberately narrow —
# distinctive tokens only, no prose like "insufficient balance" that an error
# body could echo from the email being classified (a bank alert must never
# read as the provider being broke). DeepSeek-style "Insufficient Balance"
# arrives with HTTP 402, which is treated as a balance error on status alone.
_BALANCE_SIGNATURE = re.compile(
    r"not_enough_balance|insufficient_balance|insufficient_quota"
    r"|credit balance is too low",
    re.IGNORECASE,
)

# Only non-retryable statuses may signal funds exhaustion (402 unconditionally;
# 400/403 when the body matches). 429 is excluded even though some providers
# phrase hard quota exhaustion identically to a per-minute rate limit
# ("exceeded your current quota"): wrongly converting a transient rate limit
# into a function halt (self-healing since D22, but still an hour's outage and
# a push) is worse than letting a rare 429-signaled
# out-of-funds be retried as provider unavailability (decision D19) — it
# defers threads each cycle and never strikes (D5). Its visibility is the
# per-thread WARNING and the cycle summary, not an ERROR: an account-wide 429
# fails every thread, and neither loud path covers that (the shared-cause line
# is emitted over candidate failures only, and the masquerade escalation wants
# exactly one affected thread while siblings succeed).
_BALANCE_SIGNATURE_STATUSES = frozenset({400, 403})


class LLMBalanceError(RuntimeError):
    """The provider account is out of funds/credit (e.g. Novita's 403 NOT_ENOUGH_BALANCE).

    Account-wide, not request-specific: if one request fails for lack of balance,
    every subsequent request to the same provider will too. The daemon therefore
    treats this as a halt condition (stop the affected function, notify the
    operator, re-probe the provider on a slow schedule and resume when it
    answers — decision D22) rather than a per-thread give-up: the failing thread
    is left unprocessed so it is retried once the function resumes. The halt is
    per-FUNCTION (decision D5's scope rule, D19): this error carries no function
    provenance, so the daemon's call site decides which function stops — email
    triage keeps running when the newsletter provider is the broke one, and vice
    versa. Subclasses ``RuntimeError`` so callers unaware of it (evals) still see
    a generic LLM failure; the daemon must catch it *before* its
    ``except RuntimeError`` arm.

    Structured provenance (issue #73), all optional so a bare
    ``LLMBalanceError("msg")`` still constructs: ``tier`` (the raising client's
    tier — which of the email function's two clients to re-probe), ``model``,
    ``status_code`` and ``detail`` (the provider's response body, truncated) —
    the fields the halt notification is composed from, without parsing the
    message text.
    """

    def __init__(
        self,
        message: str,
        *,
        tier: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        detail: str = "",
    ):
        super().__init__(message)
        self.tier = tier
        self.model = model
        self.status_code = status_code
        self.detail = detail


class LLMClient:
    """Client for OpenAI-compatible chat completion endpoints."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int = 8096,
        temperature: float = 0.2,
        timeout: int = 60,
        extra_body: dict | None = None,
        tier: str | None = None,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.extra_body = extra_body or {}
        # "cloud" / "local" — carried on LLMUnavailableError so callers can tell
        # a routine local outage from a surprising cloud one (issue #24).
        self.tier = tier

    def _provider(self) -> str:
        """Identify this client in error messages: which of the (up to 3) LLMClient
        instances — cloud, local MLX, newsletter — a failure came from."""
        return f"tier={self.tier or '-'} model={self.model} url={self.base_url}"

    def _is_glm_model(self) -> bool:
        """Check if model uses GLM-style reasoning_content instead of inline think tags."""
        return "glm" in self.model.lower()

    def _extra_body_disables_thinking(self) -> bool:
        """True if extra_body asks to turn thinking off.

        Recognizes the chat_template_kwargs.enable_thinking form (Qwen / LM
        Studio), a top-level enable_thinking flag, and reasoning_effort="none"
        (the Ollama dialect — only "none" disables there; "low" etc. are model
        defaults, not a disable). Matched case-insensitively: a hand-edited
        config or --extra-body JSON spelling it "None" still carries disable
        intent, and missing it would silently flip a think-off A/B arm back to
        thinking-on. GLM ignores all of those fields, so complete() uses this
        to set GLM's native thinking field to disabled rather than
        contradicting the request with enabled.
        """
        ctk = self.extra_body.get("chat_template_kwargs")
        if isinstance(ctk, dict) and ctk.get("enable_thinking") is False:
            return True
        effort = self.extra_body.get("reasoning_effort")
        if isinstance(effort, str) and effort.lower() == "none":
            return True
        return self.extra_body.get("enable_thinking") is False

    async def complete(
        self, system_prompt: str, user_content: str, include_thinking: bool = False,
    ) -> str | tuple[str, str]:
        """Send a chat completion request and return the stripped response.

        Args:
            system_prompt: System message for the LLM.
            user_content: User message content.
            include_thinking: If True, return (stripped, thinking) tuple.

        Returns:
            If include_thinking is False: stripped response string.
            If include_thinking is True: (stripped_response, thinking_content) tuple.

        Raises:
            TimeoutError: If the request times out (read/write) — request-specific
                (e.g. input too large to prefill within the timeout); a strike
                candidate under the daemon's cycle attribution (D5).
            LLMUnavailableError: If the endpoint can't serve the request right
                now — unreachable (connection refused, connect/pool timeout),
                dropped mid-request, or answering with an exhausted 429 or any
                5xx. Provider-shaped (D5): deferred and retried next cycle,
                never counted toward give-up.
            LLMBalanceError: If the non-200 response says the provider account is
                out of funds (402, or a 400/403 whose body carries a balance
                signature) — account-wide; the function whose provider this is
                halts (D5 scope, D19) rather than giving up per-thread.
            LLMContentError: If the response carries no usable answer — empty/
                missing content, or finish_reason "length" (answer truncated at
                the max_tokens budget) — request-specific, a strike candidate.
            RuntimeError: If the LLM returns any other non-200 response
                (400/401/403/404/422… without a balance signature) —
                request-specific, a strike candidate.
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            **self.extra_body,
        }

        # GLM models require an explicit thinking field. Honor a disable request
        # from extra_body (e.g. --no-think, which GLM otherwise ignores) instead of
        # contradicting it with enabled, and never override a `thinking` field the
        # caller set explicitly in extra_body.
        if include_thinking and self._is_glm_model() and "thinking" not in body:
            disabled = self._extra_body_disables_thinking()
            body["thinking"] = {"type": "disabled" if disabled else "enabled"}

        async def _do_request() -> httpx.Response:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                return await client.post(self.base_url, headers=headers, json=body)

        try:
            response = await retry_with_backoff(_do_request, f"LLM {self.model}")
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
            # Couldn't establish a connection (server down / unreachable / pool
            # exhausted). This is endpoint *unavailability*, not a problem with this
            # request, so callers retry next cycle and never count it toward giving
            # up on a thread. NOTE: ConnectTimeout/PoolTimeout subclass
            # TimeoutException, so they must be caught before the timeout clause.
            raise LLMUnavailableError(
                f"LLM endpoint {self.model} unavailable ({type(exc).__name__}): {exc}",
                tier=self.tier,
            ) from exc
        except httpx.TimeoutException:
            # Connection established but the response was too slow (read/write
            # timeout) — request-specific (e.g. a transcript too large to prefill
            # within the timeout). Surfaced as TimeoutError so the daemon may give
            # up on one oversized thread rather than retry it forever.
            raise TimeoutError(
                f"LLM request to {self.model} timed out after {self.timeout}s"
            ) from None
        except httpx.TransportError as exc:
            # Connection established then dropped/reset mid-request (ReadError,
            # WriteError, RemoteProtocolError, ...): the server crashed, restarted,
            # or flapped — an availability event, also transient. (Some of these
            # stringify to ""; the wrapper guarantees an informative, non-empty msg.)
            raise LLMUnavailableError(
                f"LLM connection to {self.model} dropped ({type(exc).__name__}): {exc}",
                tier=self.tier,
            ) from exc

        if response.status_code != 200:
            prompt_chars = len(system_prompt) + len(user_content)
            resp_body = response.text[:500]
            if response.status_code == 402 or (
                response.status_code in _BALANCE_SIGNATURE_STATUSES
                and _BALANCE_SIGNATURE.search(response.text)
            ):
                raise LLMBalanceError(
                    f"LLM provider out of funds — status {response.status_code} "
                    f"[{self._provider()}]: {resp_body}",
                    tier=self.tier,
                    model=self.model,
                    status_code=response.status_code,
                    detail=resp_body,
                )
            if response.status_code == 429 or response.status_code >= 500:
                # Provider-shaped (decision D5): an exhausted 429 (retry.py already
                # spent its retry budget, so the throttle is sustained) or a 5xx
                # means the provider can't serve anyone right now — never one
                # thread's blame, so never a strike candidate. The daemon defers
                # and retries next cycle, never gives up. A fault that singles out
                # ONE thread while siblings succeed gets loud through the
                # masquerade escalation; a provider-wide one stays at the
                # per-thread WARNING plus `Processed 0/N threads`.
                raise LLMUnavailableError(
                    f"LLM endpoint unavailable — status {response.status_code} "
                    f"[{self._provider()}]: {resp_body}",
                    tier=self.tier,
                )
            raise RuntimeError(
                f"LLM request failed with status {response.status_code} "
                f"[{self._provider()}] "
                f"(prompt ~{prompt_chars // 4} tokens, {prompt_chars} chars): {resp_body}"
            )

        choice = response.json()["choices"][0]
        msg = choice["message"]
        finish_reason = choice.get("finish_reason")
        content = msg.get("content")
        stripped = self._strip_thinking(content) if content else ""
        if not stripped or finish_reason == "length":
            # Three unusable-response shapes, all request-specific/permanent
            # (retrying as-is won't help), so all raise LLMContentError — a
            # strike candidate under the daemon's cycle-level attribution (D5),
            # never LLMUnavailableError (would retry forever) or a raw KeyError
            # (issue #64):
            #   - No `content` (null, missing key, or empty string): a reasoning
            #     model exhausted max_tokens in its thinking channel, or GLM
            #     returned only reasoning_content.
            #   - Content that strips to nothing: a fully-closed think-only
            #     response — the model spent its whole turn inside <think> and
            #     never answered.
            #   - finish_reason "length" with non-empty content: the answer was
            #     cut at the max_tokens budget, usually mid-reasoning before any
            #     label.
            # Guarding here keeps the failure at its source, with the response
            # diagnostics attached: a Stage-2 reply carrying no label keyword
            # now raises downstream too (parse_email_label, D5 Rule 1), but
            # Stage 1 still reads a content-less reply as SERVICE (D2's
            # best-effort routing), so an unguarded empty answer would quietly
            # route a person thread's body to the cloud.
            # The diagnostics report the response's actual finish_reason and
            # where the reasoning actually sat — a separate field (Ollama uses
            # `reasoning`, GLM uses `reasoning_content`) or inline <think>
            # prose in content (the mlx_lm.server / LM Studio think-on shape) —
            # the old message hardcoded a guess, which masked the real failure
            # pattern in production logs.
            populated = [f for f in ("reasoning_content", "reasoning") if msg.get(f)]
            if content and "<think>" in content:
                populated.append("inline <think> tags")
            prompt_chars = len(system_prompt) + len(user_content)
            diagnostics = (
                f"finish_reason={finish_reason!r}, max_tokens={self.max_tokens}, "
                f"prompt ~{prompt_chars // 4} tokens ({prompt_chars} chars); "
                + (f"reasoning in {' + '.join(populated)}" if populated else "no reasoning field")
            )
            if not content:
                raise LLMContentError(
                    f"LLM {self.model} returned no content ({diagnostics})"
                )
            if finish_reason == "length":
                raise LLMContentError(
                    f"LLM {self.model} answer truncated at the max_tokens budget — "
                    f"{len(content)} chars of content but no trustworthy final answer "
                    f"({diagnostics})"
                )
            raise LLMContentError(
                f"LLM {self.model} returned only reasoning, no final answer — "
                f"{len(content)} chars of content strip to nothing ({diagnostics})"
            )

        if include_thinking:
            # Separate-channel reasoning is read for ANY model — the field name
            # is backend-specific (GLM: reasoning_content; Ollama: reasoning),
            # and a proxy may populate both at once, so every string-valued
            # field is captured (mirrored duplicates collapsed) — the error
            # diagnostics above already treat the fields as simultaneously
            # reportable. Non-string values (structured reasoning blocks from
            # some proxies) are ignored rather than crashing the join. This
            # used to be GLM-gated, so Ollama-served local models silently
            # yielded thinking="" (issue #64). Inline <think> tags are captured
            # TOO, not as a mere fallback: with thinking on and budget to spare
            # the model emitted both channels at once, and the inline block is
            # stripped from the returned content — dropping it here would lose
            # it entirely. With thinking disabled there is no separate field and
            # no tags: the reasoning is the untagged content itself, so "" here
            # just means "read the content".
            separate = [
                v for v in (msg.get("reasoning_content"), msg.get("reasoning"))
                if isinstance(v, str) and v
            ]
            if len(separate) == 2 and separate[0] == separate[1]:
                separate = separate[:1]
            inline = self._extract_thinking(content)
            thinking = "\n\n".join(part for part in (*separate, inline) if part)
            return stripped, thinking
        return stripped

    async def probe(self, timeout: float | None = None) -> "AvailabilityResult":
        """Probe the endpoint, returning status detail (issue #41 item 7).

        Sends a minimal completion request. ``ok`` is True only on HTTP 200.
        ``status_code`` carries the HTTP status when a response arrived — so a
        404 (which usually means the requested model name doesn't match the
        served one) is distinguishable from an endpoint that is simply down.
        ``error`` carries the exception detail when no response arrived at all.

        Args:
            timeout: Seconds to wait for the ping. Defaults to
                DEFAULT_AVAILABILITY_TIMEOUT, generous enough to cover a server
                that loads the requested model on demand (a cold load can take
                far longer than a normal liveness probe).
        """
        if timeout is None:
            timeout = DEFAULT_AVAILABILITY_TIMEOUT
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            body = {
                "model": self.model,
                "max_tokens": 1,
                "temperature": 0,
                "messages": [{"role": "user", "content": "ping"}],
                **self.extra_body,
            }

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(self.base_url, headers=headers, json=body)

            return AvailabilityResult(ok=response.status_code == 200, status_code=response.status_code)
        except httpx.RequestError as exc:
            # Any failure to obtain a response means "not available": connect /
            # timeout, but also UnsupportedProtocol (unset/schemeless URL) and
            # read/protocol errors (e.g. a dropped connection mid cold-load).
            return AvailabilityResult(ok=False, error=f"{type(exc).__name__}: {exc}")

    # Matches <think>...</think> and <<think>>...</<think>> (DeepSeek variant)
    _THINK_PATTERN = re.compile(r"<<?think>>?.*?</<?think>>?", flags=re.DOTALL)
    _THINK_EXTRACT = re.compile(r"<<?think>>?(.*?)</<?think>>?", flags=re.DOTALL)

    @staticmethod
    def _extract_thinking(content: str) -> str:
        """Extract all think blocks (handles <think> and <<think>> variants)."""
        matches = LLMClient._THINK_EXTRACT.findall(content)
        return "\n\n".join(m.strip() for m in matches)

    @staticmethod
    def _strip_thinking(content: str) -> str:
        """Remove think blocks from LLM output (handles <think> and <<think>> variants)."""
        stripped = LLMClient._THINK_PATTERN.sub("", content)
        return stripped.strip()
