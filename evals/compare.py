"""Compare the local model with a Claude model on the email golden set (issue #92).

Arm A is the local leg as ``run_eval --local-only`` runs it: Stage 2 on the
golden set's person threads, through the ``[llm.local]`` endpoint. Arm B runs the
same threads through the same ``EmailClassifier`` with a Claude model in the
local slot, so both arms see the identical prompt. Usage and the run order are
in evals/README.md.
"""

import os
from contextlib import contextmanager

import anthropic

from llm_client import LLMClient, LLMContentError

# Request shape for arm B. Effort is sent explicitly because the default differs
# by model; max_tokens covers adaptive thinking (billed as output) plus the
# step-by-step answer the prompt asks for.
ANTHROPIC_MAX_TOKENS = 16000
ANTHROPIC_EFFORT = "medium"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
# The SDK retries 408/409/429/5xx (529 "overloaded" included) and connection
# errors with exponential backoff, honouring retry-after. Each attempt is bounded
# by the timeout, which matches [llm.local]'s 180 s request budget.
ANTHROPIC_MAX_RETRIES = 4
ANTHROPIC_TIMEOUT_SECONDS = 180.0


@contextmanager
def _without_anthropic_environment():
    """Hide every ANTHROPIC_* variable while the SDK client is constructed.

    daemon.py loads .env at import, and the eval README symlinks .env to
    agent-stack's, so the environment may hold credentials the eval must not
    use (issue #92). An explicit api_key already stops the SDK reading
    ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN, but it still reads
    ANTHROPIC_BASE_URL, ANTHROPIC_CUSTOM_HEADERS and the tracing switches at
    construction; hiding them all keeps the request to the key given here and
    the fixed base URL.
    """
    hidden = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith("ANTHROPIC_")}
    try:
        yield
    finally:
        os.environ.update(hidden)


def build_anthropic_client(api_key: str, *, http_client=None) -> anthropic.AsyncAnthropic:
    """The SDK client for arm B, built from *api_key* alone."""
    with _without_anthropic_environment():
        return anthropic.AsyncAnthropic(
            api_key=api_key, base_url=ANTHROPIC_BASE_URL, http_client=http_client,
            max_retries=ANTHROPIC_MAX_RETRIES, timeout=ANTHROPIC_TIMEOUT_SECONDS,
        )


class AnthropicLLMClient:
    """Puts a Claude model behind ``LLMClient.complete``'s interface.

    ``EmailClassifier`` calls ``complete(system, user, include_thinking=True)``
    and parses the returned answer; this adapter makes that one call through the
    Messages API, so the classifier's prompt is used unchanged.
    """

    def __init__(self, client: anthropic.AsyncAnthropic, model: str):
        self._client = client
        self.model = model

    async def complete(
        self, system_prompt: str, user_content: str, include_thinking: bool = False,
    ) -> str | tuple[str, str]:
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
            output_config={"effort": ANTHROPIC_EFFORT},
        )
        # Checked before the content is read: a reply cut at max_tokens or
        # declined by a safety classifier carries no trustworthy label, so it is
        # an error for this arm (as llm_client treats finish_reason "length").
        if response.stop_reason in ("max_tokens", "refusal"):
            raise LLMContentError(
                f"Claude {self.model} stopped with stop_reason={response.stop_reason!r}; no answer used"
            )
        text = "".join(block.text for block in response.content if block.type == "text")
        stripped = LLMClient._strip_thinking(text)
        if not stripped:
            raise LLMContentError(f"Claude {self.model} returned no answer outside its reasoning")
        if include_thinking:
            native = [block.thinking for block in response.content
                      if block.type == "thinking" and block.thinking]
            thinking = "\n\n".join(part for part in (*native, LLMClient._extract_thinking(text)) if part)
            return stripped, thinking
        return stripped
