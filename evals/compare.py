"""Compare the local model with a Claude model on the email golden set (issue #92).

Arm A is the local leg as ``run_eval --local-only`` runs it: Stage 2 on the
golden set's person threads, through the ``[llm.local]`` endpoint. Arm B runs the
same threads through the same ``EmailClassifier`` with a Claude model in the
local slot, so both arms see the identical prompt. Usage and the run order are
in evals/README.md.
"""

import os
import stat
from contextlib import contextmanager
from pathlib import Path

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

# USD per million tokens, (input, output), from the claude-api skill's model
# table (cached 2026-11-03; Anthropic first-party rates). These are also the
# models --model accepts, so every run can be costed. Haiku 5.5's rate is for
# prompts up to 100K tokens; a golden-set prompt is far below that.
PRICES_AS_OF = "2026-11-03"
PRICES_PER_MTOK = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-fable-5-2": (8.00, 40.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Dollar cost of the given token counts at *model*'s published price."""
    price_in, price_out = PRICES_PER_MTOK[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


class ApiKeyError(Exception):
    """The Anthropic key could not be taken from the source Brian named."""


def obtain_key(key_file: Path | None, *, getpass_fn) -> str:
    """The Anthropic API key, from *key_file* or else typed at a hidden prompt.

    These are the only two sources: the key is not read from argv or from the
    environment (daemon.py loads the agent-stack .env, and issue #92 keeps the
    key out of that stack). A key file must be private to its owner (no group
    or other permission bits), so a root-only file deleted after the run is
    the file route the README describes.
    """
    if key_file is None:
        key = getpass_fn("Anthropic API key (input hidden): ").strip()
        if not key:
            raise ApiKeyError("the typed key is empty")
        return key
    try:
        mode = key_file.stat().st_mode
        key = key_file.read_text().strip()
    except OSError as exc:
        raise ApiKeyError(f"cannot read --anthropic-key-file {key_file}: {exc.strerror}") from None
    if mode & 0o077:
        raise ApiKeyError(
            f"--anthropic-key-file {key_file} is readable beyond its owner "
            f"(mode {stat.S_IMODE(mode):o}); chmod 600 it first"
        )
    if not key:
        raise ApiKeyError(f"--anthropic-key-file {key_file} is empty")
    return key


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
        # Totals over every reply received, usable or not: a reply cut at
        # max_tokens is billed like any other.
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

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
        self.calls += 1
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
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
