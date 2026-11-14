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
from dataclasses import dataclass
from pathlib import Path

import anthropic

from classifier import EmailClassifier
from evals.run_eval import evaluate_single, resolve_max_thread_chars
from evals.schemas import GoldenThread
from llm_client import LLMClient, LLMContentError

# The population both arms see: Stage 2 on reviewed person threads, i.e.
# `run_eval --local-only`. Person bodies are the only traffic the local model
# serves, so this is the leg issue #92 would move.
STAGES = "stage2_only"
SENDER_TYPE = "person"

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


# Token estimate for the dry run. The API's token counter is itself an API
# call, so the estimate is offline: prompt characters over a ratio. 3.5
# characters per token is deliberately below the common rule of thumb of 4,
# because current Claude tokenizers produce up to ~1.35x more tokens than older
# ones. Output is an assumption per call (adaptive thinking is billed as
# output and cannot be predicted), so the plan also prints the ceiling: every
# reply running to max_tokens.
CHARS_PER_TOKEN = 3.5
ASSUMED_OUTPUT_TOKENS_PER_CALL = 1000


@dataclass(frozen=True)
class Estimate:
    calls: int
    input_chars: int
    input_tokens: int
    output_tokens: int
    cost: float
    ceiling: float


def estimate(model: str, prompts: list[tuple[str, str]]) -> Estimate:
    """Estimated tokens and cost of sending *prompts* (system, user) to *model*."""
    calls = len(prompts)
    input_chars = sum(len(system) + len(user) for system, user in prompts)
    input_tokens = round(input_chars / CHARS_PER_TOKEN)
    output_tokens = calls * ASSUMED_OUTPUT_TOKENS_PER_CALL
    return Estimate(
        calls=calls,
        input_chars=input_chars,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=cost_usd(model, input_tokens, output_tokens),
        ceiling=cost_usd(model, input_tokens, calls * ANTHROPIC_MAX_TOKENS),
    )


@dataclass(frozen=True)
class Plan:
    """What a run would do, as the dry run states it."""

    golden_set_path: str
    reviewed_threads: int
    person_threads: int
    max_thread_chars: int
    local_model: str
    local_url: str
    local_error: str | None  # None when the local endpoint answered its probe
    model: str
    estimate: Estimate


def format_plan(plan: Plan) -> str:
    """The dry-run text: sets, models, calls, estimated tokens and cost, and
    what leaves the house. It holds counts only, never a subject or sender."""
    est = plan.estimate
    price_in, price_out = PRICES_PER_MTOK[plan.model]
    if plan.local_error is None:
        local_state = "reachable"
    else:
        local_state = f"NOT reachable ({plan.local_error})"
    lines = [
        "Claude comparison eval: plan",
        "",
        f"Golden set:  {plan.golden_set_path}: {plan.reviewed_threads} reviewed threads, of which "
        f"{plan.person_threads} person threads are used (Stage 2, the local model's traffic).",
        f"Arm A:       local model {plan.local_model} at {plan.local_url or '<unset MLX_URL>'}: "
        f"{local_state}.",
        f"Arm B:       {plan.model} through the Anthropic API, effort {ANTHROPIC_EFFORT}, "
        f"max_tokens {ANTHROPIC_MAX_TOKENS}, no fallback model.",
        f"API calls:   {est.calls:,} message requests to Anthropic (one per thread; a rate-limit "
        "or overload reply is retried) and 1 model lookup that sends no mail.",
        f"Tokens:      about {est.input_tokens:,} input tokens ({est.input_chars:,} prompt "
        f"characters at {CHARS_PER_TOKEN} per token) and {est.output_tokens:,} output tokens "
        f"({ASSUMED_OUTPUT_TOKENS_PER_CALL:,} per reply assumed, thinking included).",
        f"Cost:        about ${est.cost:,.2f}, and at most ${est.ceiling:,.2f} if every reply "
        f"ran to max_tokens. Prices: ${price_in:.2f} / ${price_out:.2f} per million input / "
        f"output tokens for {plan.model}, as of {PRICES_AS_OF}. This is an estimate, not a quote.",
        "",
        "Privacy:     The golden sets are real mail, and arm B sends them to Anthropic: "
        f"{plan.person_threads} threads, each with its senders, subject and transcript "
        f"(up to {plan.max_thread_chars:,} characters).",
        "             Anthropic publishes its API retention: inputs are deleted within 30 days, "
        "and zero retention is available under an agreement (as issue #92 records it).",
    ]
    if plan.local_error is not None:
        lines += [
            "",
            "The local model is not reachable (is the MacBook asleep?). The run will not start, "
            "so nothing is sent to Anthropic.",
        ]
    return "\n".join(lines)


def confirmed(answer: str) -> bool:
    """True only for a typed "yes": the one approval the run asks for."""
    return answer.strip().lower() == "yes"


class CloudNotUsed:
    """Stands in the cloud slot: the comparison runs Stage 2 on person threads,
    which only the local slot serves, so a call here is a bug and fails the row."""

    model = "(cloud tier not used)"

    async def complete(self, system_prompt, user_content, include_thinking=False):
        raise RuntimeError("the comparison never calls the cloud tier")


class _PromptRecorder:
    """Local-slot stand-in that records each prompt and answers FYI."""

    model = "(prompt recorder)"

    def __init__(self):
        self.prompts: list[tuple[str, str]] = []

    async def complete(self, system_prompt, user_content, include_thinking=False):
        self.prompts.append((system_prompt, user_content))
        return ("FYI", "") if include_thinking else "FYI"


async def capture_prompts(threads: list[GoldenThread], config: dict) -> list[tuple[str, str]]:
    """The (system, user) prompt each thread sends in Stage 2, built by the
    classifier itself through the same path the arms run, without any call."""
    recorder = _PromptRecorder()
    classifier = EmailClassifier(cloud_llm=CloudNotUsed(), local_llm=recorder, config=config)
    max_chars = resolve_max_thread_chars(config)
    for thread in threads:
        await evaluate_single(thread, classifier, STAGES, max_chars)
    return recorder.prompts


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
