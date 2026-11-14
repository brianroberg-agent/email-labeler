"""Compare the local model with a Claude model on the email golden set (issue #92).

Arm A is the local leg as ``run_eval --local-only`` runs it: Stage 2 on the
golden set's person threads, through the ``[llm.local]`` endpoint. Arm B runs the
same threads through the same ``EmailClassifier`` with a Claude model in the
local slot, so both arms see the identical prompt. Usage: evals/README.md;
flags, the order of a run and the outputs: evals/README-technical.md.
"""

import argparse
import asyncio
import getpass
import hashlib
import json
import math
import os
import stat
import statistics
import tomllib
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from classifier import EmailClassifier
from config_utils import substitute_env_vars
from daemon import resolve_int_env
from evals import report
from evals.run_eval import (
    build_output_path,
    evaluate_single,
    load_golden_set,
    preflight_check,
    resolve_max_thread_chars,
    run_evaluation,
    sanitize_tag,
    write_results,
)
from evals.schemas import GoldenThread, PredictionResult, RunMeta
from llm_client import LLMClient, LLMContentError

# The population both arms see: Stage 2 on reviewed person threads, i.e.
# `run_eval --local-only`. The local model serves person bodies, so this is
# the leg issue #92 would move.
STAGES = "stage2_only"
SENDER_TYPE = "person"

# Request shape for arm B. Effort is sent explicitly because the default differs
# by model; max_tokens covers adaptive thinking (billed as output) plus the
# step-by-step answer the prompt asks for.
ANTHROPIC_MAX_TOKENS = 16000
ANTHROPIC_EFFORT = "medium"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
# The SDK retries 408/409/429/5xx (529 "overloaded" included) and connection
# errors, timeouts among them, with exponential backoff, honouring retry-after.
# Each attempt is bounded by the timeout: 600 s, the SDK's own read timeout for a
# non-streaming request, which covers its expected time for a reply that runs to
# max_tokens (3600 s x 16000 / 128000 = 450 s). A shorter bound would cut slow
# replies and retry them with no usage counted for the cut attempts.
ANTHROPIC_MAX_RETRIES = 4
ANTHROPIC_TIMEOUT_SECONDS = 600.0


class ApiKeyError(Exception):
    """The Anthropic key could not be taken from the source Brian named."""


def obtain_key(key_file: Path | None, *, getpass_fn) -> str:
    """The Anthropic API key, from *key_file* or else typed at a hidden prompt.

    These are the only two sources: the key is not read from argv or from the
    environment (daemon.py loads the agent-stack .env, and issue #92 keeps the
    key out of that stack). A key file must be private to its owner (no group
    or other permission bits), so a root-only file deleted after the run is
    the file route the README describes.

    Without a terminal, getpass warns (GetPassWarning) and then reads the key
    with echo on. The warning is raised as an error here, so it stops getpass
    before the read and the run asks for the key file instead.
    """
    if key_file is None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            try:
                key = getpass_fn("Anthropic API key (input hidden): ").strip()
            except getpass.GetPassWarning:
                raise ApiKeyError(
                    "there is no terminal to hide the typed key, so it was not read; put the key "
                    "in an owner-only file and pass --anthropic-key-file PATH"
                ) from None
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
# output and is not known in advance), so the plan also prints the ceiling: every
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
    what leaves the house. It holds counts, no subject or sender."""
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


@dataclass
class ArmRun:
    """One arm's results and, for the API arm, the usage it was billed for."""

    name: str
    results: list[PredictionResult]
    parallelism: int
    priced_model: str | None = None  # None: a local model, no API charge
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def cost(self) -> float:
        if self.priced_model is None:
            return 0.0
        return cost_usd(self.priced_model, self.input_tokens, self.output_tokens)


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile: the smallest value with at least q% of values at or below it."""
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


# The recorded error-cost order, worst first: docs/labeling-rubric.md section 3
# (decision D21). Each class is a set of (expected, predicted) label pairs; on
# person threads an expected low_priority is a cold pitch (rubric rule P1).
COST_ORDER = [
    ("1. Person mail archived (needs_response/fyi → low_priority)",
     [("needs_response", "low_priority"), ("fyi", "low_priority")]),
    ("2. needs_response read as FYI", [("needs_response", "fyi")]),
    ("3. FYI read as needs_response", [("fyi", "needs_response")]),
    ("4. Cold pitch kept (low_priority → needs_response/fyi)",
     [("low_priority", "needs_response"), ("low_priority", "fyi")]),
]


def _paired(arm_a: ArmRun, arm_b: ArmRun) -> tuple[list[PredictionResult], list[PredictionResult]]:
    """Each arm's rows for the threads that both arms answered without error."""
    ok_b = {r.thread_id for r in arm_b.results if r.error is None}
    both = {r.thread_id for r in arm_a.results if r.error is None} & ok_b
    return ([r for r in arm_a.results if r.thread_id in both],
            [r for r in arm_b.results if r.thread_id in both])


def _ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return f"N/A ({numerator}/{denominator})"
    return f"{numerator / denominator:.1%} ({numerator}/{denominator})"


def _arm_cells(rows: list[PredictionResult], arm: ArmRun) -> dict[str, str]:
    matrix = report.compute_confusion_matrix(rows, "expected_label", "predicted_label", report.LABEL_CLASSES)
    cells = {}
    for cls in report.LABEL_CLASSES:
        tp = matrix[cls][cls]
        cells[f"{cls} recall"] = _ratio(tp, sum(matrix[cls].values()))
        cells[f"{cls} precision"] = _ratio(tp, sum(matrix[other][cls] for other in report.LABEL_CLASSES))
    for label, pairs in COST_ORDER:
        cells[label] = str(sum(matrix[expected][predicted] for expected, predicted in pairs))
    errors = sum(1 for r in arm.results if r.error is not None)
    cells["Errors (no usable answer)"] = f"{errors}/{len(arm.results)}"
    seconds = [r.duration_seconds for r in rows]
    cells["Latency per message, median"] = f"{statistics.median(seconds):.1f} s" if seconds else "N/A"
    cells["Latency per message, p90"] = f"{percentile(seconds, 90):.1f} s" if seconds else "N/A"
    if arm.priced_model is None:
        cells["Cost per message"] = cells["Cost, total"] = "$0 (local)"
    else:
        price_in, price_out = PRICES_PER_MTOK[arm.priced_model]
        sent = len(arm.results)
        cells["Cost per message"] = f"${arm.cost / sent:.4f}" if sent else "N/A"
        cells["Cost, total"] = (
            f"${arm.cost:.2f} ({arm.input_tokens:,} input / {arm.output_tokens:,} output tokens "
            f"at ${price_in:.2f} / ${price_out:.2f} per million, prices of {PRICES_AS_OF})"
        )
    return cells


# Row order of the report table: needs_response recall first (the rubric's
# second-worst error is its complement), then the cost-ordered error classes,
# the remaining per-category figures, errors, latency and cost.
_ROW_ORDER = (
    ["needs_response recall"]
    + [label for label, _ in COST_ORDER]
    + ["needs_response precision", "fyi recall", "fyi precision",
       "low_priority recall", "low_priority precision",
       "Errors (no usable answer)",
       "Latency per message, median", "Latency per message, p90",
       "Cost per message", "Cost, total"]
)


def build_report(
    arm_a: ArmRun, arm_b: ArmRun, *, golden_set_path: str, reviewed_threads: int, timestamp: str,
) -> str:
    """The side-by-side Markdown report. It holds counts and figures only: no
    subject, sender, body or thread id."""
    rows_a, rows_b = _paired(arm_a, arm_b)
    cells_a, cells_b = _arm_cells(rows_a, arm_a), _arm_cells(rows_b, arm_b)
    lines = [
        f"# Claude comparison: {arm_a.name} vs {arm_b.name}",
        "",
        f"- Run: {timestamp}",
        f"- Golden set: `{golden_set_path}`, {reviewed_threads} reviewed threads; both arms ran "
        f"Stage 2 on its {len(arm_a.results)} person threads (the `run_eval --local-only` population).",
        f"- Scored on the {len(rows_a)} threads both arms answered; each arm's errors are counted "
        "in their own row.",
        "- Error classes follow the cost order of `docs/labeling-rubric.md` section 3 (decision D21), "
        "worst first.",
        "",
        f"| Metric | {arm_a.name} | {arm_b.name} |",
        "|---|---|---|",
    ]
    lines += [f"| {row} | {cells_a[row]} | {cells_b[row]} |" for row in _ROW_ORDER]
    lines += [
        "",
        f"- Concurrency: arm A {arm_a.parallelism} at a time (`LOCAL_PARALLEL` or `local_parallel`), arm B "
        f"{arm_b.parallelism} at a time. Latency is wall time per message, retries included.",
        f"- Arm B: effort {ANTHROPIC_EFFORT}, max_tokens {ANTHROPIC_MAX_TOKENS}, no temperature, no "
        "fallback model; a refusal or a reply cut at max_tokens counts as an error.",
        "- Neither arm used the eval response cache, so latency and cost are measured, not replayed.",
        "",
    ]
    return "\n".join(lines)


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


# Arm B's concurrency: modest, so a normal account's rate limit is not the
# bottleneck; 429s that do occur are retried by the SDK (see above).
ANTHROPIC_PARALLELISM = 4
DEFAULT_CONFIG = Path(__file__).parent.parent / "config.toml"


def load_config(path: str | None) -> tuple[dict, Path]:
    config_path = Path(path) if path else DEFAULT_CONFIG
    with open(config_path, "rb") as f:
        return substitute_env_vars(tomllib.load(f)), config_path


def build_local_llm(config: dict) -> LLMClient:
    """Arm A's client, built as run_eval builds the local tier (MLX_URL /
    MLX_API_KEY / [llm.local]). Where MLX_URL points decides where arm A's
    mail goes: a public stand-in sends it off-network (decision D4)."""
    local = config["llm"]["local"]
    return LLMClient(
        base_url=os.environ.get("MLX_URL", ""),
        api_key=os.environ.get("MLX_API_KEY", ""),
        model=local["model"],
        max_tokens=local["max_tokens"],
        temperature=local["temperature"],
        timeout=local["timeout"],
        extra_body=local.get("extra_body"),
    )


def _write_arm_results(
    output_dir: Path, tag: str, results: list[PredictionResult], *, config: dict,
    config_path: Path, golden_path: Path, classifier: EmailClassifier, model: str,
    parallelism: int, temperature: float | None, max_tokens: int, extra_body: dict | None,
) -> Path:
    """One arm's rows as an ordinary run_eval results file (git-ignored), so
    `evals.report --compare` can re-read the two arms without re-running them."""
    run_id = uuid.uuid4().hex
    categories = list(config["prompts"]["email_classification"]["categories"])
    meta = RunMeta(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        config_hash=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16],
        config_path=str(config_path),
        cloud_model=CloudNotUsed.model,
        local_model=model,
        golden_set_path=str(golden_path),
        golden_set_count=len(results),
        stages=STAGES,
        parallelism=parallelism,
        tag=tag,
        local_temperature=temperature,
        local_max_tokens=max_tokens,
        local_extra_body=extra_body,
        email_system_prompt=classifier._build_email_prompt(categories),
    )
    path = build_output_path(output_dir, STAGES, tag, run_id)
    write_results(path, meta, results)
    return path


async def run_comparison(
    args: argparse.Namespace, *, local_llm=None, client_factory=build_anthropic_client,
    getpass_fn=getpass.getpass, input_fn=input,
) -> int:
    """Plan, one typed yes, then both arms and the report. Returns the exit code.

    Order matters: nothing is sent to Anthropic before the plan has been shown,
    the local model has answered its probe, "yes" has been typed and the key
    and model have passed a lookup that carries no mail; arm B runs only after
    arm A has produced results.
    """
    config, config_path = load_config(args.config)
    golden_path = Path(args.golden_set)
    reviewed = load_golden_set(golden_path, reviewed_only=True)
    threads = [t for t in reviewed if t.expected_sender_type == SENDER_TYPE]
    if args.max_threads:
        threads = threads[:args.max_threads]
    if not threads:
        print(f"No reviewed person threads in {golden_path}; nothing to compare.")
        return 1

    if local_llm is None:
        local_llm = build_local_llm(config)
    local_cfg = config["llm"]["local"]
    unreachable = await preflight_check(
        None, local_llm, need_cloud=False, need_local=True, local_timeout=local_cfg["timeout"],
    )
    max_chars = resolve_max_thread_chars(config)
    plan = Plan(
        golden_set_path=str(golden_path),
        reviewed_threads=len(reviewed),
        person_threads=len(threads),
        max_thread_chars=max_chars,
        local_model=local_llm.model,
        local_url=local_llm.base_url,
        local_error="; ".join(unreachable) if unreachable else None,
        model=args.model,
        estimate=estimate(args.model, await capture_prompts(threads, config)),
    )
    print(format_plan(plan))
    if unreachable:
        return 1
    if args.dry_run:
        print("\nDry run: nothing was sent anywhere.")
        return 0

    answer = input_fn(f"\nType yes to send these {len(threads)} threads to Anthropic and run both arms: ")
    if not confirmed(answer):
        print("Stopped. Nothing was sent to Anthropic.")
        return 1
    try:
        client = client_factory(obtain_key(args.anthropic_key_file, getpass_fn=getpass_fn))
    except ApiKeyError as exc:
        print(f"Error: {exc}. Nothing was sent to Anthropic.")
        return 1
    try:
        await client.models.retrieve(args.model)
    except anthropic.APIError as exc:
        status = getattr(exc, "status_code", None)
        print(f"Error: Anthropic did not accept the key or the model {args.model} "
              f"({type(exc).__name__}{f', HTTP {status}' if status else ''}). No mail was sent.")
        return 1

    output_dir = Path(args.output_dir)
    # The daemon's local concurrency, read as daemon.py reads it.
    local_parallel = resolve_int_env("LOCAL_PARALLEL", config.get("daemon", {}).get("local_parallel", 1))
    classifier_a = EmailClassifier(cloud_llm=CloudNotUsed(), local_llm=local_llm, config=config)
    print(f"\nArm A: {local_llm.model}, {len(threads)} threads, {local_parallel} at a time...")
    results_a, _ = await run_evaluation(threads, classifier_a, STAGES, max_chars, parallelism=local_parallel)
    path_a = _write_arm_results(
        output_dir, f"compare-A-{sanitize_tag(local_llm.model)}", results_a, config=config,
        config_path=config_path, golden_path=golden_path, classifier=classifier_a,
        model=local_llm.model, parallelism=local_parallel, temperature=local_cfg["temperature"],
        max_tokens=local_cfg["max_tokens"], extra_body=local_cfg.get("extra_body"),
    )
    if all(r.error is not None for r in results_a):
        print(f"Arm A produced no results (see {path_a}), so arm B was not run and no mail "
              "was sent to Anthropic.")
        return 1

    claude = AnthropicLLMClient(client, args.model)
    classifier_b = EmailClassifier(cloud_llm=CloudNotUsed(), local_llm=claude, config=config)
    print(f"\nArm B: {args.model}, {len(threads)} threads, {ANTHROPIC_PARALLELISM} at a time...")
    results_b, _ = await run_evaluation(
        threads, classifier_b, STAGES, max_chars, parallelism=ANTHROPIC_PARALLELISM,
    )
    path_b = _write_arm_results(
        output_dir, f"compare-B-{args.model}", results_b, config=config,
        config_path=config_path, golden_path=golden_path, classifier=classifier_b,
        model=args.model, parallelism=ANTHROPIC_PARALLELISM, temperature=None,
        max_tokens=ANTHROPIC_MAX_TOKENS, extra_body={"output_config": {"effort": ANTHROPIC_EFFORT}},
    )

    now = datetime.now(timezone.utc)
    text = build_report(
        ArmRun(name=f"Arm A: local {local_llm.model}", results=results_a, parallelism=local_parallel),
        ArmRun(name=f"Arm B: {args.model}", results=results_b, parallelism=ANTHROPIC_PARALLELISM,
               priced_model=args.model, input_tokens=claude.input_tokens,
               output_tokens=claude.output_tokens),
        golden_set_path=str(golden_path), reviewed_threads=len(reviewed), timestamp=now.isoformat(),
    )
    report_path = output_dir / (
        f"{now.strftime('%Y%m%d_%H%M%S')}_compare_{sanitize_tag(local_llm.model)}_vs_{args.model}.md"
    )
    report_path.write_text(text)
    print("\n" + text)
    print(f"Report: {report_path}\nPer-arm results: {path_a}, {path_b}")
    return 0


DEFAULT_MODEL = "claude-opus-5-5"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the local model with a Claude model on the golden set (issue #92). "
                    "Prints the plan, asks for a typed yes, then runs both arms and writes a report.",
    )
    parser.add_argument("--golden-set", default="evals/golden_set.jsonl", help="Path to golden set JSONL")
    parser.add_argument("--config", help="Path to config.toml (default: the repo-root config.toml)")
    parser.add_argument("--output-dir", default="evals/results/",
                        help="Directory for the report and the two per-arm results files")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(PRICES_PER_MTOK),
                        help=f"Claude model for arm B (default: {DEFAULT_MODEL})")
    parser.add_argument("--anthropic-key-file", type=Path, metavar="PATH",
                        help="Read the Anthropic API key from this owner-only file "
                             "(default: prompt for it, input hidden)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan (sets, models, calls, estimated tokens and cost) and stop")
    parser.add_argument("--max-threads", type=int, help="Use at most this many person threads")
    return parser


def cli():
    args = build_parser().parse_args()
    raise SystemExit(asyncio.run(run_comparison(args)))


if __name__ == "__main__":
    cli()
