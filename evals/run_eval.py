"""Replay golden set through the real EmailClassifier with real LLM endpoints.

Usage:
    python -m evals.run_eval
    python -m evals.run_eval --golden-set evals/golden_set.jsonl --stages full
    python -m evals.run_eval --config config_v2.toml --tag new-prompts
    python -m evals.run_eval --stages stage1_only --include-unreviewed
    python -m evals.run_eval --dry-run
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
import tomllib
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

from classifier import EmailClassifier, SenderType, ThreadMetadata
from config_utils import substitute_env_vars
from daemon import DEFAULT_MAX_THREAD_CHARS, format_thread_transcript
from evals import format_network_error, report
from evals.llm_cache import CachedLLMClient
from evals.schemas import GoldenThread, PredictionResult, RunMeta, ThinkingEntry
from gmail_utils import get_header
from llm_client import LLMClient

VALID_STAGES = ("full", "stage1_only", "stage2_only")


def resolve_local_only(args: argparse.Namespace) -> None:
    """Expand --local-only into --stages stage2_only --sender-type person, in place.

    Stage 2b on a person body is the *only* path that exercises solely the local
    classifier (person bodies never reach the cloud), so this is the canonical
    "evaluate just the local model" configuration. A no-op unless --local-only is
    set. The default --stages ("full") is treated as unspecified and overridden.

    Raises:
        ValueError: if the user also passed an incompatible --stages or
            --sender-type alongside --local-only.
    """
    if not getattr(args, "local_only", False):
        return
    if args.stages not in ("full", "stage2_only"):
        raise ValueError(
            f"--local-only implies --stages stage2_only, which conflicts with "
            f"--stages {args.stages}"
        )
    if args.sender_type not in (None, "person"):
        raise ValueError(
            f"--local-only implies --sender-type person, which conflicts with "
            f"--sender-type {args.sender_type}"
        )
    args.stages = "stage2_only"
    args.sender_type = "person"


def sanitize_tag(name: str) -> str:
    """Make a model name safe for a results filename.

    Filenames join parts with "_", so any run of characters outside
    [A-Za-z0-9._-] (notably the "/" in HuggingFace ids like "qwen/qwen3-14b")
    is collapsed to a single "-", with leading/trailing dashes stripped.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")


def default_tag(stages: str, config: dict) -> str:
    """Derive a default results tag from the model actually under test.

    Stage 1 is the cloud model's job; everything else exercises the local model,
    so tag by whichever model the run is really measuring. Makes result files
    self-identifying without the user having to pass --tag.
    """
    section = "cloud" if stages == "stage1_only" else "local"
    return sanitize_tag(config["llm"][section]["model"])


def required_endpoints(stages: str, sender_type: str | None) -> tuple[bool, bool]:
    """Return (need_cloud, need_local): the endpoints WITHOUT which the run
    produces no results at all, so preflight only hard-aborts a doomed run.

    Stage 1 (sender) always uses the cloud; Stage 2 routes person bodies to the
    local model and service bodies to the cloud. An endpoint is "wholly required"
    only when every selected thread depends on it:
      - stage1_only -> cloud (every thread classifies a sender).
      - stage2_only person -> local; stage2_only service -> cloud.
      - stage2_only with no filter -> neither (a down endpoint loses only the
        person *or* service half, leaving partial results).
      - full -> cloud only: Stage 1 gates every thread, but a down local endpoint
        merely fails person Stage 2b, leaving Stage 1 + service Stage 2a results.
    """
    if stages == "stage2_only":
        if sender_type == "person":
            return False, True
        if sender_type == "service":
            return True, False
        return False, False
    return True, False  # stage1_only and full both wholly require only the cloud


async def preflight_check(
    cloud_base, local_base, need_cloud: bool, need_local: bool,
    cloud_timeout: float | None = None, local_timeout: float | None = None,
) -> list[str]:
    """Probe each required endpoint once; return a human-readable error per dead one.

    Turns the most common silent failure — a local MLX_MODEL that doesn't match
    the served --model (every request 404s) or an unreachable endpoint — into an
    immediate, clear error instead of a whole run of per-thread errors. Endpoints
    the run won't wholly depend on are never probed.

    Each endpoint gets its own probe timeout so the cloud probe need not wait out
    the long local cold-load budget; pass a generous *local_timeout* for a server
    that loads the requested model on demand, so a cold load is not read as "down".
    """
    errors = []
    if need_cloud:
        result = await cloud_base.probe(cloud_timeout)
        if not result.ok:
            errors.append(
                f"cloud LLM '{cloud_base.model}' not reachable at "
                f"{cloud_base.base_url or '<unset CLOUD_LLM_URL>'}"
                + _detail_suffix(result)
            )
    if need_local:
        result = await local_base.probe(local_timeout)
        if not result.ok:
            errors.append(
                f"local LLM '{local_base.model}' not reachable at "
                f"{local_base.base_url or '<unset MLX_URL>'}"
                + _detail_suffix(result)
            )
    return errors


def _detail_suffix(result) -> str:
    """Parenthesized probe-failure detail for a preflight message, or "" if none.

    Finding 1: surfaces probe()'s captured diagnosis (HTTP status / exception
    text / 404 model-mismatch hint) instead of discarding it. All wording lives on
    AvailabilityResult.detail() so this and newsletter_run's preflight can't drift
    (Finding 2)."""
    detail = result.detail()
    return f" ({detail})" if detail else ""


def resolve_parallelism(cli_value: int | None, config: dict) -> int:
    """Resolve parallelism: CLI flag wins, otherwise fall back to config."""
    if cli_value is not None:
        return cli_value
    return config.get("daemon", {}).get("cloud_parallel", 1)


def resolve_max_thread_chars(config: dict) -> int:
    """Transcript char cap: config value if present, else the shared daemon default.

    Uses DEFAULT_MAX_THREAD_CHARS (imported from daemon) so an eval run truncates
    transcripts exactly as production does, even for a config that omits the key.
    """
    return config.get("daemon", {}).get("max_thread_chars", DEFAULT_MAX_THREAD_CHARS)


def resolve_extra_body(base: dict | None, no_think: bool, override_json: str | None) -> dict | None:
    """Merge CLI extra_body overrides onto one LLM's config extra_body.

    Precedence (later wins):
      1. config's existing extra_body (*base*)
      2. --no-think -> sets BOTH disable dialects: chat_template_kwargs.
         enable_thinking = False (mlx_lm.server / LM Studio) and top-level
         reasoning_effort = "none" (Ollama — the only field its OpenAI-compat
         endpoint honors; issue #64). A backend ignores the dialect it doesn't
         know, so emitting both keeps the flag meaningful across backends.
      3. --extra-body JSON object (its keys override the above)

    Returns the merged dict, or None when nothing is set (so the LLMClient keeps
    its default). The eval cache key includes extra_body, so thinking-on vs
    thinking-off runs never collide in the cache. Does not mutate *base*.

    Raises:
        ValueError: if override_json is not a valid JSON object.
    """
    result = dict(base) if base else {}
    if no_think:
        ctk = dict(result.get("chat_template_kwargs") or {})
        ctk["enable_thinking"] = False
        result["chat_template_kwargs"] = ctk
        result["reasoning_effort"] = "none"
    if override_json:
        try:
            override = json.loads(override_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid --extra-body JSON: {exc}") from exc
        if not isinstance(override, dict):
            raise ValueError("--extra-body must be a JSON object")
        result.update(override)
    return result or None


def apply_config_overrides(config: dict, args: argparse.Namespace) -> None:
    """Apply all CLI overrides onto *config* in place (CLI always wins over config).

    Covers the model/temperature/max_tokens/timeout scalars and extra_body
    (thinking toggles / provider-specific params) in one pass, so all CLI->config
    plumbing lives in a single tested function.

    A scalar CLI value of None means "not provided" and leaves the config value
    intact; 0 / 0.0 are valid overrides (e.g. temperature 0), so the guard is
    `is not None`, not truthiness. extra_body is merged via resolve_extra_body
    (the eval cache key includes extra_body, so thinking-on vs -off runs don't
    collide).

    Raises:
        ValueError: if a --*-extra-body value is not a valid JSON object.
    """
    scalars = {
        "cloud": {"model": args.cloud_model, "temperature": args.cloud_temperature,
                  "max_tokens": args.cloud_max_tokens, "timeout": args.cloud_timeout},
        "local": {"model": args.local_model, "temperature": args.local_temperature,
                  "max_tokens": args.local_max_tokens, "timeout": args.local_timeout},
    }
    for section, fields in scalars.items():
        for key, value in fields.items():
            if value is not None:
                config["llm"][section][key] = value

    extra_body = {
        "cloud": (args.cloud_no_think, args.cloud_extra_body),
        "local": (args.local_no_think, args.local_extra_body),
    }
    for section, (no_think, eb_json) in extra_body.items():
        resolved = resolve_extra_body(
            config["llm"][section].get("extra_body"), no_think, eb_json,
        )
        if resolved is not None:
            config["llm"][section]["extra_body"] = resolved


def load_golden_set(path: Path, reviewed_only: bool = False) -> list[GoldenThread]:
    """Load golden set from JSONL."""
    threads = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                gt = GoldenThread.from_dict(json.loads(line))
                if gt.excluded:
                    continue
                if reviewed_only and not gt.reviewed:
                    continue
                threads.append(gt)
    return threads


def reconstruct_thread_metadata(golden: GoldenThread) -> ThreadMetadata:
    """Reconstruct ThreadMetadata from a golden set entry.

    Mirrors daemon.py:137-160 — sort by internalDate, extract unique senders,
    first-message subject, last-message snippet.
    """
    messages = sorted(golden.messages, key=lambda m: int(m.get("internalDate", "0")))

    senders = []
    seen: set[str] = set()
    for msg in messages:
        sender = get_header(msg["payload"]["headers"], "From")
        if sender and sender not in seen:
            senders.append(sender)
            seen.add(sender)

    if not senders:
        senders = golden.senders  # Fallback to harvested senders

    first_headers = messages[0]["payload"]["headers"] if messages else []
    subject = get_header(first_headers, "Subject") if first_headers else golden.subject
    snippet = messages[-1].get("snippet", "") if messages else golden.snippet

    return ThreadMetadata(
        thread_id=golden.thread_id,
        senders=senders,
        subject=subject,
        snippet=snippet,
    )


def reconstruct_transcript(golden: GoldenThread, max_chars: int) -> str:
    """Reconstruct thread transcript using daemon.format_thread_transcript()."""
    messages = sorted(golden.messages, key=lambda m: int(m.get("internalDate", "0")))
    return format_thread_transcript(messages, max_chars)


async def evaluate_single(
    golden: GoldenThread,
    classifier: EmailClassifier,
    stages: str,
    max_thread_chars: int,
) -> tuple[PredictionResult, ThinkingEntry]:
    """Evaluate a single golden thread.

    Args:
        golden: Golden set entry with ground truth.
        classifier: Real EmailClassifier instance.
        stages: "full", "stage1_only", or "stage2_only".
        max_thread_chars: Max chars for thread transcript.

    Returns:
        Tuple of (PredictionResult, ThinkingEntry).
    """
    result = PredictionResult(
        thread_id=golden.thread_id,
        expected_sender_type=golden.expected_sender_type,
        expected_label=golden.expected_label,
        # Carried through so the report can score the assistant field without
        # re-reading the golden set. Nothing predicts it yet (issue #78
        # groundwork), so predicted_assistant/assistant_correct stay None.
        expected_assistant=golden.expected_assistant,
    )
    thinking = ThinkingEntry(thread_id=golden.thread_id)

    metadata = reconstruct_thread_metadata(golden)
    transcript = reconstruct_transcript(golden, max_thread_chars)

    cloud = classifier.cloud_llm
    local = classifier.local_llm

    # Drain any leftover thinking from previous calls
    if hasattr(cloud, "take_thinking"):
        cloud.take_thinking()
        local.take_thinking()

    start = time.monotonic()
    try:
        if stages == "stage1_only":
            sender_type, sender_raw, sender_cot = await classifier.classify_sender(metadata)
            result.predicted_sender_type = sender_type.value
            result.predicted_sender_type_raw = sender_raw
            result.sender_type_correct = result.predicted_sender_type == golden.expected_sender_type
            result.privacy_violation = (
                golden.expected_sender_type == "person" and result.predicted_sender_type == "service"
            )
            thinking.stage1_thinking = sender_cot

        elif stages == "stage2_only":
            # Use expected sender type as input (skip Stage 1)
            sender_type = SenderType(golden.expected_sender_type)
            label, label_raw, label_cot = await classifier.classify_email(metadata, transcript, sender_type)
            result.predicted_sender_type = golden.expected_sender_type  # Passed through
            result.predicted_label = label.value
            result.predicted_label_raw = label_raw
            result.sender_type_correct = True  # By definition
            result.label_correct = result.predicted_label == golden.expected_label
            thinking.stage2_thinking = label_cot

        else:  # full
            sender_type, sender_raw, sender_cot = await classifier.classify_sender(metadata)
            thinking.stage1_thinking = sender_cot

            # Stage 2
            vip = classifier._is_vip(metadata)
            label, label_raw, label_cot = await classifier.classify_email(
                metadata, transcript, sender_type, vip=vip,
            )
            thinking.stage2_thinking = label_cot

            result.predicted_sender_type = sender_type.value
            result.predicted_sender_type_raw = sender_raw
            result.predicted_label = label.value
            result.predicted_label_raw = label_raw
            result.sender_type_correct = result.predicted_sender_type == golden.expected_sender_type
            result.label_correct = result.predicted_label == golden.expected_label
            result.privacy_violation = (
                golden.expected_sender_type == "person" and result.predicted_sender_type == "service"
            )

    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        result.error = format_network_error(exc, "LLM endpoint")
    except Exception as exc:
        result.error = str(exc)

    # Use actual LLM time when caching (0 for pure cache hits); wall time otherwise
    if hasattr(cloud, "take_llm_seconds"):
        result.duration_seconds = round(
            cloud.take_llm_seconds() + local.take_llm_seconds(), 3
        )
    else:
        result.duration_seconds = round(time.monotonic() - start, 3)
    return result, thinking


async def run_evaluation(
    golden_set: list[GoldenThread],
    classifier: EmailClassifier,
    stages: str,
    max_thread_chars: int,
    parallelism: int = 1,
) -> tuple[list[PredictionResult], list[ThinkingEntry]]:
    """Run evaluation across all golden threads.

    Args:
        golden_set: List of golden set entries.
        classifier: Real EmailClassifier instance.
        stages: Which stages to run.
        max_thread_chars: Max chars for transcript.
        parallelism: Number of concurrent evaluations.

    Returns:
        Tuple of (predictions, thinking_entries).
    """
    semaphore = asyncio.Semaphore(parallelism)
    total = len(golden_set)

    async def eval_with_semaphore(
        golden: GoldenThread, idx: int,
    ) -> tuple[PredictionResult, ThinkingEntry]:
        async with semaphore:
            r, t = await evaluate_single(golden, classifier, stages, max_thread_chars)
            status = "error" if r.error else "ok"
            print(f"  [{idx + 1}/{total}] {golden.thread_id[:12]}... {status}", file=sys.stderr)
            return r, t

    tasks = [eval_with_semaphore(gt, i) for i, gt in enumerate(golden_set)]
    pairs = await asyncio.gather(*tasks)
    results = [p[0] for p in pairs]
    thinking = [p[1] for p in pairs]
    return results, thinking


def build_output_path(output_dir: Path, stages: str, tag: str, run_id: str) -> Path:
    """Build output file path from run parameters."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parts = [timestamp, stages]
    if tag:
        parts.append(tag)
    parts.append(run_id[:8])
    filename = "_".join(parts) + ".jsonl"
    return output_dir / filename


def write_results(
    path: Path,
    meta: RunMeta,
    results: list[PredictionResult],
) -> None:
    """Write run metadata and results to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps(meta.to_dict()) + "\n")
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")


def write_thinking_sidecar(
    results_path: Path,
    thinking_entries: list[ThinkingEntry],
) -> None:
    """Write chain-of-thought content to sidecar .cot.jsonl file.

    Only writes entries that have at least one non-empty thinking field.
    """
    entries = [t for t in thinking_entries if t.stage1_thinking or t.stage2_thinking]
    if not entries:
        return
    sidecar_path = results_path.with_suffix(".cot.jsonl")
    with open(sidecar_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry.to_dict()) + "\n")


def maybe_report(
    meta: RunMeta,
    results: list[PredictionResult],
    report_enabled: bool,
    compare_to: str | None,
) -> None:
    """Print a metrics report (and optional comparison) for a just-finished run.

    Saves the find-the-file-then-invoke-report round-trip: run_eval already knows
    the run it produced. Reuses report.py so all formatting lives in one place.
    A no-op unless --report or --compare-to was passed.
    """
    if not report_enabled and not compare_to:
        return
    metrics = report.compute_metrics(results)
    if report_enabled:
        report.print_report(meta, metrics)
    if compare_to:
        compare_path = Path(compare_to)
        if not compare_path.exists():
            print(f"Error: --compare-to file not found: {compare_path}", file=sys.stderr)
            return
        try:
            meta_b, results_b = report.load_results(compare_path)
        except (ValueError, OSError) as exc:
            print(f"Error: could not read --compare-to results from {compare_path}: {exc}",
                  file=sys.stderr)
            return
        if meta_b.stages != meta.stages:
            print(
                f"Warning: baseline stages={meta_b.stages} differ from this run's "
                f"stages={meta.stages}; deltas may span different thread populations.",
                file=sys.stderr,
            )
        # Baseline is Run A and the just-finished run is Run B, so print_comparison's
        # Delta column reads (this run - baseline) — positive means improvement.
        # verbose=True surfaces the per-thread regression/improvement list.
        report.print_comparison(
            meta_b, report.compute_metrics(results_b), meta, metrics,
            verbose=True, results1=results_b, results2=results,
        )


async def main(args: argparse.Namespace) -> None:
    # Load config
    config_path = Path(args.config) if args.config else Path(__file__).parent.parent / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    config = substitute_env_vars(config)

    # Resolve parallelism: CLI wins, otherwise fall back to config
    args.parallelism = resolve_parallelism(args.parallelism, config)

    # Apply CLI overrides to config (CLI always wins over config file). extra_body
    # parsing and --local-only conflict detection can fail, so surface those as a
    # clean exit.
    try:
        resolve_local_only(args)
        apply_config_overrides(config, args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    # Default the results tag to the model under test, so result files (and the
    # report trend table) self-identify without the user having to pass --tag.
    if args.tag is None:
        args.tag = default_tag(args.stages, config)

    # Load golden set
    golden_path = Path(args.golden_set)
    golden_set = load_golden_set(golden_path, reviewed_only=not args.include_unreviewed)
    if args.sender_type:
        golden_set = [g for g in golden_set if g.expected_sender_type == args.sender_type]
    if args.max_threads:
        golden_set = golden_set[:args.max_threads]

    if not golden_set:
        print("No threads to evaluate.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(golden_set)} threads from {golden_path}", file=sys.stderr)

    if args.dry_run:
        print("Dry run — would evaluate:", file=sys.stderr)
        for gt in golden_set:
            print(f"  {gt.thread_id}: {gt.subject} ({gt.expected_sender_type}/{gt.expected_label})",
                  file=sys.stderr)
        return

    # Build LLM clients (same as daemon.run_daemon())
    cloud_llm_base = LLMClient(
        base_url=os.environ.get("CLOUD_LLM_URL", ""),
        api_key=os.environ.get("CLOUD_LLM_API_KEY", ""),
        model=config["llm"]["cloud"]["model"],
        max_tokens=config["llm"]["cloud"]["max_tokens"],
        temperature=config["llm"]["cloud"]["temperature"],
        timeout=config["llm"]["cloud"]["timeout"],
        extra_body=config["llm"]["cloud"].get("extra_body"),
    )
    local_llm_base = LLMClient(
        base_url=os.environ.get("MLX_URL", ""),
        api_key=os.environ.get("MLX_API_KEY", ""),
        model=config["llm"]["local"]["model"],
        max_tokens=config["llm"]["local"]["max_tokens"],
        temperature=config["llm"]["local"]["temperature"],
        timeout=config["llm"]["local"]["timeout"],
        extra_body=config["llm"]["local"].get("extra_body"),
    )

    # Preflight: probe the endpoints this run will actually hit and fail fast on
    # an unreachable / name-mismatched one, instead of a whole run of per-thread
    # errors. (Skipped on --dry-run, which returned above.)
    if not args.skip_preflight:
        need_cloud, need_local = required_endpoints(args.stages, args.sender_type)
        # Each probe defaults to its own endpoint's request timeout (so the local
        # probe is generous enough for a load-on-demand cold start, while the cloud
        # probe still fails fast); an explicit --preflight-timeout overrides both.
        cloud_timeout = (
            args.preflight_timeout if args.preflight_timeout is not None
            else config["llm"]["cloud"]["timeout"]
        )
        local_timeout = (
            args.preflight_timeout if args.preflight_timeout is not None
            else config["llm"]["local"]["timeout"]
        )
        unreachable = await preflight_check(
            cloud_llm_base, local_llm_base, need_cloud, need_local,
            cloud_timeout=cloud_timeout, local_timeout=local_timeout,
        )
        if unreachable:
            for msg in unreachable:
                print(f"Error: {msg}", file=sys.stderr)
            sys.exit(1)

    # Wrap with cache unless --no-cache
    if args.no_cache:
        cloud_llm = cloud_llm_base
        local_llm = local_llm_base
    else:
        cache_path = Path(__file__).parent / "cache" / "llm_cache.jsonl"
        cloud_llm = CachedLLMClient(cloud_llm_base, cache_path)
        local_llm = CachedLLMClient(local_llm_base, cache_path)

    classifier = EmailClassifier(cloud_llm=cloud_llm, local_llm=local_llm, config=config)

    max_thread_chars = resolve_max_thread_chars(config)

    # Run evaluation — flush cache even if something fails after evaluation
    try:
        print(f"Running evaluation (stages={args.stages}, parallelism={args.parallelism})...",
              file=sys.stderr)
        results, thinking_entries = await run_evaluation(
            golden_set=golden_set,
            classifier=classifier,
            stages=args.stages,
            max_thread_chars=max_thread_chars,
            parallelism=args.parallelism,
        )

        # Build run metadata
        config_bytes = json.dumps(config, sort_keys=True).encode()
        run_id = uuid.uuid4().hex
        cloud_cfg = config["llm"]["cloud"]
        local_cfg = config["llm"]["local"]
        all_categories = list(config["prompts"]["email_classification"]["categories"].keys())
        vip_categories = config.get("vip_senders", {}).get("categories", ["NEEDS_RESPONSE", "FYI"])
        meta = RunMeta(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            config_hash=hashlib.sha256(config_bytes).hexdigest()[:16],
            config_path=str(config_path),
            cloud_model=cloud_cfg["model"],
            local_model=local_cfg["model"],
            golden_set_path=str(golden_path),
            golden_set_count=len(golden_set),
            stages=args.stages,
            parallelism=args.parallelism,
            tag=args.tag or "",
            cloud_temperature=cloud_cfg["temperature"],
            cloud_max_tokens=cloud_cfg["max_tokens"],
            cloud_extra_body=cloud_cfg.get("extra_body"),
            local_temperature=local_cfg["temperature"],
            local_max_tokens=local_cfg["max_tokens"],
            local_extra_body=local_cfg.get("extra_body"),
            sender_system_prompt=classifier.sender_config["system"],
            email_system_prompt=classifier._build_email_prompt(all_categories),
            vip_email_system_prompt=classifier._build_email_prompt(vip_categories),
        )

        # Write results + thinking sidecar
        output_dir = Path(args.output_dir)
        output_path = build_output_path(output_dir, args.stages, args.tag or "", run_id)
        write_results(output_path, meta, results)
        write_thinking_sidecar(output_path, thinking_entries)

        # Optional inline metrics report / comparison (saves a separate
        # `python -m evals.report` invocation on the file we just wrote).
        maybe_report(meta, results, args.report, args.compare_to)

        # Summary
        errors = sum(1 for r in results if r.error)
        st_correct = sum(1 for r in results if r.sender_type_correct is True)
        st_total = sum(1 for r in results if r.sender_type_correct is not None)
        lb_correct = sum(1 for r in results if r.label_correct is True)
        lb_total = sum(1 for r in results if r.label_correct is not None)
        violations = sum(1 for r in results if r.privacy_violation)

        print(f"\nResults written to {output_path}", file=sys.stderr)
        print(f"  Threads: {len(results)} ({errors} errors)", file=sys.stderr)
        if st_total:
            pct = st_correct / st_total
            print(f"  Stage 1 accuracy: {st_correct}/{st_total} ({pct:.1%})", file=sys.stderr)
        if lb_total:
            pct = lb_correct / lb_total
            print(f"  Stage 2 accuracy: {lb_correct}/{lb_total} ({pct:.1%})", file=sys.stderr)
        if violations:
            print(f"  Privacy violations: {violations}", file=sys.stderr)
        if not args.no_cache:
            total_hits = cloud_llm.hits + local_llm.hits
            total_misses = cloud_llm.misses + local_llm.misses
            total_calls = total_hits + total_misses
            if total_calls:
                rate = total_hits / total_calls
                print(f"  Cache: {total_hits}/{total_calls} hits ({rate:.1%})", file=sys.stderr)
    finally:
        if not args.no_cache:
            cloud_llm.flush()
            local_llm.flush()


def cli():
    parser = argparse.ArgumentParser(description="Run classification evaluation")
    parser.add_argument("--golden-set", default="evals/golden_set.jsonl", help="Path to golden set JSONL")
    parser.add_argument("--config", help="Path to config.toml (default: ./config.toml)")
    parser.add_argument("--output-dir", default="evals/results/", help="Output directory for results")
    parser.add_argument("--stages", choices=VALID_STAGES, default="full",
                        help="Which stages to evaluate")
    parser.add_argument("--parallelism", type=int, default=None,
                        help="Concurrent evaluations (default: cloud_parallel from config)")
    parser.add_argument("--include-unreviewed", action="store_true",
                        help="Also evaluate threads not yet reviewed (default: reviewed only)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be evaluated")
    parser.add_argument("--tag", help="Tag for the results file name")
    parser.add_argument("--no-cache", action="store_true", help="Disable LLM response cache")
    parser.add_argument("--sender-type", choices=("person", "service"),
                        help="Only evaluate threads with this expected sender type")
    parser.add_argument("--max-threads", type=int, help="Max threads to evaluate")
    parser.add_argument("--local-only", action="store_true",
                        help="Shortcut for --stages stage2_only --sender-type person "
                             "(evaluate only the local classifier; no cloud creds needed)")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip the endpoint reachability check before evaluating")
    parser.add_argument("--preflight-timeout", type=float, metavar="SECONDS",
                        help="Timeout for the pre-run endpoint check (default: the local "
                             "model's request timeout; raise it for a slow cold model load)")
    parser.add_argument("--report", action="store_true",
                        help="Print a metrics report for this run after it completes")
    parser.add_argument("--compare-to", metavar="PATH",
                        help="After the run, print a comparison against this prior results file")
    # Config overrides (CLI always wins over config file)
    parser.add_argument("--cloud-model", help="Override cloud LLM model name")
    parser.add_argument("--local-model", help="Override local LLM model name")
    parser.add_argument("--cloud-temperature", "--cloud-temp", type=float, dest="cloud_temperature",
                        help="Override cloud LLM temperature")
    parser.add_argument("--local-temperature", "--local-temp", type=float, dest="local_temperature",
                        help="Override local LLM temperature")
    parser.add_argument("--cloud-max-tokens", type=int, help="Override cloud LLM max tokens")
    parser.add_argument("--local-max-tokens", type=int, help="Override local LLM max tokens")
    parser.add_argument("--cloud-timeout", type=int, help="Override cloud LLM request timeout (seconds)")
    parser.add_argument("--local-timeout", type=int, help="Override local LLM request timeout (seconds)")
    parser.add_argument("--cloud-extra-body",
                        help="JSON object merged into every cloud LLM request body")
    parser.add_argument("--local-extra-body",
                        help='JSON object merged into every local LLM request body '
                             '(e.g. \'{"chat_template_kwargs": {"enable_thinking": false}}\')')
    parser.add_argument("--cloud-no-think", action="store_true",
                        help="Disable thinking for the cloud LLM (sets "
                             "chat_template_kwargs.enable_thinking=false and "
                             "reasoning_effort=none)")
    parser.add_argument("--local-no-think", action="store_true",
                        help="Disable thinking for the local LLM (sets "
                             "chat_template_kwargs.enable_thinking=false and "
                             "reasoning_effort=none)")
    args = parser.parse_args()

    if args.stages not in VALID_STAGES:
        parser.error(f"Invalid stages: {args.stages}")

    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
