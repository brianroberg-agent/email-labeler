"""Harvest processed threads from Gmail as golden set data.

Pulls threads labeled agent/processed, infers ground truth from their
existing classification labels, and appends to JSONL.

Always appends to the output file, deduplicating by thread ID — it never
overwrites an existing golden set (that file also holds manual review state).
To start fresh, delete the file manually.

Usage:
    python -m evals.harvest --output evals/golden_set.jsonl --max-threads 200
    python -m evals.harvest --output evals/golden_set.jsonl --sender-type person
    python -m evals.harvest --gmail-label eval/harvest   # hand-picked threads
"""

import argparse
import asyncio
import json
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import httpx

from config_utils import substitute_env_vars
from evals import format_network_error
from evals.schemas import GoldenThread
from gmail_utils import get_header
from proxy_client import GmailProxyClient, ProxyAuthError, ProxyError, ProxyForbiddenError

_NETWORK_ERRORS = (
    httpx.ConnectError, httpx.TimeoutException, ProxyAuthError, ProxyForbiddenError, ProxyError,
)

# Label name -> (field, value) for ground truth inference
_SENDER_TYPE_LABELS = {
    "personal": "person",
    "non_personal": "service",
}

_CLASSIFICATION_LABELS = {
    "needs_response": "needs_response",
    "fyi": "fyi",
    "low_priority": "low_priority",
}


def load_eval_config(config_path: str | None = None) -> dict:
    """Load and preprocess config.toml from the given path."""
    path = Path(config_path) if config_path else Path(__file__).parent.parent / "config.toml"
    with open(path, "rb") as f:
        config = tomllib.load(f)
    return substitute_env_vars(config)


def infer_ground_truth(
    messages: list[dict],
    label_id_to_name: dict[str, str],
    labels_config: dict,
) -> tuple[str, str]:
    """Infer sender_type and classification label from message labelIds.

    Args:
        messages: Gmail message resources (with labelIds).
        label_id_to_name: Mapping from Gmail label ID to label name (e.g. "Label_7" -> "agent/personal").
        labels_config: The [labels] section from config.toml.

    Returns:
        (sender_type, label) tuple. sender_type is "person"/"service"/""
        and label is "needs_response"/"fyi"/"low_priority"/"".
    """
    # Build reverse map: label_name -> config_key
    name_to_config_key = {}
    for key in list(_SENDER_TYPE_LABELS) + list(_CLASSIFICATION_LABELS):
        label_name = labels_config.get(key, "")
        if label_name:
            name_to_config_key[label_name] = key

    sender_type = ""
    label = ""

    for msg in messages:
        for label_id in msg.get("labelIds", []):
            label_name = label_id_to_name.get(label_id, "")
            config_key = name_to_config_key.get(label_name, "")

            if config_key in _SENDER_TYPE_LABELS:
                sender_type = _SENDER_TYPE_LABELS[config_key]
            elif config_key in _CLASSIFICATION_LABELS:
                label = _CLASSIFICATION_LABELS[config_key]

    return sender_type, label


def load_existing_thread_ids(existing_path: Path) -> set[str]:
    """Thread IDs already present in the golden set file (empty if absent)."""
    existing_ids: set[str] = set()
    if existing_path.exists():
        with open(existing_path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                # Tolerate a corrupt/partial line (e.g. an interrupted append)
                # so one bad row can't abort every future harvest — the golden
                # set is hand-editable and append-only.
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: skipping malformed line {lineno} in {existing_path}",
                          file=sys.stderr)
                    continue
                thread_id = entry.get("thread_id")
                if thread_id:
                    existing_ids.add(thread_id)
    return existing_ids


def deduplicate(new_threads: list[GoldenThread], existing_path: Path) -> list[GoldenThread]:
    """Remove threads already present in the existing golden set file.

    Args:
        new_threads: Newly harvested threads.
        existing_path: Path to existing golden set JSONL file.

    Returns:
        Threads not already in the file.
    """
    existing_ids = load_existing_thread_ids(existing_path)
    return [t for t in new_threads if t.thread_id not in existing_ids]


def gmail_label_arg(value: str) -> str:
    """argparse type for --gmail-label: a non-empty Gmail label name, trimmed.

    An empty value (e.g. an unset shell variable) is rejected rather than
    treated as "no filter", which would silently widen a hand-picked harvest
    to the whole processed pool. A double quote is rejected because the name
    is embedded in a quoted Gmail search term and Gmail has no escape for it.
    """
    name = value.strip()
    if not name:
        raise argparse.ArgumentTypeError("expected a non-empty Gmail label name")
    if '"' in name:
        raise argparse.ArgumentTypeError("a Gmail label name cannot contain a double quote here")
    return name


def build_query(
    processed_label: str,
    filter_label: str | None = None,
    gmail_label: str | None = None,
) -> str:
    """Build the Gmail search query for the harvest fetch from resolved label names.

    Always matches ``processed_label``; ``filter_label`` (the classification
    label's Gmail name, not its config key) and ``gmail_label`` are ANDed in,
    in that order, when given.

    The appended labels are passed quoted (``label:"eval/harvest"``) so a
    user-supplied name containing spaces stays a single search term. Gmail's
    own search box shows labels in a hyphen-substituted form
    (``label:my-label``); if a quoted name returns nothing, that form is the
    documented fallback.
    """
    terms = [f"label:{processed_label}"]
    for name in (filter_label, gmail_label):
        if name:
            terms.append(f'label:"{name}"')
    return " ".join(terms)


async def harvest_threads(
    proxy: GmailProxyClient,
    config: dict,
    max_threads: int = 200,
    sender_type_filter: str | None = None,
    label_filter: str | None = None,
    gmail_label: str | None = None,
    skip_thread_ids: set[str] | None = None,
) -> list[GoldenThread]:
    """Fetch processed threads and build golden set entries.

    Args:
        proxy: Gmail proxy client.
        config: Full parsed config dict.
        max_threads: Maximum threads to fetch.
        sender_type_filter: Optional filter for "person" or "service".
        label_filter: Optional filter for classification label.
        gmail_label: Optional Gmail label name ANDed into the query, for
            hand-picked threads (e.g. eval/harvest). Must exist in Gmail.
        skip_thread_ids: Threads to leave unfetched (already in the golden
            set); they do not count against ``max_threads``.

    Returns:
        List of GoldenThread objects.
    """
    labels_config = config["labels"]
    skip_thread_ids = skip_thread_ids or set()
    now = datetime.now(timezone.utc).isoformat()

    # Build label ID -> name map
    try:
        labels_response = await proxy.list_labels()
    except _NETWORK_ERRORS as exc:
        print(f"Error: {format_network_error(exc, 'api-proxy')}", file=sys.stderr)
        sys.exit(1)
    label_id_to_name = {lbl["id"]: lbl["name"] for lbl in labels_response["labels"]}

    # Resolve the classification filter to its Gmail label name. An unmapped
    # key can never match a harvested thread (infer_ground_truth only knows
    # mapped keys), so it is an error, not a degraded query.
    filter_label_name = None
    if label_filter:
        filter_label_name = labels_config.get(label_filter)
        if not filter_label_name:
            print(f"Error: --label '{label_filter}' has no mapping in [labels]", file=sys.stderr)
            sys.exit(1)

    # A hand-picked label must exist in Gmail, or "no messages found" would
    # be indistinguishable from "nothing labeled yet". Match case-insensitively
    # but query with the spelling list_labels reported — the one known to
    # exist — so how Gmail treats case in label: search does not matter here.
    if gmail_label:
        canonical_names = {name.lower(): name for name in label_id_to_name.values()}
        canonical = canonical_names.get(gmail_label.lower())
        if canonical is None:
            print(f"Error: --gmail-label '{gmail_label}' is not a label in this Gmail account",
                  file=sys.stderr)
            sys.exit(1)
        gmail_label = canonical

    # Fetch message stubs with agent/processed label. Filters are ANDed into
    # the Gmail query so the fetch returns a dense pool of matching threads
    # instead of relying on the recent processed window happening to contain
    # them (label_filter is also re-checked per thread below, since a
    # thread's messages can carry multiple labels).
    query = build_query(labels_config["processed"], filter_label_name, gmail_label)
    try:
        response = await proxy.list_messages(
            q=query,
            max_results=max_threads * 3,  # Over-fetch since we group by thread
        )
    except _NETWORK_ERRORS as exc:
        print(f"Error: {format_network_error(exc, 'api-proxy')}", file=sys.stderr)
        sys.exit(1)
    msg_stubs = response.get("messages", [])

    if not msg_stubs:
        print(f"No messages found for query: {query}", file=sys.stderr)
        return []

    # Group by threadId
    thread_ids: dict[str, list[str]] = {}
    for stub in msg_stubs:
        tid = stub.get("threadId", stub["id"])
        thread_ids.setdefault(tid, []).append(stub["id"])

    print(f"Found {len(thread_ids)} unique threads from {len(msg_stubs)} messages", file=sys.stderr)

    # Threads already in the golden set are skipped before fetching and do
    # not consume max_threads slots — otherwise a hand-picked label larger
    # than the cap could never reach its older picks on a re-run.
    candidates = [tid for tid in thread_ids if tid not in skip_thread_ids]
    if len(candidates) < len(thread_ids):
        print(f"Skipping {len(thread_ids) - len(candidates)} threads already in the golden set",
              file=sys.stderr)

    # The fetch is a message-level budget with no pagination, so long threads
    # can exhaust it before max_threads distinct threads have surfaced. Say so
    # when that happened, since matching threads are then missing silently.
    # Counted AFTER the skip: on a re-run the window may hold only known
    # threads while the unharvested picks sit beyond it.
    if len(candidates) < max_threads and (
        response.get("nextPageToken") or len(msg_stubs) >= max_threads * 3
    ):
        print("Warning: message budget exhausted (max_threads * 3) before reaching "
              "--max-threads with new threads; some matching threads were not "
              "fetched — raise --max-threads, narrow the query, or remove the label "
              "from already-harvested threads", file=sys.stderr)

    # Fetch each thread and build golden entries
    results: list[GoldenThread] = []
    for i, tid in enumerate(candidates[:max_threads]):
        try:
            thread_data = await proxy.get_thread(tid)
            messages = thread_data.get("messages", [])
            if not messages:
                print(f"  Skipping thread {tid}: no messages", file=sys.stderr)
                continue

            # Sort chronologically
            messages.sort(key=lambda m: int(m.get("internalDate", "0")))

            # Infer ground truth
            sender_type, label = infer_ground_truth(messages, label_id_to_name, labels_config)
            if not sender_type or not label:
                print(f"  Skipping thread {tid}: incomplete labels (sender={sender_type}, label={label})",
                      file=sys.stderr)
                continue

            # Apply filters, naming the thread so a hand-picked shortfall is diagnosable
            if sender_type_filter and sender_type != sender_type_filter:
                print(f"  Skipping thread {tid}: sender_type={sender_type}, "
                      f"filter wants {sender_type_filter}", file=sys.stderr)
                continue
            if label_filter and label != label_filter:
                print(f"  Skipping thread {tid}: label={label}, filter wants {label_filter}",
                      file=sys.stderr)
                continue

            # Extract metadata
            senders = []
            seen: set[str] = set()
            for msg in messages:
                sender = get_header(msg["payload"]["headers"], "From")
                if sender and sender not in seen:
                    senders.append(sender)
                    seen.add(sender)

            first_headers = messages[0]["payload"]["headers"]
            subject = get_header(first_headers, "Subject")
            snippet = messages[-1].get("snippet", "")

            golden = GoldenThread(
                thread_id=tid,
                messages=messages,
                senders=senders,
                subject=subject,
                snippet=snippet,
                expected_sender_type=sender_type,
                expected_label=label,
                source="harvested",
                harvested_at=now,
                # Hand-picked rows carry labels the daemon (by hypothesis) got
                # wrong; the review TUI shows notes, so name the label there.
                notes=f"hand-picked via Gmail label {gmail_label}" if gmail_label else "",
            )
            results.append(golden)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{min(len(candidates), max_threads)} threads...", file=sys.stderr)

        except _NETWORK_ERRORS as exc:
            print(f"  Error fetching thread {tid}: {format_network_error(exc, 'api-proxy')}",
                  file=sys.stderr)
        except Exception as exc:
            print(f"  Error processing thread {tid}: {exc}", file=sys.stderr)

    print(f"Harvested {len(results)} threads", file=sys.stderr)
    return results


def write_golden_set(threads: list[GoldenThread], output_path: Path) -> None:
    """Append golden set entries to the JSONL file.

    Always appends — harvest never truncates an existing golden set, since that
    file also holds manual review state (confirmed labels, exclusions, notes).
    To start fresh, delete the file manually.
    """
    with open(output_path, "a") as f:
        for thread in threads:
            f.write(json.dumps(thread.to_dict()) + "\n")


async def main(args: argparse.Namespace) -> None:
    config = load_eval_config(args.config)
    proxy = GmailProxyClient(proxy_url=args.proxy_url)
    output_path = Path(args.output)

    threads = await harvest_threads(
        proxy=proxy,
        config=config,
        max_threads=args.max_threads,
        sender_type_filter=args.sender_type,
        label_filter=args.label,
        gmail_label=args.gmail_label,
        skip_thread_ids=load_existing_thread_ids(output_path),
    )

    if not threads:
        print("No threads to write.", file=sys.stderr)
        return

    # Backstop: known threads were skipped before fetching, but the file may
    # have gained rows during the run.
    threads = deduplicate(threads, output_path)
    if not threads:
        print("All threads already in golden set.", file=sys.stderr)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_golden_set(threads, output_path)
    print(f"Appended {len(threads)} threads to {output_path}", file=sys.stderr)


def cli():
    parser = argparse.ArgumentParser(description="Harvest processed emails for golden set")
    parser.add_argument("--output", default="evals/golden_set.jsonl", help="Output JSONL path")
    parser.add_argument("--max-threads", type=int, default=200, help="Max threads to fetch")
    parser.add_argument("--sender-type", choices=["person", "service"], help="Filter by sender type")
    parser.add_argument("--label", choices=list(_CLASSIFICATION_LABELS),
                        help="Filter by classification label")
    parser.add_argument(
        "--gmail-label", metavar="LABEL", type=gmail_label_arg,
        help="Only harvest threads also carrying this Gmail label (hand-picked test cases)",
    )
    parser.add_argument("--config", help="Path to config.toml (default: ./config.toml)")
    parser.add_argument("--proxy-url", help="API proxy URL (overrides PROXY_URL env var)")
    # Deprecated no-op: harvest always appends now. Kept so existing automation
    # that still passes --append doesn't abort with an argparse error.
    parser.add_argument("--append", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.append:
        print("Note: --append is deprecated and ignored; harvest always appends now.",
              file=sys.stderr)
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
