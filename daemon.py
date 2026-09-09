"""Email labeler daemon — main entry point.

Continuously polls Gmail for unclassified emails, classifies them
using a two-tier LLM system, and applies labels autonomously.

Privacy invariant: Person email bodies NEVER leave the local network.
"""

import asyncio
import logging
import os
import sys
import time
import tomllib
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv

from classifier import EmailClassifier, EmailLabel, SenderType, ThreadMetadata
from config_utils import substitute_env_vars
from gmail_utils import decode_body, get_header
from labeler import LabelManager, _get_priority
from llm_client import HALT_REPROBE_TIMEOUT, LLMBalanceError, LLMClient, LLMUnavailableError
from newsletter import (
    AssessmentSinkError,
    NewsletterClassifier,
    NewsletterTier,
    aggregate_theme_grades,
    count_records,
    covering_mount,
    is_newsletter,
    mount_persistence_warning,
    parse_send_date,
    read_mountinfo,
    running_in_container,
    sink_writability_warning,
    write_assessment,
)
from notify import HaltNotifier, format_downtime, halt_message, resume_message
from proxy_client import (
    TRANSIENT_TRANSPORT_ERRORS,
    GmailProxyClient,
    ProxyError,
    ProxyForbiddenError,
    ProxyUnavailableError,
)

load_dotenv()

_TIER_RANK = {
    NewsletterTier.POOR: 0,
    NewsletterTier.FAIR: 1,
    NewsletterTier.GOOD: 2,
    NewsletterTier.EXCELLENT: 3,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("email-labeler")


def quiet_http_logging() -> None:
    """Silence httpx/httpcore per-request INFO logs (issue #58).

    httpx logs 'HTTP Request: … 200 OK' at INFO for every poll, and the URL
    embeds the gmail_query ('-label:agent/processed -label:agent/attempted') —
    healthy polls read as alarms to anyone scanning the log. Only these library
    loggers are raised; the email-labeler logger stays at INFO. Canonical copy —
    the eval CLIs import it (evals may depend on daemon, never the reverse).
    Each entry point calls it explicitly rather than at import, so merely
    importing daemon's helpers never mutates process-wide logging.
    """
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


# Default cap on transcript chars sent to the classifier when config.toml omits
# max_thread_chars. Shared with the eval harness (evals/run_eval.py) so the two
# never drift — an eval must truncate transcripts exactly as production does.
DEFAULT_MAX_THREAD_CHARS = 16000


def load_config() -> dict:
    """Load configuration from config.toml.

    After parsing, any {env.VAR_NAME} placeholders in string values are
    replaced with the corresponding environment variable (empty string if unset).
    """
    config_path = Path(__file__).parent / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    return substitute_env_vars(config)


def resolve_int_env(env_var: str, default: int, minimum: int = 1) -> int:
    """Return an int from *env_var* if set and valid, otherwise *default*.

    Lets operators override numeric daemon settings (e.g. concurrency) per run
    without editing config.toml. {env.VAR} substitution only works for string
    config values, so numeric overrides are read here instead.

    A value that is unparseable OR below *minimum* falls back to the default with
    a warning rather than crashing. The lower bound matters: these values feed
    asyncio.Semaphore() and Gmail maxResults, where 0 deadlocks the poll loop and
    a negative crashes the daemon at startup.
    """
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        log.warning("Invalid %s=%r (expected an integer); using %d", env_var, raw, default)
        return default
    if value < minimum:
        log.warning(
            "%s=%d is below the minimum of %d; using %d", env_var, value, minimum, default
        )
        return default
    return value


def positive_int_setting(daemon_config: dict, key: str, default: int) -> int:
    """Return config.toml ``[daemon] key`` as an int >= 1, or *default* if absent.

    Raises ValueError with an operator-readable message for anything else — a
    quoted number, 0, a negative, a float, a bool. The halt machinery's
    settings are validated here at startup rather than where they are first
    used: ``halt_probe_interval_seconds`` feeds ``DaemonHalt.probe_due`` at the
    loop head, outside the cycle's try/except, and only once a halt has
    tripped — so a quoted value used to pass startup and kill the daemon at
    the first poll after a halt, exactly when self-heal was meant to take over
    (review of PR #81). Callers exit(1) on the error, like a missing label.
    """
    value = daemon_config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"config.toml [daemon] {key} must be an integer >= 1, got {value!r}"
        )
    return value


def resolve_newsletter_llm_endpoint() -> tuple[str, str]:
    """Return (base_url, api_key) for the newsletter grading LLM.

    Defaults to the cloud classification endpoint (CLOUD_LLM_URL / CLOUD_LLM_API_KEY)
    so a single provider serves both. Set NEWSLETTER_LLM_URL / NEWSLETTER_LLM_API_KEY
    when the newsletter model lives elsewhere — e.g. config.toml grades newsletters with
    a Claude model (`claude-sonnet-4-6`) that the cloud provider doesn't serve, so it must
    target a Claude-serving endpoint (Anthropic's OpenAI-compatible API, or a gateway).

    The override is atomic: once NEWSLETTER_LLM_URL is set, the key comes solely
    from NEWSLETTER_LLM_API_KEY (empty if unset) and never falls back to the cloud
    key — pairing an override endpoint with the cloud provider's credential would
    authenticate against the wrong provider and fail in a confusing way.
    """
    override_url = os.environ.get("NEWSLETTER_LLM_URL")
    if override_url:
        return override_url, os.environ.get("NEWSLETTER_LLM_API_KEY", "")
    return os.environ.get("CLOUD_LLM_URL", ""), os.environ.get("CLOUD_LLM_API_KEY", "")


class FailureTracker:
    """Counts strikes against threads the cycle-level attribution blamed, to
    break infinite retry loops.

    Fed only by the poll loop's post-gather attribution step
    (attribute_cycle_failures) — never inline from process_single_thread.
    Provider-shaped faults (LLM/proxy unavailability, exhausted 429s, 5xx)
    never land here: decision D5, deliberately reversing issue #26's proxy-5xx
    counting. A candidate failure lands only when correlation says the thread
    is the problem — its signature unique in the cycle with siblings
    succeeding, or a singleton cycle. At max_failures the thread is marked
    agent/attempted. In-memory and session-scoped: counts reset on daemon
    restart, so a thread that failed for a since-resolved reason gets another
    chance after a restart.
    """

    def __init__(self, max_failures: int = 5):
        self.max_failures = max_failures
        self._counts: dict[str, int] = {}
        self._given_up: list[str] = []  # threads abandoned since the last take_given_up()

    def record_failure(self, thread_id: str) -> None:
        self._counts[thread_id] = self._counts.get(thread_id, 0) + 1

    def should_give_up(self, thread_id: str) -> bool:
        return self._counts.get(thread_id, 0) >= self.max_failures

    def clear(self, thread_id: str) -> None:
        self._counts.pop(thread_id, None)

    def prune(self, active_thread_ids) -> None:
        """Drop failure counts for threads no longer pending.

        A thread that fails a few times (below the give-up threshold) and then
        disappears from the query — read, archived, or relabeled externally —
        would otherwise leak its count for the daemon's lifetime. Pruning each
        cycle to the threads on that cycle's page keeps the map bounded; a
        consecutively failing thread reappears every cycle, so it survives —
        unless a backlog larger than max_emails_per_cycle crowds it off a page,
        which only forgives strikes it had already accrued.
        """
        active = set(active_thread_ids)
        self._counts = {tid: n for tid, n in self._counts.items() if tid in active}

    def record_give_up(self, thread_id: str) -> None:
        """Record that a thread was abandoned (marked agent/attempted, no classification),
        so the per-cycle summary can report give-ups distinctly from classifications."""
        self._given_up.append(thread_id)

    def take_given_up(self) -> list[str]:
        """Return the thread ids given up since the last call, resetting the list."""
        out = self._given_up
        self._given_up = []
        return out


@dataclass
class CycleFailure:
    """One thread's failure in one poll cycle, recorded by process_single_thread
    for the poll loop's post-gather attribution step (decision D5 Rule 2).

    ``signature`` is the exception's class qualname — the correlation key: two
    threads failing with the same signature in one cycle look like one shared
    cause, not two coincidental poison threads. ``provider_shaped`` entries
    (proxy/LLM unavailability) never strike; they exist so the attribution can
    spot the single-thread masquerade (see MasqueradeTracker).
    """

    thread_id: str
    ids_to_mark: list[str]
    signature: str
    provider_shaped: bool = False


class MasqueradeTracker:
    """Watches for the single-thread masquerade: provider-shaped errors on one
    thread while siblings succeed (decision D5; issue #26's poison-thread
    scenario). Such a thread is retried forever — provider-shaped faults never
    strike — but must not fail silently: at ``max_failures`` qualifying cycles
    it becomes a suspect and the poll loop escalates with a distinct ERROR,
    repeated at most once per status interval while any suspect persists.

    Mirrors FailureTracker's shape: per-thread counter, success-clears, pruned
    each cycle to that cycle's page of pending threads, in-memory and
    session-scoped (so a page miss or a restart costs a suspect its count). A cycle
    only counts when it carries correlation evidence — exactly one thread
    failed provider-shaped AND a sibling was handled successfully. Singleton
    and zero-success cycles never increment it, so a genuine short provider
    outage with one pending thread never false-alarms; the per-thread WARNING
    each cycle remains its visibility. Only a thread's own success clears a
    count, whatever the cycle's shape — a success is never ambiguous.
    Local-tier LLM unavailability never lands here at all: the deliberately-
    offline MLX host makes "person threads defer while service siblings
    succeed" the routine local state (issue #24), and tracking it would
    false-alarm every night the laptop is closed.
    """

    def __init__(self, max_failures: int = 5):
        self.max_failures = max_failures
        self._counts: dict[str, int] = {}
        self._last_escalation: float | None = None

    def record_masquerade(self, thread_id: str) -> None:
        self._counts[thread_id] = self._counts.get(thread_id, 0) + 1

    def clear(self, thread_id: str) -> None:
        self._counts.pop(thread_id, None)

    def prune(self, active_thread_ids) -> None:
        """Drop counts for threads no longer pending (mirrors FailureTracker.prune)."""
        active = set(active_thread_ids)
        self._counts = {tid: n for tid, n in self._counts.items() if tid in active}

    def suspects(self) -> dict[str, int]:
        """Threads at/above the escalation threshold, with their cycle counts."""
        return {tid: n for tid, n in self._counts.items() if n >= self.max_failures}

    def escalation_line(self, now: float, status_interval: float) -> str | None:
        """Return the distinct escalation ERROR line, or None when quiet.

        Its own small throttled emitter (idle_report only runs on idle cycles):
        emits when a suspect first exists, then at most once per
        ``status_interval`` while any suspect persists. The throttle resets when
        no suspect remains, so a future suspect escalates immediately.
        """
        suspects = self.suspects()
        if not suspects:
            self._last_escalation = None
            return None
        if self._last_escalation is not None and now - self._last_escalation < status_interval:
            return None
        self._last_escalation = now
        detail = ", ".join(f"{tid} ({n} cycles)" for tid, n in sorted(suspects.items()))
        return (
            f"Thread(s) failing with provider-shaped errors while siblings "
            f"succeed: {detail} — retrying forever per the failure model (D5), "
            f"never abandoned; investigate the thread or the provider route"
        )


@dataclass
class CachedEmailResult:
    """ResultCache payload for the email pipeline: recorded after Stage 2
    succeeds, consumed by a later cycle's label-write retry."""

    label: EmailLabel
    sender_type: SenderType


@dataclass
class CachedNewsletterResult:
    """ResultCache payload for the newsletter pipeline.

    ``story_results`` is cached because a sink-fault retry rebuilds the JSONL
    record from it. ``assessment_written`` flips once ``write_assessment``
    returns, so a labels-only retry never appends a duplicate record for the
    same fingerprint; a fingerprint invalidation legitimately re-grades and
    re-appends, with D18's newest-timestamp dedup as the read-side semantics.
    """

    best_tier: NewsletterTier | None
    all_themes: dict[str, str]
    story_results: list
    assessment_written: bool = False


class ResultCache:
    """Keeps finished classification results across cycles while their label
    writes keep failing (issue #29).

    A transient write-phase fault used to discard the whole classification and
    re-run it every cycle — Stage 1 + Stage 2 (the scarce local GPU pass for
    person threads), or a full newsletter extraction + grading. Caching the
    result means a failed write costs only a write retry.

    Maps thread_id → (fingerprint, payload). The fingerprint is the sorted
    tuple of the thread's message ids: a new message changes the input, so a
    mismatch drops the entry and the thread classifies fresh. In-memory and
    session-scoped like FailureTracker; entries are cleared on a successful
    label write and pruned each cycle to the threads on that cycle's page.

    Best-effort by construction, not a durability guarantee: a restart drops
    the whole cache, and a backlog larger than max_emails_per_cycle can push a
    still-pending thread off a cycle's page. Either way the thread reclassifies
    when it next comes round — the cost is LLM spend, plus (for a newsletter) a
    second assessment record for the same content, which D18's newest-timestamp
    dedup on read absorbs.
    """

    def __init__(self):
        self._entries: dict[str, tuple[tuple[str, ...], object]] = {}

    def get(self, thread_id: str, fingerprint: tuple[str, ...]) -> object | None:
        """Return the cached payload if *fingerprint* still matches, else None.

        A mismatched fingerprint means the thread's messages changed since it
        was classified — the stale entry is dropped so the caller reclassifies.
        """
        entry = self._entries.get(thread_id)
        if entry is None:
            return None
        cached_fingerprint, payload = entry
        if cached_fingerprint != fingerprint:
            del self._entries[thread_id]
            return None
        return payload

    def put(self, thread_id: str, fingerprint: tuple[str, ...], payload: object) -> None:
        self._entries[thread_id] = (fingerprint, payload)

    def clear(self, thread_id: str) -> None:
        self._entries.pop(thread_id, None)

    def prune(self, active_thread_ids) -> None:
        """Drop entries for threads absent from this cycle (mirrors FailureTracker.prune).

        A cached result whose thread leaves the query — labeled externally,
        archived, or given up — would otherwise leak for the daemon's lifetime;
        a thread with a pending write re-matches the query every cycle, so it
        normally survives. Normally, because *active_thread_ids* is the cycle's
        max_emails_per_cycle page rather than the whole pending set: a large
        backlog can evict a still-pending thread, which then reclassifies when
        it comes back (class docstring — an optimisation, not a guarantee).
        """
        active = set(active_thread_ids)
        self._entries = {tid: e for tid, e in self._entries.items() if tid in active}


# Fallback for config.toml [daemon] balance_halt_strikes when the key is absent.
# The value and its rationale are homed there (decision D7's one-home rule; D22,
# issue #73) — do not restate them here.
DEFAULT_BALANCE_HALT_STRIKES = 3


class DaemonHalt:
    """Halt state for ONE function's account-level faults (provider out of funds).

    Unlike a poison thread (FailureTracker's territory), an out-of-funds provider
    fails EVERY request it serves: retrying per-thread just re-fails that
    function's whole backlog every cycle, against a provider that cannot answer
    any of it. Tripping this stops the function. The slot self-heals (decision
    D22): the poll loop re-probes ``probe_client`` on a slow schedule while
    tripped and calls ``clear()`` when the provider answers again — a restart
    clears it too, since the state is in-memory. First tripper wins: threads in
    one asyncio.gather cycle may race to trip, and the reason must stay stable.

    Tripping takes ``strikes_to_trip`` consecutive balance faults (config.toml
    ``[daemon] balance_halt_strikes``), counted by ``record_balance_error``;
    ``record_success`` resets the count (D22) but does not clear a tripped
    slot — only the probe does that.

    One slot per function, held together by FunctionHalts.
    """

    def __init__(self, strikes_to_trip: int = DEFAULT_BALANCE_HALT_STRIKES):
        self.reason: str | None = None
        self.strikes_to_trip = strikes_to_trip
        self.consecutive_faults = 0
        # The exception that tripped the slot — provenance for the notification.
        self.fault: LLMBalanceError | None = None
        # The LLMClient whose provider reported the fault; re-probed while tripped.
        self.probe_client: LLMClient | None = None
        # The cloud-tier client seen in the CURRENT streak, if any: preferred over
        # the third fault's client at trip, since the local tier is a paid
        # provider only under D4's eval-only stand-in (review of PR #81).
        self._streak_cloud_client: LLMClient | None = None
        # time.monotonic() at trip and at the last probe (scheduling), and
        # time.time() at trip (the notification's wall-clock "since").
        self.tripped_at: float | None = None
        self.tripped_wall: float | None = None
        self.last_probe_at: float | None = None
        # True once a halt push has LANDED (send returned True); a failed push is
        # re-attempted on the probe cadence, paced by last_notify_attempt_at.
        self.notified = False
        self.last_notify_attempt_at: float | None = None

    def trip(
        self,
        reason: str,
        *,
        fault: LLMBalanceError | None = None,
        probe_client: LLMClient | None = None,
        now: float | None = None,
    ) -> None:
        if self.reason is None:
            self.reason = reason
            if fault is not None:
                # Kept for the notification's four scalar fields, not for its
                # frames: a live traceback would pin the thread JSON, transcript,
                # request body and response below the raise for the whole halt
                # (hours to days). Same object, so `halt.fault is exc` still holds.
                fault.__traceback__ = None
            self.fault = fault
            self.probe_client = probe_client
            self.tripped_at = time.monotonic() if now is None else now
            self.tripped_wall = time.time()
            self.last_probe_at = self.tripped_at

    def record_balance_error(
        self,
        exc: LLMBalanceError,
        *,
        probe_client: LLMClient | None = None,
        now: float | None = None,
    ) -> bool:
        """Count one balance fault; trip at ``strikes_to_trip`` consecutive.

        Returns True only on the call that trips the slot. A fault on an
        already-tripped slot changes nothing (first tripper wins). The client
        recorded for the re-probe is a cloud-tier one if any fault in the
        streak came from the cloud tier, otherwise the tripping fault's.
        """
        if self.tripped:
            return False
        self.consecutive_faults += 1
        if probe_client is not None and exc.tier != "local" and self._streak_cloud_client is None:
            self._streak_cloud_client = probe_client
        if self.consecutive_faults < self.strikes_to_trip:
            return False
        self.trip(
            str(exc), fault=exc, probe_client=self._streak_cloud_client or probe_client, now=now
        )
        return True

    def record_success(self) -> None:
        """A request this function's provider answered: the faults were not
        consecutive after all. Does not clear a tripped slot (D22: only the
        probe resumes a halted function)."""
        self.consecutive_faults = 0
        self._streak_cloud_client = None

    def clear(self) -> None:
        """Resume: the probe got an answer. Back to the untripped initial state."""
        self.reason = None
        self.consecutive_faults = 0
        self.fault = None
        self.probe_client = None
        self._streak_cloud_client = None
        self.tripped_at = None
        self.tripped_wall = None
        self.last_probe_at = None
        self.notified = False
        self.last_notify_attempt_at = None

    @property
    def tripped(self) -> bool:
        return self.reason is not None

    def probe_due(self, now: float, interval: float) -> bool:
        """True when this tripped slot's next re-probe is due: ``interval``
        seconds since the trip, then since the last probe (D22)."""
        if not self.tripped or self.last_probe_at is None:
            return False
        return now - self.last_probe_at >= interval


class FunctionHalts:
    """Per-function halt state (decision D5's scope rule; D19).

    The two functions fail independently, so an out-of-funds provider stops only
    the function whose requests it was serving: a newsletter-tier balance fault
    halts newsletter grading while email triage keeps classifying, and vice
    versa. Each function owns a DaemonHalt slot (first-tripper-wins, cleared by
    a successful re-probe of its own provider — D22); the poll loop stands down
    entirely only when EVERY enabled function is halted.

    "Enabled" is a deployment fact, not a runtime one: newsletter grading iff
    [newsletter] is configured, email triage iff NEWSLETTER_ONLY is unset. A
    function that isn't running can't be halted, and must never be counted as
    halted when deciding whether anything is left to do.

    When the two functions share one LLM client ([newsletter.llm] absent), a
    shared-provider fault trips both slots within a cycle or two as each
    function hits its own request — correct, since the fault does disable both.
    """

    def __init__(
        self,
        email_enabled: bool = True,
        newsletter_enabled: bool = False,
        strikes_to_trip: int = DEFAULT_BALANCE_HALT_STRIKES,
    ):
        self.email = DaemonHalt(strikes_to_trip)
        self.newsletter = DaemonHalt(strikes_to_trip)
        self.email_enabled = email_enabled
        self.newsletter_enabled = newsletter_enabled

    def enabled_slots(self) -> list[tuple[str, DaemonHalt]]:
        """(function name, slot) for each ENABLED function, in a fixed order."""
        slots = []
        if self.email_enabled:
            slots.append(("email triage", self.email))
        if self.newsletter_enabled:
            slots.append(("newsletter grading", self.newsletter))
        return slots

    @property
    def all_halted(self) -> bool:
        """True when every enabled function is halted — nothing is left to poll for."""
        slots = self.enabled_slots()
        return bool(slots) and all(h.tripped for _name, h in slots)

    @property
    def any_halted(self) -> bool:
        return any(h.tripped for _name, h in self.enabled_slots())

    @property
    def email_only_halted(self) -> bool:
        """Email triage is halted while newsletter grading still runs — the one
        direction the Gmail query can narrow (a `to:recipient` clause keeps the
        halted function's backlog from crowding the page; the mirror direction
        can't be expressed as "not to:recipient" reliably)."""
        return (
            self.email_enabled
            and self.email.tripped
            and self.newsletter_enabled
            and not self.newsletter.tripped
        )

    def halted_summary(self) -> str:
        """`function: reason` for each halted enabled function — the operator line."""
        return "; ".join(
            f"{name}: {h.reason}" for name, h in self.enabled_slots() if h.tripped
        )


async def reprobe_halts(
    halts: FunctionHalts, now: float, interval: float
) -> list[tuple[str, float]]:
    """Re-probe each halted enabled function whose probe is due; clear the ones
    whose provider answers (decision D22, issue #73).

    One request (``LLMClient.probe``: the client's own request shape and a
    fixed innocuous prompt, no email content) per ``interval`` seconds per
    halted function, through THAT function's own client — email and newsletter
    may sit on different providers. Every due slot is probed at once
    (``asyncio.gather``) with ``HALT_REPROBE_TIMEOUT``, so the loop head stalls
    for at most one short timeout however many slots hang, keeping the
    heartbeat well inside the healthcheck threshold. A probe that answers 200
    clears the slot and the function resumes in this very cycle (the loop
    re-probes at its head, before the poll); a probe that
    does not leaves the slot tripped and logs below ERROR (the per-cycle halt
    line is already the loudness — an hourly ERROR would only repeat it).
    Nothing raised by a probe escapes: this runs outside the poll loop's
    try/except.

    Returns ``(function name, seconds halted)`` for each function that resumed,
    in enabled-slot order, so the caller can notify and undo any halt-time
    state of its own.
    """
    due = [(name, slot) for name, slot in halts.enabled_slots() if slot.probe_due(now, interval)]
    for _name, slot in due:
        slot.last_probe_at = now

    async def probe_one(name: str, slot: DaemonHalt) -> tuple[str, float] | None:
        if slot.probe_client is None:
            log.info("%s still halted — no provider client recorded to re-probe", name)
            return None
        try:
            result = await slot.probe_client.probe(timeout=HALT_REPROBE_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 — a probe fault must never kill the loop
            log.warning("%s still halted — re-probe raised %s: %s", name, type(exc).__name__, exc)
            return None
        if result.ok:
            downtime = now - (slot.tripped_at if slot.tripped_at is not None else now)
            log.info(
                "%s resumed — provider answered the re-probe after %s halted; "
                "normal processing resumes in this cycle",
                name, format_downtime(downtime),
            )
            slot.clear()
            return (name, downtime)
        log.info(
            "%s still halted — re-probe failed (%s); next probe in %ds",
            name, result.detail() or "no response detail", interval,
        )
        return None

    outcomes = await asyncio.gather(*(probe_one(name, slot) for name, slot in due))
    return [outcome for outcome in outcomes if outcome is not None]


async def notify_new_halts(
    halts: FunctionHalts, notifier: HaltNotifier, probe_interval: int, now: float
) -> None:
    """Push once per halt (D22): each tripped slot whose push has not yet landed.

    Called at the top of each poll cycle, BEFORE the re-probe, so a halt that
    trips and resumes between two cycles still reports both events in order.
    The halt push therefore lags the trip by at most one poll interval. A push
    that does not land (``send`` returns False — ntfy unreachable, say, when a
    host reboot restarts both containers) is re-attempted on the probe cadence:
    the first attempt is immediate, later ones ``probe_interval`` apart, until
    one succeeds — so a dead ntfy costs one POST per probe interval, not one
    per cycle, and a halt is still announced once ntfy is back. Wrapped so that
    even a notifier bug cannot reach the loop (``HaltNotifier.send`` already
    never raises).
    """
    if not notifier.enabled:
        return
    try:
        for name, slot in halts.enabled_slots():
            if not slot.tripped or slot.notified:
                continue
            if (
                slot.last_notify_attempt_at is not None
                and now - slot.last_notify_attempt_at < probe_interval
            ):
                continue
            slot.last_notify_attempt_at = now
            landed = await notifier.send(
                *halt_message(
                    name, slot.fault, tripped_wall=slot.tripped_wall,
                    probe_interval=probe_interval,
                )
            )
            if landed:
                slot.notified = True
    except Exception as exc:  # noqa: BLE001 — a notification must never fail the daemon
        log.warning("Halt notification failed (%s: %s)", type(exc).__name__, exc)


async def notify_resumes(notifier: HaltNotifier, resumed: list[tuple[str, float]]) -> None:
    """Push once per resume (D22), in the cycle the probe answered."""
    try:
        for name, downtime in resumed:
            await notifier.send(*resume_message(name, downtime))
    except Exception as exc:  # noqa: BLE001 — a notification must never fail the daemon
        log.warning("Resume notification failed (%s: %s)", type(exc).__name__, exc)


def attribute_cycle_failures(
    thread_items: list[tuple[str, list[str]]],
    results: list,
    failures: list["CycleFailure"],
    failure_tracker: FailureTracker,
    masquerade_tracker: MasqueradeTracker,
) -> list["CycleFailure"]:
    """Post-gather attribution (decision D5 Rule 2): decide which of this cycle's
    failures were the thread's own fault, strike them, and return the entries
    that just struck out — the poll loop marks those agent/attempted.

    Blame by correlation, cycle-level:

      * A candidate failure (Timeout / RuntimeError / unexpected Exception)
        counts a strike iff its signature is unique among the cycle's candidate
        failures AND — when other threads ATTEMPTED work this cycle — at least
        one of them was handled successfully. Everything else is treated as
        shared cause (provider, proxy, disk, or our own config/code): no
        strikes, one ERROR line, backlog kept.
      * The correlation denominator is the threads that attempted work, not
        every thread the cycle fetched: only a thread that succeeded or
        recorded a failure is evidence about anything. Threads that merely
        DEFERRED — their function halted, the local tier offline, a
        NEWSLETTER_ONLY skip, a 403-rejected write, an assessment-sink fault —
        tried nothing and committed nothing, so they neither blame nor absolve.
      * A singleton cycle counts: with no attempting siblings to correlate
        against, bounded strikes to a findable agent/attempted is the honest
        fallback — and the poison-thread case is typically a singleton
        (everything else processed away).
      * Provider-shaped failures never strike. When exactly one thread failed
        provider-shaped and a sibling succeeded, its masquerade counter
        advances (see MasqueradeTracker); ambiguous cycles (singleton,
        zero-success) never advance it. Only a thread's own success clears a
        count, whatever the cycle's shape — a success is never ambiguous.

    Marking eligibility derives from THIS cycle's strikes only — never from raw
    tracker counts — so a stale at-threshold count left by a failed marker
    write can never mark a thread that has since stopped failing.
    """
    any_success = any(result is True for result in results)
    # Correlate over the threads that ATTEMPTED work — handled successfully, or
    # recorded a CycleFailure. A deferral-only thread is not correlation
    # evidence: leaving it in the denominator made a cycle of "one poisoned
    # thread + one permanently-deferred thread" look multi-thread-and-
    # zero-success forever, so the poisoned thread never struck and never
    # converged to a findable agent/attempted — silently voiding D5 Rule 1's
    # set-aside guarantee (a halted function re-fetches and re-defers its
    # threads every cycle, so the shielding persists until it resumes).
    failed_threads = {f.thread_id for f in failures}
    attempted = {
        tid
        for (tid, _msg_ids), result in zip(thread_items, results)
        if result is not False or tid in failed_threads
    }
    multi_thread = len(attempted) > 1
    thread_blame = not multi_thread or any_success

    candidates = [f for f in failures if not f.provider_shaped]
    signature_counts = Counter(f.signature for f in candidates)
    striking = [f for f in candidates if thread_blame and signature_counts[f.signature] == 1]
    shared = [f for f in candidates if not (thread_blame and signature_counts[f.signature] == 1)]
    if shared:
        log.error(
            "%d thread(s) failed this cycle with no correlation evidence of a "
            "thread-specific fault (signatures: %s) — shared cause suspected "
            "(D5): no strikes, backlog kept",
            len(shared),
            ", ".join(sorted({f.signature for f in shared})),
        )
    for f in striking:
        failure_tracker.record_failure(f.thread_id)

    # Masquerade bookkeeping (D5): success clears; exactly one provider-shaped
    # failing thread plus a successful sibling increments. Local-tier LLM
    # deferrals never appear in `failures` at all (see the LLMUnavailableError
    # arm), so a closed MLX laptop can't accrue here.
    #
    # This half needs no attempting-denominator treatment: it moves only on
    # POSITIVE evidence — a thread that succeeded, and provider-shaped entries
    # that are attempts by construction — and a deferral-only thread is neither,
    # so it can neither increment nor clear a counter. (The blame rule above is
    # the asymmetric one: its no-siblings fallback is to blame, so a phantom
    # sibling actively suppresses strikes.) The prune below stays over every
    # FETCHED thread, as does FailureTracker's in summarize_cycle: a deferred
    # thread is still pending, and pruning it would drop a count it needs next
    # cycle.
    for (tid, _msg_ids), result in zip(thread_items, results):
        if result is True:
            masquerade_tracker.clear(tid)
    provider_threads = {f.thread_id for f in failures if f.provider_shaped}
    if len(provider_threads) == 1 and any_success:
        masquerade_tracker.record_masquerade(provider_threads.pop())
    masquerade_tracker.prune(tid for tid, _msg_ids in thread_items)

    return [f for f in striking if failure_tracker.should_give_up(f.thread_id)]


async def _mark_thread_attempted(
    thread_id: str,
    msg_ids: list[str],
    failure_tracker: FailureTracker,
    label_manager: LabelManager,
) -> bool:
    """Mark a struck-out thread agent/attempted so it stops being retried every
    cycle. Called from the poll loop for threads the cycle attribution just
    struck out — attribute_cycle_failures owns the strike accounting (decision
    D5); this function owns only the guarded marker write.

    Uses agent/attempted (not agent/processed): the thread is excluded from the
    unprocessed query either way, but the distinct label keeps abandoned threads
    findable and separate from successfully-classified mail.

    Returns True if the marker landed (thread abandoned), or False if the write
    failed and the thread stays pending.
    """
    try:
        await label_manager.mark_attempted(msg_ids)
    except ProxyUnavailableError as exc:
        # The proxy is transiently down, so the agent/attempted marker can't be written
        # right now — expected during an outage, not a bug. We return without recording
        # the give-up and without clearing the count, so the count stays at the
        # threshold and the thread's next strike re-qualifies it for marking once the
        # proxy recovers. Log a clean warning rather than spamming a traceback for an
        # expected transient condition. (The thread keeps re-matching the query until it
        # is actually labeled; if its classification already succeeded, the ResultCache
        # serves it back in that window, so retry cycles re-attempt only the write —
        # issue #29.)
        log.warning(
            "Proxy unavailable marking stuck thread %s attempted — will retry next cycle: %s",
            thread_id, exc,
        )
        return False
    except ProxyForbiddenError as exc:
        # A proxy 403 on the marker write is a human answer — the operator rejected
        # the confirmation, or the op is blocked — not a failure (decision D6). One
        # clean line, no traceback. The rejection blocks the marker; the thread stays
        # give-up-eligible (its count is left at the threshold, so its next strike
        # re-offers the marker write next cycle). A rejection never *causes* the
        # give-up — the strikes that reached the threshold came from elsewhere.
        log.info(
            "Marker write rejected/blocked by the proxy for thread %s — "
            "re-offering next cycle: %s",
            thread_id, exc,
        )
        return False
    except Exception:
        # An unexpected, likely-permanent marker-write failure (a real bug, or a
        # misconfigured label): keep the full traceback so it stays diagnosable rather
        # than looping silently as if benign.
        log.exception("Could not mark stuck thread %s attempted", thread_id)
        return False
    # Logged once here — AFTER the marker write actually lands — not before it: a
    # transient write-outage can retry this for several cycles, and logging on every
    # attempt would re-spam ERROR for what is an expected, recoverable condition.
    log.error(
        "Thread %s failed %d+ times — marked agent/attempted to break the retry loop",
        thread_id, failure_tracker.max_failures,
    )
    failure_tracker.record_give_up(thread_id)
    # Give-ups return False from process_single_thread (they are failures, not
    # handled work — D5), so summarize_cycle's success-clear never runs for
    # them: clear the count here, beside the successful mark.
    failure_tracker.clear(thread_id)
    return True


def format_thread_transcript(messages: list[dict], max_chars: int) -> str:
    """Format Gmail messages into a chronological thread transcript.

    Messages should be pre-sorted by internalDate (chronological).
    If the transcript exceeds max_chars, the oldest messages are dropped
    and a truncation notice is prepended.

    Args:
        messages: List of Gmail message resources (with payload.headers and payload.body).
        max_chars: Maximum character limit for the transcript.

    Returns:
        Formatted transcript string.
    """
    parts = []
    for msg in messages:
        headers = msg["payload"]["headers"]
        sender = get_header(headers, "From")
        date = get_header(headers, "Date")
        body = decode_body(msg["payload"])

        part = f"--- Message from {sender} on {date} ---\n{body}"
        parts.append(part)

    full = "\n\n".join(parts)
    if len(full) <= max_chars:
        return full

    # Truncate from the oldest messages first
    while len(parts) > 1:
        parts.pop(0)
        candidate = "[Earlier messages truncated]\n\n" + "\n\n".join(parts)
        if len(candidate) <= max_chars:
            return candidate

    # Single message still too long — hard truncate
    return parts[0][:max_chars]


async def process_single_thread(
    thread_id: str,
    msg_ids: list[str],
    proxy_client: GmailProxyClient,
    classifier: EmailClassifier,
    label_manager: LabelManager,
    cloud_sem: asyncio.Semaphore,
    local_sem: asyncio.Semaphore,
    max_thread_chars: int,
    newsletter_classifier: NewsletterClassifier | None = None,
    newsletter_recipient: str = "",
    newsletter_output_file: str = "",
    newsletter_only: bool = False,
    cycle_failures: "list[CycleFailure] | None" = None,
    fetch_sem: asyncio.Semaphore | None = None,
    local_deferrals: list[str] | None = None,
    halts: "FunctionHalts | None" = None,
    result_cache: "ResultCache | None" = None,
) -> bool:
    """Process a single thread through the classification pipeline.

    Fetches the full thread, formats all messages into a transcript,
    classifies once, and applies labels to all messages in the thread.
    A classification whose label write failed in a prior cycle is reused from
    result_cache (issue #29) instead of being re-run, as long as the thread's
    message ids are unchanged.

    Uses semaphores to bound concurrent LLM requests:
    - cloud_sem: acquired for Stage 1 (sender classification) and Stage 2 SERVICE emails
    - local_sem: acquired for Stage 2 PERSON emails (local MLX LLM)

    If the local LLM is unreachable, ConnectError is raised immediately.
    If it times out (e.g. model loading), the configured timeout applies.

    Enforces no-downgrade rule: if the thread already has a classification
    with equal or higher priority, it is skipped.

    Failures are never handled inline (decision D5): each failure arm records a
    CycleFailure into ``cycle_failures`` — candidate or provider-shaped — and
    returns False; the poll loop's post-gather attribution step decides strikes
    and marking (attribute_cycle_failures). Three arms deliberately record no
    CycleFailure, for three different reasons: the assessment-sink arm
    (AssessmentSinkError), because a sink fault is shared-cause by construction
    (the disk, not the thread) — nothing for correlation to weigh, retried
    forever; the proxy-403 arm (ProxyForbiddenError), because a rejected write
    is a human answer rather than a failure (decision D6) and is simply
    re-offered; and a LOCAL-tier LLMUnavailableError, a routine deferral
    counted in ``local_deferrals`` for the per-cycle summary instead (issue
    #24).

    Halts are per-function (decision D5's scope rule, D19): a balance fault
    trips only the halted function's slot in ``halts``, and a thread whose own
    function is halted defers (returns False) without a CycleFailure — a halt
    is a deferral, not a failure to attribute.

    Args:
        thread_id: Gmail thread ID.
        msg_ids: List of message IDs in the thread (from list_messages stubs).
        proxy_client: Gmail proxy client.
        classifier: Email classifier instance.
        label_manager: Label manager instance.
        cloud_sem: Semaphore bounding concurrent cloud LLM requests.
        local_sem: Semaphore bounding concurrent local LLM requests.
        max_thread_chars: Maximum characters for thread transcript.

    Returns:
        True if the thread was successfully classified and labeled,
        False if skipped or errored.
    """
    # Messages to mark processed if we give up. Starts as the query stubs (all we
    # know before fetching); upgraded to the full message list once the thread is
    # fetched, so a give-up marks every message and the thread stops re-matching
    # the query (otherwise unmarked siblings re-surface it and the retry loop the
    # give-up exists to break never converges).
    ids_to_mark = msg_ids
    try:
        # Nothing this daemon still does can succeed: every enabled function is
        # halted (each one's provider is out of funds account-wide), so don't
        # even fetch. A PARTIAL halt keeps going — this thread's function isn't
        # known until it is routed, so the function-aware checks sit below
        # newsletter detection instead (decision D5's scope rule).
        if halts is not None and halts.all_halted:
            return False

        # Fetch full thread (all messages in one API call). Bounded by fetch_sem so
        # a large max_emails_per_cycle can't burst one concurrent proxy read per
        # thread (the cloud/local semaphores gate only the classify calls).
        async with (fetch_sem or nullcontext()):
            thread_data = await proxy_client.get_thread(thread_id)
        messages = thread_data.get("messages", [])
        if not messages:
            log.warning("Thread %s has no messages, skipping", thread_id)
            return False

        # Sort chronologically
        messages.sort(key=lambda m: int(m.get("internalDate", "0")))

        # Full message list — the thread may carry more messages than the query
        # returned. Upgrade the give-up target from the query stubs to every message
        # id now (see the ids_to_mark note above), so any give-up or skip-and-mark
        # path below marks the whole thread and it stops re-matching the query.
        # Every post-fetch branch shares this single computation.
        all_msg_ids = [msg["id"] for msg in messages]
        ids_to_mark = all_msg_ids

        # Input fingerprint for the session ResultCache (issue #29): a new
        # message changes the sorted id tuple, so a cached classification is
        # only ever reused for exactly the content it was computed from.
        fingerprint = tuple(sorted(all_msg_ids))

        # Newsletter detection — route to newsletter pipeline if applicable
        if newsletter_classifier and newsletter_recipient:
            if is_newsletter(messages, newsletter_recipient):
                # This thread belongs to the newsletter function, and that
                # function is halted (its provider is out of funds): defer —
                # no strike, no marker, nothing committed — and let email
                # triage carry on. Also catches the sibling that tripped the
                # halt while this thread was fetching, before any LLM request.
                if halts is not None and halts.newsletter.tripped:
                    return False

                first_headers = messages[0]["payload"]["headers"]
                subject = get_header(first_headers, "Subject")
                sender = get_header(first_headers, "From")
                send_date = parse_send_date(
                    get_header(first_headers, "Date"), messages[0].get("internalDate")
                )

                # Reuse a finished grading whose write failed in a prior cycle
                # (issue #29): skip re-extraction/re-grading and go straight to
                # the writes it still owes.
                cached_nl = result_cache.get(thread_id, fingerprint) if result_cache else None
                if cached_nl is not None:
                    log.info(
                        "Newsletter thread %s: reusing cached grading from a prior "
                        "cycle (write still pending)",
                        thread_id,
                    )
                    story_results = cached_nl.story_results
                    best_tier = cached_nl.best_tier
                    all_themes = cached_nl.all_themes
                else:
                    transcript = format_thread_transcript(messages, max_thread_chars)

                    try:
                        async with cloud_sem:
                            # Grading is several LLM calls; each one the provider
                            # answers resets the newsletter slot's consecutive-fault
                            # count as it lands (D22 item 2), so an answer followed
                            # by a dropped connection still counts as an answer.
                            story_results = await newsletter_classifier.classify_newsletter(
                                transcript,
                                on_answer=(
                                    halts.newsletter.record_success if halts is not None else None
                                ),
                            )
                    except LLMBalanceError as exc:
                        # The NEWSLETTER function's provider is out of funds. Its
                        # [newsletter.llm] endpoint is configured independently of
                        # the email tiers (and LLMBalanceError carries no function
                        # provenance), so the call site is what tells the two
                        # functions apart: halt newsletter grading only — email
                        # triage keeps running (decision D5's scope rule, D19).
                        # The thread is left unprocessed, no strike, and is
                        # re-graded once the function resumes (D22: the third
                        # consecutive fault trips the slot; the poll loop's
                        # re-probe of the newsletter client clears it).
                        log.error("Newsletter thread %s deferred — %s", thread_id, exc)
                        if halts is not None:
                            halts.newsletter.record_balance_error(
                                exc, probe_client=newsletter_classifier.cloud_llm
                            )
                        return False
                    if halts is not None:
                        # The whole grading landed (every call answered — on_answer
                        # above already recorded each; this keeps the reset explicit
                        # at the call site).
                        halts.newsletter.record_success()

                    # Determine overall tier (best story's tier)
                    best_tier = None
                    for sr in story_results:
                        if sr.tier is not None:
                            if best_tier is None or _TIER_RANK.get(sr.tier, 0) > _TIER_RANK.get(best_tier, 0):
                                best_tier = sr.tier
                    # Merge graded themes across stories (strongest grade per theme)
                    all_themes = aggregate_theme_grades(story_results)

                    # Cache the grading before any write is attempted, so a sink
                    # or label fault costs a write retry, not an LLM re-run.
                    cached_nl = CachedNewsletterResult(
                        best_tier=best_tier,
                        all_themes=all_themes,
                        story_results=story_results,
                    )
                    if result_cache is not None:
                        result_cache.put(thread_id, fingerprint, cached_nl)

                # Persist the assessment BEFORE the labels commit. The JSONL record
                # is the only durable copy of the grading — Gmail keeps just the
                # coarse tier/theme labels, and apply_newsletter_classification also
                # adds agent/processed, which drops the thread out of gmail_query for
                # good. Writing afterwards (and swallowing the error) turned any sink
                # fault — a bind mount gone read-only, a full disk, bad permissions —
                # into permanent silent data loss: labels applied, grading gone,
                # thread never re-graded. Writing first means a sink fault leaves the
                # thread unprocessed, so the next cycle retries it — and keeps
                # retrying: a sink fault is shared-cause (the disk, not the thread),
                # never counted toward give-up (decision D5's sink corollary), so the
                # newsletter waits for the operator instead of being abandoned.
                # Skipped once assessment_written: within a daemon session an
                # unchanged fingerprint re-attempts only the labels. The cache is
                # session-scoped and pruned to the cycle's page, so a restart —
                # or a backlog that pushes the thread off a page — mid-retry
                # re-grades and re-appends; D18's newest-timestamp dedup on read
                # is the backstop for that record.
                if newsletter_output_file and not cached_nl.assessment_written:
                    try:
                        write_assessment(
                            output_file=newsletter_output_file,
                            message_id=all_msg_ids[0],
                            thread_id=thread_id,
                            sender=sender,
                            subject=subject,
                            overall_tier=best_tier,
                            stories=story_results,
                            send_date=send_date,
                            model=newsletter_classifier.cloud_llm.model,
                        )
                    except OSError as exc:
                        # Named at ERROR with the resolved path so a sink fault is
                        # distinguishable from a grading fault at a glance. This
                        # per-cycle line is the loudness the never-counted retry
                        # relies on (the startup preflight already screams about an
                        # unusable sink).
                        log.error(
                            "Cannot write newsletter assessment to %s: %s — thread %s "
                            "left unprocessed for retry (check the path exists, is "
                            "writable, and is volume-mounted)",
                            Path(newsletter_output_file).resolve(),
                            exc,
                            thread_id,
                        )
                        # Re-raise as a dedicated type, not the bare OSError: the
                        # arm below must be able to catch sink faults WITHOUT
                        # catching TimeoutError, which subclasses OSError and is a
                        # strike candidate (decision D5).
                        raise AssessmentSinkError(str(exc)) from exc
                    cached_nl.assessment_written = True

                await label_manager.apply_newsletter_classification(
                    message_ids=all_msg_ids,
                    tier=best_tier,
                    themes=all_themes,
                )
                if result_cache is not None:
                    result_cache.clear(thread_id)

                story_count = len(story_results)
                log.info(
                    "Newsletter thread %s: %d stories, tier=%s, themes=%s — %s",
                    thread_id,
                    story_count,
                    best_tier.value if best_tier else "no-stories",
                    all_themes,
                    subject,
                )
                return True

        # Newsletter-only mode: skip non-newsletter threads
        if newsletter_only:
            log.debug("Skipping non-newsletter thread %s (newsletter-only mode)", thread_id)
            return False

        # This thread belongs to the email function, and that function is halted
        # (its provider is out of funds): defer — no strike, no marker — while
        # newsletter grading carries on. Deliberately ABOVE the max-priority
        # branch below, so a halted email function commits nothing at all, not
        # even a mark_processed write. Also catches the sibling that tripped the
        # halt while this thread was fetching, before any LLM request.
        if halts is not None and halts.email.tripped:
            return False

        # Check priority — skip if already classified at max priority
        existing_priority = label_manager.get_existing_priority(messages)
        if existing_priority is not None and existing_priority >= _get_priority(EmailLabel.NEEDS_RESPONSE):
            # Already at max priority, so there's nothing to classify — but mark it
            # processed so it drops out of the unprocessed query. Otherwise the thread
            # has no agent/processed label and re-matches every cycle forever, costing
            # a full get_thread round-trip per thread per poll (same retry-loop reasoning
            # as the no-downgrade branch below). ids_to_mark was upgraded to all_msg_ids
            # above the priority check, so a failed write here gives up on the whole
            # thread, not just the query stubs.
            await label_manager.mark_processed(all_msg_ids)
            if result_cache is not None:
                result_cache.clear(thread_id)
            log.info("Thread %s already at max priority, marking processed", thread_id)
            return True

        first_headers = messages[0]["payload"]["headers"]
        subject = get_header(first_headers, "Subject")

        # Reuse a finished classification whose label write failed in a prior
        # cycle (issue #29): skip Stage 1/Stage 2 (the scarce local GPU pass
        # for person threads) and go straight to the label write. The
        # no-downgrade check below still runs, against the freshly fetched
        # thread's labels.
        result = None
        cached_email = result_cache.get(thread_id, fingerprint) if result_cache else None
        if cached_email is not None:
            log.info(
                "Thread %s: reusing cached classification from a prior cycle "
                "(write still pending)",
                thread_id,
            )
            label = cached_email.label
            applied_sender_type = cached_email.sender_type
        else:
            # Extract unique senders (preserve order)
            senders = []
            seen = set()
            for msg in messages:
                headers = msg["payload"]["headers"]
                sender = get_header(headers, "From")
                if sender and sender not in seen:
                    senders.append(sender)
                    seen.add(sender)

            if not senders:
                log.warning("Thread %s has no valid senders, skipping", thread_id)
                return False

            snippet = messages[-1].get("snippet", "")  # latest message snippet

            metadata = ThreadMetadata(
                thread_id=thread_id,
                senders=senders,
                subject=subject,
                snippet=snippet,
            )

            # Format thread transcript
            transcript = format_thread_transcript(messages, max_thread_chars)

            # Stage 1: classify sender (always cloud LLM)
            async with cloud_sem:
                sender_type, sender_raw, sender_cot = await classifier.classify_sender(metadata)
            if halts is not None and sender_raw != "VIP":
                # The cloud provider answered Stage 1: whatever Stage 2 does (the
                # local tier may be offline), the email function's provider is not
                # out of funds, so its consecutive-fault count restarts here (D22
                # item 2). The VIP short-circuit makes no LLM call and so says
                # nothing about the provider.
                halts.email.record_success()

            # Stage 2: classify email (routed by sender type)
            if sender_type == SenderType.PERSON:
                async with local_sem:
                    result = await classifier.classify(metadata, transcript, sender_type, sender_raw)
            else:
                async with cloud_sem:
                    result = await classifier.classify(metadata, transcript, sender_type, sender_raw)

            if halts is not None:
                # Stage 2 answered too. Stage 1 already recorded its answer above;
                # this covers the VIP path, whose only LLM call is Stage 2 (D22).
                halts.email.record_success()

            label = result.label
            applied_sender_type = result.sender_type
            # Cache before the write: a write fault must not discard the
            # finished classification (issue #29).
            if result_cache is not None:
                result_cache.put(
                    thread_id, fingerprint, CachedEmailResult(label, applied_sender_type)
                )

        # Enforce no-downgrade
        new_priority = _get_priority(label)
        if existing_priority is not None and existing_priority >= new_priority:
            log.info(
                "Thread %s: existing priority %d >= new %d, skipping downgrade",
                thread_id,
                existing_priority,
                new_priority,
            )
            # Still mark as processed so the thread isn't retried every cycle
            await label_manager.mark_processed(all_msg_ids)
            if result_cache is not None:
                result_cache.clear(thread_id)
            return True

        # Apply labels to ALL messages in thread. The proxy-write burst is
        # bounded inside LabelManager (issue #33): its write semaphore gates
        # each modify_message call, so a large max_emails_per_cycle can't
        # burst one concurrent write per message.
        await label_manager.apply_classification(all_msg_ids, label, applied_sender_type)
        if result_cache is not None:
            result_cache.clear(thread_id)
        log.info(
            "Classified thread %s (%d msgs): sender=%s label=%s — %s",
            thread_id,
            len(all_msg_ids),
            applied_sender_type.value,
            label.value,
            subject,
        )
        if result is not None:
            log.debug("Thread %s CoT — sender: %s", thread_id, result.sender_cot)
            log.debug("Thread %s CoT — label: %s", thread_id, result.label_cot)
        return True

    except AssessmentSinkError:
        # The assessments sink itself failed — a read-only bind mount, a full
        # disk, bad permissions. Shared cause (decision D5's sink corollary):
        # the disk is the problem, not this newsletter, so it never counts
        # toward give-up. No strike, no CycleFailure (nothing for the cycle
        # attribution to weigh), no marker — the thread stays pending and is
        # retried every cycle, forever, until the operator fixes the sink. The
        # ERROR line at the raise site (with the resolved path) already fired,
        # and with the ResultCache (issue #29) the retry re-attempts only the
        # JSONL write: the grading is cached, so "forever" costs no LLM spend.
        return False
    except LLMUnavailableError as exc:
        # LLM endpoint can't serve requests right now — unreachable, dropped
        # mid-request, or answering with an exhausted 429/5xx. Provider-shaped
        # (decision D5): never a strike; defer and retry next cycle (preserves
        # graceful degradation of the privacy invariant).
        #
        # The LOCAL tier being down is a routine operating condition (the MLX
        # laptop is deliberately offline for hours at a time), not an incident:
        # log per-thread detail at DEBUG and count the deferral so the poll loop
        # can emit a single per-cycle INFO summary instead of N warnings
        # (issue #24). Local-tier deferrals are also excluded from masquerade
        # bookkeeping entirely — person threads deferring while service siblings
        # succeed is the routine local state, and tracking it would false-alarm
        # every night the laptop is closed. A cloud (or tier-less) outage stays
        # a WARNING and is recorded provider-shaped so the cycle attribution can
        # spot the single-thread masquerade.
        if exc.tier == "local":
            log.debug("Local LLM unavailable processing thread %s: %s", thread_id, exc)
            if local_deferrals is not None:
                local_deferrals.append(thread_id)
        else:
            log.warning("LLM unavailable processing thread %s: %s", thread_id, exc)
            if cycle_failures is not None:
                cycle_failures.append(CycleFailure(
                    thread_id, ids_to_mark, type(exc).__qualname__, provider_shaped=True,
                ))
        return False
    except ProxyUnavailableError as exc:
        # api-proxy unavailable for THIS thread's call — connection refused, a timeout,
        # a dropped connection, a 5xx / exhausted-429 response, or a non-JSON 2xx body.
        # Provider-shaped (decision D5, deliberately reversing issue #26's give-up
        # counting): the proxy failing to serve a request is never the thread's
        # blame — no strike, defer and retry next cycle, backlog kept. The residual
        # #26 worried about — a deterministic per-thread 5xx masquerading as an
        # outage — is handled by the attribution step instead: provider-shaped
        # failures on one thread while siblings succeed accrue in the
        # MasqueradeTracker and escalate with a distinct repeated ERROR, retried
        # forever rather than abandoned.
        log.warning("api-proxy unavailable processing thread %s: %s", thread_id, exc)
        if cycle_failures is not None:
            cycle_failures.append(CycleFailure(
                thread_id, ids_to_mark, type(exc).__qualname__, provider_shaped=True,
            ))
        return False
    except ProxyForbiddenError as exc:
        # A proxy 403 on a gated write is a human answer — the operator said
        # "not now", or the op is blocked — not a failure (decision D6, issue #28
        # Option A): one clean line, no strike, no marker, no traceback. The thread
        # is re-offered next cycle; the ResultCache keeps the finished
        # classification, so the re-offer costs one write attempt, not an LLM
        # re-run. (A 403 during startup label verification is different — a blocked
        # op there is a config error, and verify_labels_with_retry lets it
        # propagate as permanent.)
        log.info(
            "Write rejected/blocked by the proxy for thread %s — "
            "re-offering next cycle: %s",
            thread_id, exc,
        )
        return False
    except httpx.ConnectError as exc:
        # Defensive: a raw ConnectError shouldn't escape the wrapped clients, but if
        # one does it's still a transient outage — provider-shaped (D5): retry next
        # cycle, never strike.
        log.warning("Connection error processing thread %s: %s", thread_id, exc)
        if cycle_failures is not None:
            cycle_failures.append(CycleFailure(
                thread_id, ids_to_mark, type(exc).__qualname__, provider_shaped=True,
            ))
        return False
    except TimeoutError as exc:
        # Request-specific slowness (e.g. a transcript too large to prefill within
        # the timeout) — a strike candidate: the cycle attribution counts it when
        # correlation blames the thread, so one huge thread can't be retried
        # forever. (Connect/pool timeouts are LLMUnavailableError, handled above.)
        log.error("Timeout processing thread %s: %s", thread_id, exc)
        if cycle_failures is not None:
            cycle_failures.append(CycleFailure(thread_id, ids_to_mark, type(exc).__qualname__))
        return False
    except LLMBalanceError as exc:
        # Account-wide, not a thread fault (and must precede the RuntimeError arm,
        # which it subclasses): don't count toward give-up, don't mark anything —
        # the thread is re-processed once the function resumes. Reaching this
        # arm means the fault came from the EMAIL pipeline's tiers (the
        # newsletter branch traps its own balance faults at the call site), so
        # it halts email triage only — newsletter grading keeps running
        # (decision D5's scope rule, D19). The balance_halt_strikes-th
        # consecutive fault trips the slot (D22); the client to re-probe is the
        # tier that raised — normally cloud; local only with a public stand-in
        # on that slot (D4), and even then a cloud client seen earlier in the
        # streak is preferred (DaemonHalt.record_balance_error).
        log.error("Thread %s deferred — %s", thread_id, exc)
        if halts is not None:
            probe_client = (
                classifier.local_llm if exc.tier == "local" else classifier.cloud_llm
            )
            halts.email.record_balance_error(exc, probe_client=probe_client)
        return False
    except RuntimeError as exc:
        # Request-specific LLM failure — a non-balance 4xx-shaped response, or an
        # unusable reply (LLMContentError): a strike candidate under the cycle
        # attribution (D5).
        log.error("Thread %s: %s", thread_id, exc)
        if cycle_failures is not None:
            cycle_failures.append(CycleFailure(thread_id, ids_to_mark, type(exc).__qualname__))
        return False
    except Exception as exc:
        # Unexpected failure — keep the traceback. A strike candidate under the
        # cycle attribution (D5): a poison thread converges to agent/attempted,
        # while a code bug failing every thread the same way correlates to
        # shared-cause (no strikes, loud, backlog kept).
        log.exception("Error processing thread %s", thread_id)
        if cycle_failures is not None:
            cycle_failures.append(CycleFailure(thread_id, ids_to_mark, type(exc).__qualname__))
        return False


def summarize_cycle(
    thread_items: list[tuple[str, list[str]]],
    results: list,
    failure_tracker: FailureTracker,
) -> tuple[int, list[str]]:
    """Tally a poll cycle's outcomes and update the failure tracker.

    A thread is "handled" when process_single_thread returned True (classified,
    or skipped at max priority); its failure count is cleared. Give-ups return
    False (they are failures, not handled work — decision D5), so given_up is
    NOT a subset of the handled count: the poll loop's marking step records
    them via record_give_up and clears their counts itself. Drains the
    per-cycle give-up list and prunes counts for threads no longer pending.
    Returns (handled_count, given_up_thread_ids).
    """
    processed = 0
    for (tid, _msg_ids), result in zip(thread_items, results):
        if result is True:
            processed += 1
            failure_tracker.clear(tid)  # success resets the failure count
    given_up = failure_tracker.take_given_up()
    failure_tracker.prune(tid for tid, _msg_ids in thread_items)
    return processed, given_up


def log_local_deferrals(deferred: list[str]) -> None:
    """One INFO line per cycle when person threads deferred on a local-LLM outage.

    The per-thread handler logs each deferral at DEBUG (issue #24: a closed
    laptop with N person emails used to emit N WARNINGs every cycle for hours);
    this summary is the single visible trace of a routine local outage.
    """
    if deferred:
        log.info(
            "Local LLM offline — deferred %d person email thread(s) this cycle",
            len(deferred),
        )


@dataclass
class IdleState:
    """Tracks the caught-up stretch between busy poll cycles (issue #58)."""

    idle_since: float | None = None
    last_heartbeat: float | None = None

    def reset(self) -> None:
        """Forget the current idle stretch (work arrived, or a cycle failed)."""
        self.idle_since = None
        self.last_heartbeat = None


def idle_report(had_work: bool, now: float, state: IdleState, status_interval: float) -> str | None:
    """One call per successful poll cycle. Mutates state; returns a line to log or None.

    Busy → idle logs a one-shot "caught up"; a continuing idle stretch logs a
    heartbeat every status_interval seconds so "healthy and caught up" stays
    distinguishable from "hung". Failed cycles reset the stretch (the except
    arms call state.reset()) so the heartbeat's minute count never includes
    outage time — "Still idle (Nm) — last poll ok" only measures healthy idling.
    """
    if had_work:
        state.reset()
        return None
    if state.idle_since is None:
        state.idle_since = now
        state.last_heartbeat = now
        return "Inbox caught up — nothing to process"
    if now - state.last_heartbeat >= status_interval:
        state.last_heartbeat = now
        return f"Still idle ({int((now - state.idle_since) / 60)}m) — last poll ok"
    return None


async def verify_labels_with_retry(
    label_manager: LabelManager,
    initial_backoff: int = 5,
    max_backoff: int = 60,
) -> list[str]:
    """Verify required Gmail labels, waiting out a transiently-unreachable proxy.

    The api-proxy may be slow, not yet up, or up-but-still-warming when the
    daemon starts. Two failure modes are transient and worth waiting out with
    capped exponential backoff, instead of letting the daemon exit and crash-loop
    under Docker:

      * a transport fault — connection refused, a connect/read timeout, a
        dropped connection; and
      * a proxy that is reachable but whose Gmail backend is still initializing,
        which answers 5xx (or 429).

    ``verify_labels`` reaches the proxy through ``_send``, which already wraps both
    of those into ``proxy_client.ProxyUnavailableError`` (a ``ProxyError`` subclass),
    so catching ``ProxyError`` covers them. The ``TRANSIENT_TRANSPORT_ERRORS`` prefix
    is kept as defence-in-depth in case a future code path reaches a raw transport
    fault here without going through ``_send``.

    Permanent failures propagate immediately so the operator sees an actionable
    error rather than a silent, endless retry: a misconfigured ``PROXY_URL``
    (``httpx.UnsupportedProtocol`` — itself a ``TransportError`` we deliberately
    do NOT catch), a bad key (``ProxyAuthError``), a blocked op
    (``ProxyForbiddenError``), or a programming error.

    Returns the list of missing label names (see LabelManager.verify_labels).
    """
    proxy_url = label_manager.proxy.proxy_url
    backoff = initial_backoff
    while True:
        try:
            return await label_manager.verify_labels()
        except TRANSIENT_TRANSPORT_ERRORS + (ProxyError,) as exc:
            log.warning(
                "Cannot reach api-proxy at %s (%s) — retrying in %ds",
                proxy_url,
                type(exc).__name__,
                backoff,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)


def preflight_assessment_sink(output_file: str) -> None:
    """Report — loudly, at startup — where newsletter assessments will land and
    whether that destination can actually keep them.

    Every way this sink fails is silent in normal operation, so each one gets a
    line the operator can act on before a single newsletter is graded:

      * the RESOLVED absolute path, since ``output_file`` is relative to the
        working directory (``/app`` in the image), plus how many records it
        already holds — a long-running daemon reporting 0 is appending
        somewhere other than the file being reviewed;
      * an ERROR when the path is not writable (a read-only mount) or is not a
        file at all, which now blocks labeling rather than being swallowed; and
      * an ERROR when nothing persists the path — in a container with no volume
        over it (or only a tmpfs), writes succeed and every record dies with the
        container.
    """
    path = Path(output_file).resolve()
    existing = count_records(path)
    if existing is None:
        # Not merely uninformative: the sink cannot be read, so nothing below can
        # vouch for it and write_assessment is very likely to fail the same way.
        log.error(
            "Newsletter assessments append to: %s — but that path cannot be read, so "
            "its existing records cannot be counted and writes are likely to fail too",
            path,
        )
    else:
        log.info("Newsletter assessments append to: %s (%d existing record(s))", path, existing)

    writability = sink_writability_warning(path)
    if writability:
        log.error("%s", writability)

    if running_in_container():
        # One mountinfo read AND one parse serve both the warning and the source
        # line below — they answer the same question about the same mount.
        mount = covering_mount(path, read_mountinfo())
        persistence = mount_persistence_warning(path, mount)
        if persistence:
            log.error("%s", persistence)
        elif mount:
            # A mount holds the sink, but only the operator knows whether it is
            # the directory they review: a volume pointed somewhere else fails
            # just as silently as no volume at all. Name the source so the answer
            # is one log line away instead of a docker inspect away. It is the
            # path within the source FILESYSTEM, so it lacks that filesystem's
            # own host mount point (a bind of /srv/stack/data reports
            # /stack/data when /srv is a separate filesystem) — say so rather
            # than claiming a host path that may not exist.
            log.info(
                "Assessments are persisted by the %s mount at %s (source %s, relative "
                "to that filesystem's root) — confirm that is the directory you review",
                mount[2],
                mount[0],
                mount[1],
            )


async def run_daemon() -> None:
    """Main polling loop."""
    # Release identity (decision D11): the Docker build stamps GIT_SHA into the
    # image; logging it here (not at import — daemon.py:64-77 contract) answers
    # "what is deployed?" from the logs alone.
    log.info("email-labeler starting — build %s", os.environ.get("GIT_SHA", "unknown"))
    config = load_config()
    daemon_config = config["daemon"]

    proxy_client = GmailProxyClient()
    cloud_llm = LLMClient(
        base_url=os.environ.get("CLOUD_LLM_URL", ""),
        api_key=os.environ.get("CLOUD_LLM_API_KEY", ""),
        model=config["llm"]["cloud"]["model"],
        max_tokens=config["llm"]["cloud"]["max_tokens"],
        temperature=config["llm"]["cloud"]["temperature"],
        timeout=config["llm"]["cloud"]["timeout"],
        extra_body=config["llm"]["cloud"].get("extra_body"),
        tier="cloud",
    )
    local_llm = LLMClient(
        base_url=os.environ.get("MLX_URL", ""),
        api_key=os.environ.get("MLX_API_KEY", ""),
        model=config["llm"]["local"]["model"],
        max_tokens=config["llm"]["local"]["max_tokens"],
        temperature=config["llm"]["local"]["temperature"],
        timeout=config["llm"]["local"]["timeout"],
        extra_body=config["llm"]["local"].get("extra_body"),
        tier="local",
    )

    classifier = EmailClassifier(
        cloud_llm=cloud_llm,
        local_llm=local_llm,
        config=config,
    )
    # The write semaphore is owned by the LabelManager (issue #33): each
    # modify_message write acquires one slot, so write_parallel bounds writes
    # in flight, not threads writing. Sizing rationale: config.toml [daemon].
    write_parallel = resolve_int_env("WRITE_PARALLEL", daemon_config.get("write_parallel", 4))
    write_sem = asyncio.Semaphore(write_parallel)
    label_manager = LabelManager(proxy_client=proxy_client, config=config, write_sem=write_sem)

    # Newsletter classifier (if configured)
    nl_config = config.get("newsletter")
    newsletter_classifier = None
    newsletter_recipient = ""
    newsletter_output_file = ""
    if nl_config:
        nl_llm_config = nl_config.get("llm")
        if nl_llm_config:
            nl_base_url, nl_api_key = resolve_newsletter_llm_endpoint()
            nl_llm = LLMClient(
                base_url=nl_base_url,
                api_key=nl_api_key,
                model=nl_llm_config["model"],
                max_tokens=nl_llm_config.get("max_tokens", 1024),
                temperature=nl_llm_config.get("temperature", 0),
                timeout=nl_llm_config.get("timeout", 60),
                extra_body=nl_llm_config.get("extra_body"),
                tier="cloud",
            )
        else:
            nl_llm = cloud_llm
        newsletter_classifier = NewsletterClassifier(cloud_llm=nl_llm, config=config)
        newsletter_recipient = nl_config["recipient"]
        newsletter_output_file = nl_config.get("output_file", "")
        log.info("Newsletter classification enabled for: %s", newsletter_recipient)
        if newsletter_output_file:
            preflight_assessment_sink(newsletter_output_file)
        else:
            # Grading with no sink is silent by construction: labels apply, the
            # per-thread summary logs, and nothing is ever recorded to review.
            log.error(
                "Newsletter classification is enabled but [newsletter] output_file is "
                "not set in config.toml — newsletters will be graded and labeled, but "
                "no assessment records will be written"
            )

    newsletter_only = os.environ.get("NEWSLETTER_ONLY", "").strip().lower() in ("1", "true", "yes")
    if newsletter_only:
        log.info("Newsletter-only mode: non-newsletter threads will be skipped")

    cloud_sem = asyncio.Semaphore(daemon_config.get("cloud_parallel", 2))
    local_parallel = resolve_int_env("LOCAL_PARALLEL", daemon_config.get("local_parallel", 1))
    local_sem = asyncio.Semaphore(local_parallel)
    fetch_sem = asyncio.Semaphore(daemon_config.get("fetch_parallel", 4))
    log.info(
        "Concurrency limits: cloud=%d, local=%d, fetch=%d, write=%d",
        cloud_sem._value, local_sem._value, fetch_sem._value, write_sem._value,
    )
    if local_parallel > 8:
        log.warning(
            "local_parallel=%d exceeds 8 — some MLX servers exhibit KV-cache "
            "cross-contamination at high concurrency (mlx-lm at 16+)",
            local_parallel,
        )

    # Breaks infinite retry loops: a thread the cycle-level attribution keeps
    # blaming (decision D5) is marked agent/attempted after a few strikes.
    # Session-scoped — counts reset on restart. Sizing rationale: config.toml
    # [daemon] max_failures (authoritative); override per run with MAX_FAILURES.
    failure_tracker = FailureTracker(
        max_failures=resolve_int_env("MAX_FAILURES", daemon_config.get("max_failures", 5))
    )

    # Watches the single-thread masquerade (provider-shaped failures on one
    # thread while siblings succeed, decision D5): never abandoned, escalated
    # with a distinct ERROR on the status heartbeat. Session-scoped.
    masquerade_tracker = MasqueradeTracker(max_failures=failure_tracker.max_failures)

    # Keeps finished classifications across cycles while their label writes
    # keep failing (issue #29): a write fault costs a write retry, never an
    # LLM re-run. Session-scoped, like the tracker.
    result_cache = ResultCache()

    # Account-level fault switches (provider out of funds), one per function
    # (decision D5's scope rule, D19): a tripped slot stops that function until
    # its re-probe gets an answer (D22); the poll loop stands down only once
    # every enabled function is halted. Session-scoped. Both knobs are homed in
    # config.toml [daemon] (authoritative, with rationale): balance_halt_strikes
    # (override BALANCE_HALT_STRIKES) and halt_probe_interval_seconds (override
    # HALT_PROBE_INTERVAL_SECONDS). They are validated HERE, at startup: a bad
    # value must not wait for the first halt to surface (review of PR #81).
    try:
        balance_halt_strikes = positive_int_setting(
            daemon_config, "balance_halt_strikes", DEFAULT_BALANCE_HALT_STRIKES
        )
        halt_probe_interval_default = positive_int_setting(
            daemon_config, "halt_probe_interval_seconds", 3600
        )
    except ValueError as exc:
        log.error("%s", exc)
        sys.exit(1)
    halts = FunctionHalts(
        email_enabled=not newsletter_only,
        newsletter_enabled=bool(newsletter_classifier and newsletter_recipient),
        strikes_to_trip=resolve_int_env("BALANCE_HALT_STRIKES", balance_halt_strikes),
    )
    halt_probe_interval = resolve_int_env(
        "HALT_PROBE_INTERVAL_SECONDS", halt_probe_interval_default
    )
    # Push on halt and on resume (D22). Disabled — one WARNING here, then
    # no-ops — unless NTFY_URL and NTFY_TOKEN are both set.
    notifier = HaltNotifier.from_env()

    # Wait for a transiently-unreachable api-proxy to come up, then verify labels.
    missing = await verify_labels_with_retry(label_manager)
    if missing:
        log.error("Missing Gmail labels: %s", missing)
        log.error("Create these labels manually in Gmail before running the daemon.")
        sys.exit(1)

    log.info("All labels verified. Starting poll loop.")

    poll_interval = daemon_config["poll_interval_seconds"]
    max_emails = resolve_int_env("MAX_EMAILS_PER_CYCLE", daemon_config["max_emails_per_cycle"])
    gmail_query = daemon_config["gmail_query"]
    if newsletter_only and newsletter_recipient:
        gmail_query += f" to:{newsletter_recipient}"
        log.info("Gmail query narrowed to: %s", gmail_query)
    # The query as configured (plus any NEWSLETTER_ONLY clause), restored when
    # an email-only halt that narrowed it resumes (D22).
    base_query = gmail_query
    narrowed_by_halt = False
    healthcheck_file = Path(daemon_config["healthcheck_file"])
    backoff = poll_interval
    status_interval = daemon_config.get("status_interval_seconds", 900)
    idle_state = IdleState()
    proxy_lost = False  # set by the lost-connection arm, cleared on the next good poll

    while True:
        # Halted functions re-probe their provider on the slow schedule and
        # clear themselves when it answers (D22). Runs before the stand-down
        # check so a resumed function polls in this very cycle.
        now = time.monotonic()
        await notify_new_halts(halts, notifier, halt_probe_interval, now)
        resumed = await reprobe_halts(halts, now, halt_probe_interval)
        await notify_resumes(notifier, resumed)
        if narrowed_by_halt and not halts.email.tripped:
            # Email triage resumed: its backlog must be fetched again.
            gmail_query = base_query
            narrowed_by_halt = False
            log.info("Email triage resumed — Gmail query restored: %s", gmail_query)
        if halts.all_halted:
            # Every enabled function's provider is out of funds, and such a fault
            # fails EVERY request — polling on would only burn the backlog into
            # agent/attempted. Stand down but stay alive: the heartbeat stays
            # fresh (deliberately halted, not hung), and the instruction repeats
            # at ERROR every cycle so it can't scroll out of the logs. The
            # re-probe above is the reset (a restart also clears the in-memory
            # state, but is no longer required — D22).
            log.error(
                "Daemon halted — every enabled function stopped (%s). Add funds if "
                "the provider account is out of them; the daemon re-probes the "
                "provider every %ds and resumes on its own once it answers.",
                halts.halted_summary(), halt_probe_interval,
            )
            try:
                healthcheck_file.write_text(str(asyncio.get_event_loop().time()))
            except OSError as exc:
                # This branch sits outside the loop's try/except; a transient
                # filesystem fault must not kill the daemon — the recurring
                # instruction above is the whole point of the halt state.
                log.warning("Failed to update healthcheck while halted: %s", exc)
            await asyncio.sleep(poll_interval)
            continue
        if halts.any_halted:
            # A PARTIAL halt: one function is stopped, the other still has work
            # to do, so the loop keeps polling. The halted function must not go
            # quiet about it — same repeated-ERROR discipline as the full
            # stand-down, naming which function needs the funds.
            log.error(
                "Function halted — %s. Add funds if the provider account is out of "
                "them; the daemon re-probes the provider every %ds and resumes on its "
                "own once it answers; the other function keeps running.",
                halts.halted_summary(), halt_probe_interval,
            )
            if halts.email_only_halted and not narrowed_by_halt:
                # Email triage is stopped but newsletter grading is not: narrow
                # the query the way NEWSLETTER_ONLY does, so the halted
                # function's backlog stops costing a get_thread per thread per
                # cycle and can't crowd newsletter threads out of the
                # max_results page. Holds until the email function resumes,
                # when the block at the top of the loop restores base_query.
                gmail_query = f"{base_query} to:{newsletter_recipient}"
                narrowed_by_halt = True
                log.info(
                    "Email triage halted — Gmail query narrowed to the newsletter "
                    "function: %s",
                    gmail_query,
                )
        try:
            response = await proxy_client.list_messages(q=gmail_query, max_results=max_emails)
            messages = response.get("messages", [])

            if proxy_lost:
                # Logged before the cycle's work so the narrative reads
                # lost → reconnected → found/processed (issue #58).
                log.info("Reconnected to api-proxy — resuming normal polling")
                proxy_lost = False

            if messages:
                log.info("Found %d unprocessed message(s)", len(messages))

            # Group messages by threadId
            threads: dict[str, list[str]] = {}
            for msg_stub in messages:
                tid = msg_stub.get("threadId", msg_stub["id"])
                threads.setdefault(tid, []).append(msg_stub["id"])

            if threads:
                log.info("Grouped into %d thread(s)", len(threads))

            max_thread_chars = daemon_config.get("max_thread_chars", DEFAULT_MAX_THREAD_CHARS)
            thread_items = list(threads.items())
            local_deferrals: list[str] = []
            cycle_failures: list[CycleFailure] = []
            results = await asyncio.gather(
                *(
                    process_single_thread(
                        tid,
                        msg_ids,
                        proxy_client,
                        classifier,
                        label_manager,
                        cloud_sem,
                        local_sem,
                        max_thread_chars,
                        newsletter_classifier=newsletter_classifier,
                        newsletter_recipient=newsletter_recipient,
                        newsletter_output_file=newsletter_output_file,
                        newsletter_only=newsletter_only,
                        cycle_failures=cycle_failures,
                        fetch_sem=fetch_sem,
                        local_deferrals=local_deferrals,
                        halts=halts,
                        result_cache=result_cache,
                    )
                    for tid, msg_ids in thread_items
                ),
                return_exceptions=True,
            )
            # Post-gather failure handling, strictly ordered (decision D5):
            # attribute → strike → mark → summarize → cycle log.
            struck_out = attribute_cycle_failures(
                thread_items, results, cycle_failures, failure_tracker, masquerade_tracker,
            )
            for entry in struck_out:
                await _mark_thread_attempted(
                    entry.thread_id, entry.ids_to_mark, failure_tracker, label_manager,
                )
            processed, given_up = summarize_cycle(thread_items, results, failure_tracker)
            result_cache.prune(tid for tid, _msg_ids in thread_items)
            log_local_deferrals(local_deferrals)
            if threads:
                if given_up:
                    log.info(
                        "Processed %d/%d threads (%d abandoned after repeated failures: %s)",
                        processed, len(threads), len(given_up), given_up,
                    )
                else:
                    log.info("Processed %d/%d threads", processed, len(threads))
            escalation = masquerade_tracker.escalation_line(
                asyncio.get_event_loop().time(), status_interval
            )
            if escalation:
                log.error(escalation)

            # Update healthcheck
            healthcheck_file.write_text(str(asyncio.get_event_loop().time()))

            # Reset backoff on success
            backoff = poll_interval

            line = idle_report(
                bool(messages), asyncio.get_event_loop().time(), idle_state, status_interval
            )
            if line:
                log.info(line)

        except TRANSIENT_TRANSPORT_ERRORS + (ProxyUnavailableError,) as exc:
            # A transiently-unreachable proxy (raw transport fault, or a wrapped
            # ProxyUnavailableError — connection/timeout/5xx — from list_messages):
            # back off and retry rather than logging a full traceback.
            log.warning(
                "Lost connection to api-proxy at %s (%s) — retrying in %ds",
                proxy_client.proxy_url,
                type(exc).__name__,
                backoff,
            )
            proxy_lost = True
            backoff = min(backoff * 2, poll_interval * 10)
            # A failed cycle breaks the idle stretch: the next quiet poll logs
            # "caught up" afresh, so heartbeat minutes never include downtime.
            idle_state.reset()
        except ProxyError as exc:
            # A request-specific proxy fault from the cycle-level list_messages call:
            # a 4xx, e.g. a malformed gmail_query 400. It won't fix itself, but it's a
            # known, named condition: log it as a warning and back off rather than
            # spewing a full traceback every cycle. (The transient faults — 5xx, an
            # exhausted 429, a non-JSON 2xx body — are ProxyUnavailableError, handled
            # by the arm above.)
            log.warning(
                "api-proxy rejected the poll request (%s: %s) — retrying in %ds",
                type(exc).__name__, exc, backoff,
            )
            backoff = min(backoff * 2, poll_interval * 10)
            idle_state.reset()
        except Exception:
            log.exception("Error in poll cycle")
            backoff = min(backoff * 2, poll_interval * 10)
            idle_state.reset()

        await asyncio.sleep(backoff)


def main():
    """Entry point."""
    quiet_http_logging()
    asyncio.run(run_daemon())


if __name__ == "__main__":
    main()
