"""Push notifications for daemon halts and resumes, via ntfy (decision D22, issue #73).

A halt is exactly the condition the operator needs to hear about before next
sitting down — the 2026-08-25 halt went unnoticed for fourteen days because the
daemon only logged it. Configuration is two env vars: ``NTFY_URL`` (the full
topic URL) and ``NTFY_TOKEN`` (a bearer token minted for the labeler — not
shared with any other service). With either unset the notifier is disabled:
one WARNING at startup, and every ``send`` is a no-op.

Contract: ``send`` never raises. A notification failure is logged and the
daemon carries on — the push is a courtesy on top of the halt, never a reason
to fail it. Message bodies carry provider identity, HTTP status, the provider's
reason text (capped) and times — no email content and no credentials.
"""

import logging
import os
from datetime import datetime

import httpx

from llm_client import LLMBalanceError

log = logging.getLogger("email-labeler")

# Seconds to wait on the ntfy POST. Short: this runs on the poll loop's own
# task, and a slow notification must not hold up the next cycle.
NOTIFY_TIMEOUT = 10.0
# Longest provider reason text carried into a push body; the full text is in
# the daemon log already.
DETAIL_CAP = 300

DISABLED_WARNING = "halt notifications disabled: NTFY_URL/NTFY_TOKEN not set"


def _format_wall(ts: float | None) -> str:
    if ts is None:
        return "unknown"
    return datetime.fromtimestamp(ts).astimezone().strftime("%Y-%m-%d %H:%M %Z")


def format_downtime(seconds: float) -> str:
    """`Xh Ym` / `Ym` for a halt's duration — the resume line and push."""
    minutes = int(seconds // 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m"


def halt_message(
    function: str,
    fault: LLMBalanceError | None,
    *,
    tripped_wall: float | None,
    probe_interval: int,
) -> tuple[str, str]:
    """(title, body) for the halt push: what stopped, which provider said what, when."""
    title = f"email-labeler halted: {function}"
    lines = [f"{function} stopped after repeated out-of-funds responses from its LLM provider."]
    if fault is not None:
        provider = f"tier={fault.tier or '-'} model={fault.model or '-'}"
        status = f"HTTP {fault.status_code}" if fault.status_code is not None else "no HTTP status"
        lines.append(f"Provider: {provider} ({status}).")
        if fault.detail:
            lines.append(f"Provider said: {fault.detail[:DETAIL_CAP]}")
    lines.append(f"Halted at: {_format_wall(tripped_wall)}.")
    lines.append(
        f"The daemon re-probes the provider every {probe_interval}s and resumes on its "
        f"own once it answers; if the account really is empty, add funds."
    )
    return title, "\n".join(lines)


def resume_message(function: str, downtime_seconds: float) -> tuple[str, str]:
    """(title, body) for the resume push: what came back and how long it was down."""
    title = f"email-labeler resumed: {function}"
    body = (
        f"{function} resumed — the provider answered the re-probe after "
        f"{format_downtime(downtime_seconds)} halted. Normal processing has resumed."
    )
    return title, body


class HaltNotifier:
    """Sends ntfy pushes; disabled (no-op) when the URL or token is missing."""

    def __init__(self, url: str | None, token: str | None, *, timeout: float = NOTIFY_TIMEOUT):
        self.url = (url or "").strip() or None
        self._token = (token or "").strip() or None
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return self.url is not None and self._token is not None

    @classmethod
    def from_env(cls) -> "HaltNotifier":
        """Build from NTFY_URL / NTFY_TOKEN; one WARNING when disabled, no values echoed."""
        notifier = cls(os.environ.get("NTFY_URL"), os.environ.get("NTFY_TOKEN"))
        if notifier.enabled:
            log.info("Halt notifications enabled (ntfy)")
        else:
            log.warning(DISABLED_WARNING)
        return notifier

    async def send(self, title: str, body: str, *, priority: str = "high") -> bool:
        """POST one push. Returns True on a 2xx; logs and returns False otherwise.
        Never raises — the daemon loop calls this outside its try/except."""
        if not self.enabled:
            return False
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Title": title,
            "Priority": priority,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.url, headers=headers, content=body)
        except Exception as exc:  # noqa: BLE001 — by contract, nothing escapes
            log.warning(
                "Halt notification not sent (%s: %s) — %r", type(exc).__name__, exc, title
            )
            return False
        if not 200 <= response.status_code < 300:
            log.warning(
                "Halt notification rejected — HTTP %d from ntfy — %r", response.status_code, title
            )
            return False
        return True
