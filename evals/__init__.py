"""Classification evaluation suite for the email labeler."""

import json
import os
import stat
import tempfile
from pathlib import Path

import httpx

from proxy_client import ProxyAuthError, ProxyError, ProxyForbiddenError


def atomic_write_jsonl(records, path) -> None:
    """Write *records* to *path* as JSONL atomically (temp file + rename).

    Each record must expose ``.to_dict()``. The temp file lives in the target's
    directory so the rename is atomic on the same filesystem; on any error the
    temp file is cleaned up and the original *path* is left untouched. Shared by
    the golden-set editors (issue #50)."""
    path = Path(path)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".jsonl.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            for record in records:
                f.write(json.dumps(record.to_dict()) + "\n")
        # mkstemp creates the temp file 0600. Carry the real file's permissions
        # across the rename (or, for a new file, what a plain open() would
        # produce) so a save never narrows who can read the file.
        os.chmod(tmp_path, _target_mode(path))
        os.rename(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _target_mode(path: Path) -> int:
    """Permission bits the atomically written file should end up with.

    The existing file's mode if there is one; otherwise 0666 masked by the
    process umask, i.e. what ``open(path, "w")`` would have created."""
    try:
        return stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        current_umask = os.umask(0)
        os.umask(current_umask)
        return 0o666 & ~current_umask


def plural(n: int, singular: str, plural_form: str | None = None) -> str:
    """'1 story' / 'N stories' — the one pluralizer for eval status lines.

    Defaults the plural form to ``singular + "s"``; pass *plural_form* for
    irregulars ("story"/"stories").
    """
    if n == 1:
        return f"1 {singular}"
    return f"{n} {plural_form or singular + 's'}"


def format_network_error(exc: Exception, service: str = "service") -> str:
    """Format a network exception into a human-readable error message.

    Args:
        exc: The caught exception.
        service: Name of the service being contacted (e.g. "api-proxy", "cloud LLM").

    Returns:
        A one-line error message suitable for printing to stderr.
    """
    if isinstance(exc, httpx.ConnectError):
        msg = str(exc)
        if "Name or service not known" in msg or "getaddrinfo failed" in msg:
            return f"DNS lookup failed for {service} — is the hostname correct? ({msg})"
        if "Connection refused" in msg:
            return f"Connection refused by {service} — is the service running? ({msg})"
        return f"Could not connect to {service}: {msg}"
    if isinstance(exc, httpx.TimeoutException):
        return f"Request to {service} timed out: {exc}"
    if isinstance(exc, ProxyAuthError):
        return f"Authentication failed with {service}: {exc}"
    if isinstance(exc, ProxyForbiddenError):
        return f"Request forbidden by {service}: {exc}"
    if isinstance(exc, ProxyError):
        return f"Error from {service}: {exc}"
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code} from {service}: {exc}"
    return f"Unexpected error contacting {service}: {exc}"
