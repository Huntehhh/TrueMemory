"""
Auth-Failure Recovery (opt-in)
==============================

Generic, opt-in recovery for authentication failures in the background
fact-extraction path.

When a one-shot LLM CLI backend (e.g. an OAuth-authenticated ``claude -p``
call) fails because its credential is dead/expired, the ingest pipeline
raises :class:`~truememory.ingest.models.LLMAuthError`. The ingest CLI then
re-queues the session to the backlog so its facts aren't lost. Separately,
the SessionStart drain can run a *user-configured* re-auth command once per
drain to fix the credential, so the re-queued session succeeds on the next
attempt.

This module is deliberately vendor-neutral: the re-auth command is whatever
the user configures. TrueMemory never assumes which CLI or auth flow is in
use, and ships with no command by default (so this is strictly opt-in).

Configuration (first match wins):
    1. ``~/.truememory/config.json`` key ``on_auth_failure_cmd`` (string)
    2. ``TRUEMEMORY_ON_AUTH_FAILURE_CMD`` environment variable

If neither is set, recovery is a no-op.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# How many times a single session may be re-queued through the auth-recovery
# path before we give up and stop re-running the re-auth command for it. This
# prevents an infinite re-auth ↔ re-ingest loop when the credential can't be
# repaired automatically (e.g. the command needs an interactive browser the
# user never completes).
MAX_AUTH_RETRIES = 2

_CONFIG_PATH = Path.home() / ".truememory" / "config.json"

# Bound the re-auth command so a hung login flow can't wedge the drain.
_DEFAULT_RECOVERY_TIMEOUT = 180


def get_on_auth_failure_cmd() -> str | None:
    """Return the user-configured re-auth command, or ``None`` if unset.

    Resolution order: ``~/.truememory/config.json`` (``on_auth_failure_cmd``)
    first, then the ``TRUEMEMORY_ON_AUTH_FAILURE_CMD`` environment variable.
    Matches the config-load pattern used elsewhere (``cli._load_truememory_config``,
    ``hooks.core._get_current_tier``): a missing or corrupt config falls back
    to the env var and never raises.
    """
    try:
        if _CONFIG_PATH.exists():
            config = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(config, dict):
                cmd = config.get("on_auth_failure_cmd")
                if isinstance(cmd, str) and cmd.strip():
                    return cmd.strip()
    except (json.JSONDecodeError, OSError, ValueError):
        # Corrupt/unreadable config — fall through to the env var.
        pass

    env_cmd = os.environ.get("TRUEMEMORY_ON_AUTH_FAILURE_CMD", "")
    if env_cmd.strip():
        return env_cmd.strip()

    return None


def run_auth_recovery(timeout: int = _DEFAULT_RECOVERY_TIMEOUT) -> bool:
    """Run the user-configured re-auth command, bounded by ``timeout`` seconds.

    Returns ``True`` only if a command is configured AND it exits 0. Returns
    ``False`` for the no-op case (no command configured), a non-zero exit, a
    timeout, or any unexpected error.

    This function NEVER raises — it is called from hook/drain code paths where
    an exception would break session startup or lose the backlog. Everything
    is caught and logged.

    The command is run as a shell string (``shell=True``) so users can write
    natural one-liners. Its stdout/stderr are captured (not inherited) so a
    chatty login flow doesn't pollute hook output or Claude Code's stdio.
    """
    cmd = get_on_auth_failure_cmd()
    if not cmd:
        log.debug("auth recovery: no on_auth_failure_cmd configured, skipping")
        return False

    log.info("auth recovery: running configured re-auth command (timeout=%ds)", timeout)
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        log.warning("auth recovery: re-auth command timed out after %ds", timeout)
        return False
    except Exception as e:  # noqa: BLE001 — must never raise to the caller
        log.warning("auth recovery: re-auth command failed to launch: %s", e)
        return False

    if proc.returncode == 0:
        log.info("auth recovery: re-auth command succeeded (exit 0)")
        return True

    stderr = (proc.stderr or "").strip()[:500]
    log.warning(
        "auth recovery: re-auth command exited %d: %s",
        proc.returncode, stderr or "no stderr",
    )
    return False


def spawn_auth_recovery(timeout: int = _DEFAULT_RECOVERY_TIMEOUT) -> bool:
    """Fire-and-forget the re-auth command in a DETACHED, bounded subprocess.

    Returns ``True`` if a recovery process was launched, ``False`` if no
    command is configured or the launch failed. Unlike :func:`run_auth_recovery`,
    this does NOT wait for the command to finish — it is used by the
    SessionStart drain so re-authentication (which may pop a browser) never
    blocks session startup synchronously.

    Detachment mirrors the Stop hook's ``_run_background_ingestion`` pattern:
    POSIX uses ``start_new_session``; Windows uses
    ``CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS``. The bounded timeout is
    enforced by wrapping the user command with a watchdog Python one-liner so
    the detached process can't hang forever even though we don't wait on it.

    Never raises.
    """
    cmd = get_on_auth_failure_cmd()
    if not cmd:
        log.debug("auth recovery: no on_auth_failure_cmd configured, not spawning")
        return False

    detach_kwargs: dict = {}
    if sys.platform == "win32":
        detach_kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        detach_kwargs["start_new_session"] = True

    # Run the user's shell command under a Python watchdog so the detached
    # process is still bounded by ``timeout`` even though we don't wait on it.
    # We pass the command via argv (not string-interpolated into source) to
    # avoid any quoting/escaping issues with the user's command.
    watchdog = (
        "import subprocess,sys;"
        "sys.exit(subprocess.run(sys.argv[1],shell=True,timeout=int(sys.argv[2])).returncode)"
    )
    spawn_cmd = [sys.executable, "-c", watchdog, cmd, str(timeout)]

    try:
        subprocess.Popen(
            spawn_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            close_fds=(sys.platform != "win32"),
            **detach_kwargs,
        )
        log.info("auth recovery: spawned detached re-auth command (timeout=%ds)", timeout)
        return True
    except Exception as e:  # noqa: BLE001 — must never raise to the caller
        log.warning("auth recovery: failed to spawn detached re-auth command: %s", e)
        return False
