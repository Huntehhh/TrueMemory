"""Lifecycle hardening for the TrueMemory MCP server.

Solves the zombie-MCP problem: when Claude Code closes ungracefully
(crash, force-quit, system reboot), the ``truememory.mcp_server`` child
process is not always reaped, leaving an orphan that holds a SQLite
read lock on ``~/.truememory/memories.db``. Stacking enough orphans
triggers 60s search timeouts and stale-cache pathologies.

Two complementary mechanisms cover the common failure modes:

    1. ``sweep_orphan_siblings()`` — at server startup, find any OTHER
       MCP server processes whose parent PID is dead and kill them.
       Catches every zombie the next time Claude Code starts.

    2. ``start_parent_watchdog()`` — daemon thread that polls our own
       parent PID every N seconds. If the parent dies, we exit hard
       (``os._exit``, bypassing atexit). Catches the case where the
       server is running but Claude Code went down.

This module is also runnable directly for ad-hoc cleanup::

    # See what would be swept (no kills):
    python -m truememory._lifecycle sweep --dry-run

    # Actually sweep:
    python -m truememory._lifecycle sweep

A third option — Windows Job Objects with
``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` — would be the gold-standard
fix, but it requires the PARENT (Claude Code) to wrap the spawn in a
job object, and we don't own Claude Code's process-spawn code. So we
work from the child side with the two mechanisms above.

PID-recycling caveat (Windows): when a process dies its PID can be
reassigned. ``psutil.pid_exists(ppid)`` can therefore return True even
after our actual parent is gone, if some other process happened to
land on the same PID. We mitigate by comparing the candidate parent's
``create_time`` against the child's (a real parent must be older).
False positives mean the watchdog briefly fails to fire, which is the
safe failure mode: the next server restart cleans up via
``sweep_orphan_siblings()`` anyway.

Module is named ``_lifecycle`` (leading underscore) to mark it as
internal — public API surface for this hardening is the two helpers
imported from ``truememory.mcp_server.main()``.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Iterable

# We deliberately do not raise from any function in this module — the
# MCP server must boot regardless of whether the cleanup helpers can
# run. Every entry point swallows exceptions and continues.

# Canonical invocation forms we must detect:
#   * ``python -m truememory.mcp_server``  (module form, joined cmdline
#     contains ``truememory.mcp_server``)
#   * ``truememory-mcp`` / ``truememory-mcp.exe`` (entry-point shim
#     installed by pyproject.toml [project.scripts])
#
# Note: ``setproctitle.setproctitle("TrueMemory MCP")`` is a no-op on
# Windows for ``psutil.Process.cmdline()`` — the cmdline still reflects
# the original argv. On POSIX it overwrites argv[0] visible in ``ps``,
# but ``psutil.cmdline()`` is read from ``/proc/<pid>/cmdline`` and may
# or may not reflect the new title depending on platform. We match the
# canonical module/script forms above, which are reliable across both.
_TOKEN_MODULE = "truememory.mcp_server"
_TOKEN_SCRIPT = "truememory-mcp"
_DEFAULT_WATCHDOG_INTERVAL_S = 30.0


def _is_truememory_mcp_server(proc) -> bool:
    """Return True iff ``proc`` is a python process running the MCP server.

    We accept the server running under any python.exe path (venv,
    pyenv, system python, etc.). Match is on the joined command-line
    containing either ``truememory.mcp_server`` (module form) or
    ``truememory-mcp`` (entry-point shim form).
    """
    try:
        cmdline = proc.info.get("cmdline") if hasattr(proc, "info") else proc.cmdline()
    except Exception:
        try:
            cmdline = proc.cmdline()
        except Exception:
            return False
    if not cmdline:
        return False
    try:
        joined = " ".join(str(p) for p in cmdline).lower()
    except Exception:
        return False
    return _TOKEN_MODULE in joined or _TOKEN_SCRIPT in joined


def _parent_seems_alive(child_proc, child_create_time: float | None = None) -> bool:
    """Best-effort check whether ``child_proc``'s parent is its REAL parent.

    Returns False if the parent PID does not exist, or if the parent
    process exists but was created AFTER the child (which means the
    PID was recycled and the original parent is gone).

    Failure mode is fail-open: if we cannot read parent metadata for
    any reason, we report the parent as alive so we never kill a
    process whose lineage we're uncertain about.
    """
    try:
        import psutil
    except ImportError:
        return True  # fail open — never kill if we can't verify

    try:
        ppid = child_proc.info.get("ppid") if hasattr(child_proc, "info") else child_proc.ppid()
    except Exception:
        return True

    if not ppid:
        return False  # no parent recorded = orphan
    try:
        if not psutil.pid_exists(ppid):
            return False
    except Exception:
        return True

    # PID-recycling guard: a real parent's ``create_time`` must be
    # <= the child's. If the parent's ``create_time`` is greater, the
    # original parent is gone and the PID was reassigned.
    try:
        parent = psutil.Process(ppid)
        parent_ct = parent.create_time()
        if child_create_time is None:
            child_create_time = child_proc.create_time()
        if parent_ct > child_create_time:
            return False  # PID was recycled; original parent is gone
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    except Exception:
        # If we can't read the parent's create_time, fail open.
        pass

    return True


def sweep_orphan_siblings(dry_run: bool = False) -> list[int]:
    """Find and kill orphaned sibling MCP server processes.

    A sibling is any OTHER python process whose command line contains
    ``truememory.mcp_server`` (or the ``truememory-mcp`` entry-point
    shim). An orphan is a sibling whose parent PID is dead (or was
    recycled — see PID-recycling caveat in the module docstring).

    The current process is always spared, even if its own parent
    looks dead (we are after all the one running). That case is
    handled by ``start_parent_watchdog`` instead.

    Returns the list of PIDs killed (or the list that WOULD be killed
    when ``dry_run=True``). Never raises — every failure mode is
    swallowed so the MCP server can boot even if psutil is missing or
    ``process_iter`` errors.
    """
    try:
        import psutil
    except ImportError:
        return []

    try:
        my_pid = os.getpid()
    except Exception:
        return []

    swept: list[int] = []
    try:
        proc_iter = psutil.process_iter(["pid", "ppid", "cmdline", "create_time"])
    except Exception:
        return swept

    for proc in proc_iter:
        try:
            if proc.pid == my_pid:
                continue
            if not _is_truememory_mcp_server(proc):
                continue
            try:
                child_ct = proc.info.get("create_time") if hasattr(proc, "info") else None
                if child_ct is None:
                    child_ct = proc.create_time()
            except Exception:
                child_ct = None
            if _parent_seems_alive(proc, child_ct):
                continue
            swept.append(proc.pid)
            if not dry_run:
                try:
                    proc.kill()
                    try:
                        proc.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        pass
                except psutil.NoSuchProcess:
                    # Already gone between detection and kill — count it.
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception:
            # Defensive: any unexpected psutil hiccup must not break boot.
            continue
    return swept


def start_parent_watchdog(
    poll_interval_s: float = _DEFAULT_WATCHDOG_INTERVAL_S,
) -> threading.Thread | None:
    """Spawn a daemon thread that exits us hard when our parent dies.

    Uses ``os._exit`` so we skip atexit handlers — they can hang
    waiting on the dead parent's STDIO pipes, which defeats the whole
    point. The exit code is 0 because this is an expected lifecycle
    event (parent gone), not a crash.

    Returns the started ``Thread``, or ``None`` if psutil is
    unavailable. The thread is ``daemon=True`` so it does not delay
    normal MCP shutdown when the parent is alive and asks us to exit.
    """
    try:
        import psutil
    except ImportError:
        return None

    my_pid = os.getpid()
    my_ppid = os.getppid()

    try:
        my_create_time = psutil.Process(my_pid).create_time()
    except Exception:
        my_create_time = None

    def _watch() -> None:
        try:
            import psutil  # local import inside thread is fine
        except ImportError:
            return
        while True:
            try:
                time.sleep(poll_interval_s)
            except Exception:
                # Should never happen, but if sleep raises we exit the loop.
                return
            try:
                if not psutil.pid_exists(my_ppid):
                    _exit_with_message(my_ppid, "PID does not exist")
                    return
                # PID-recycling guard.
                if my_create_time is not None:
                    try:
                        parent_ct = psutil.Process(my_ppid).create_time()
                        if parent_ct > my_create_time:
                            _exit_with_message(my_ppid, "PID was recycled")
                            return
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        _exit_with_message(my_ppid, "parent unreadable")
                        return
            except Exception:
                # Any psutil hiccup: wait for next tick.
                continue

    t = threading.Thread(target=_watch, daemon=True, name="truememory-ppid-watchdog")
    t.start()
    return t


def _exit_with_message(ppid: int, reason: str) -> None:
    try:
        sys.stderr.write(
            f"truememory.mcp_server: parent PID {ppid} gone ({reason}), exiting.\n"
        )
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(0)


# ---- CLI ----------------------------------------------------------------


def _cli(argv: Iterable[str]) -> int:
    args = list(argv)
    if not args or args[0] in ("-h", "--help", "help"):
        sys.stdout.write(
            "usage: python -m truememory._lifecycle sweep [--dry-run]\n"
            "\n"
            "Find and kill orphaned truememory.mcp_server processes\n"
            "whose parent Claude Code is gone.\n"
        )
        return 0
    if args[0] != "sweep":
        sys.stderr.write(f"unknown command: {args[0]!r}\n")
        return 2
    dry_run = "--dry-run" in args[1:]
    swept = sweep_orphan_siblings(dry_run=dry_run)
    label = "would kill" if dry_run else "killed"
    if not swept:
        sys.stdout.write("no orphaned MCP servers found.\n")
    else:
        sys.stdout.write(f"{label} {len(swept)} orphaned MCP server(s): {swept}\n")
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
