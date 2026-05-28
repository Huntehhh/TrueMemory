"""Regression locks for the Windows orphan-launcher reaper.

When Claude Code dies ungracefully, the ``truememory.mcp_server`` child
can survive as a zombie holding the SQLite read lock. The hardening in
``truememory._lifecycle`` covers it with two mechanisms:

    1. ``sweep_orphan_siblings()`` — kill other MCP processes whose
       parent PID is dead, on every server boot.
    2. ``start_parent_watchdog()`` — daemon thread that exits us when
       our parent dies.

These tests lock in the unit-level behaviour of (1) and the
end-to-end behaviour of (2) on Windows (POSIX has different reaping
semantics so the integration test is gated on ``sys.platform``).
"""
from __future__ import annotations

import os
import sys
import time

import pytest


# ---------------------------------------------------------------------------
# Helpers — psutil mocks
# ---------------------------------------------------------------------------


class _FakeProc:
    """Minimal psutil.Process stand-in.

    Mirrors the subset of the psutil.Process API that
    ``sweep_orphan_siblings`` exercises: ``pid``, ``info``, ``kill``,
    ``wait``, ``create_time``, ``ppid``, ``cmdline``.

    ``info`` is the dict returned by ``process_iter([...])`` and is the
    fast-path psutil reads. The instance methods are the slow-path
    fallback when ``info`` is missing a key.
    """

    def __init__(
        self,
        pid: int,
        ppid: int,
        cmdline: list[str],
        create_time: float,
        *,
        no_such_on_kill: bool = False,
    ) -> None:
        self.pid = pid
        self.info = {
            "pid": pid,
            "ppid": ppid,
            "cmdline": cmdline,
            "create_time": create_time,
        }
        self._ppid = ppid
        self._cmdline = cmdline
        self._create_time = create_time
        self._no_such_on_kill = no_such_on_kill
        self.killed = False
        self.waited = False

    def ppid(self):
        return self._ppid

    def cmdline(self):
        return self._cmdline

    def create_time(self):
        return self._create_time

    def kill(self):
        import psutil
        if self._no_such_on_kill:
            raise psutil.NoSuchProcess(self.pid)
        self.killed = True

    def wait(self, timeout=None):
        self.waited = True
        return 0


class _FakePsutilParent:
    """Stand-in for ``psutil.Process(<ppid>)`` returning a parent stub.

    Used by ``_parent_seems_alive`` to query ``create_time`` of a
    candidate parent. We construct these via ``_make_pid_map`` below
    rather than instantiating directly.
    """

    def __init__(self, pid: int, create_time: float) -> None:
        self.pid = pid
        self._create_time = create_time

    def create_time(self):
        return self._create_time


def _install_fake_psutil(
    monkeypatch,
    *,
    processes: list[_FakeProc],
    live_pids: dict[int, float],
):
    """Monkey-patch ``psutil`` so the helpers see a controlled fake world.

    Args:
        processes: ``_FakeProc`` instances returned by ``process_iter``.
        live_pids: mapping ``{pid: create_time}`` of every PID that
            ``pid_exists`` should report alive. ``psutil.Process(pid)``
            looks up here for ``create_time``.
    """
    import psutil

    def fake_process_iter(attrs=None):
        return list(processes)

    def fake_pid_exists(pid):
        return pid in live_pids

    real_Process = psutil.Process

    def fake_Process(pid):
        if pid in live_pids:
            return _FakePsutilParent(pid, live_pids[pid])
        # Mirror real psutil: raise NoSuchProcess for unknown PIDs.
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(psutil, "process_iter", fake_process_iter)
    monkeypatch.setattr(psutil, "pid_exists", fake_pid_exists)
    monkeypatch.setattr(psutil, "Process", fake_Process)
    return real_Process


# ---------------------------------------------------------------------------
# Unit tests — sweep_orphan_siblings()
# ---------------------------------------------------------------------------


def test_sweep_kills_orphan_with_dead_parent(monkeypatch):
    """Case (a): a sibling whose PPID is no longer alive must be killed."""
    from truememory import _lifecycle

    monkeypatch.setattr(_lifecycle.os, "getpid", lambda: 100)
    orphan = _FakeProc(
        pid=200,
        ppid=9999,  # parent is dead
        cmdline=["python.exe", "-m", "truememory.mcp_server"],
        create_time=1000.0,
    )
    _install_fake_psutil(
        monkeypatch,
        processes=[orphan],
        live_pids={},  # PPID 9999 NOT in live_pids → parent is dead
    )

    swept = _lifecycle.sweep_orphan_siblings()

    assert swept == [200]
    assert orphan.killed is True


def test_sweep_kills_orphan_when_ppid_recycled(monkeypatch):
    """Case (b): PPID exists but ``create_time`` newer than child → recycled.

    PID-recycling: child's parent died, but Windows reassigned the same
    PID to a newer unrelated process. We detect this because the
    candidate parent's ``create_time`` is later than the child's.
    """
    from truememory import _lifecycle

    monkeypatch.setattr(_lifecycle.os, "getpid", lambda: 100)
    orphan = _FakeProc(
        pid=200,
        ppid=300,
        cmdline=["python.exe", "-m", "truememory.mcp_server"],
        create_time=1000.0,  # child created at t=1000
    )
    _install_fake_psutil(
        monkeypatch,
        processes=[orphan],
        # PID 300 exists, but was created at t=2000 — AFTER the child.
        # Real parent is gone; PID was recycled.
        live_pids={300: 2000.0},
    )

    swept = _lifecycle.sweep_orphan_siblings()

    assert swept == [200]
    assert orphan.killed is True


def test_sweep_spares_live_sibling(monkeypatch):
    """Case (c): a sibling with a real, older parent must NOT be killed."""
    from truememory import _lifecycle

    monkeypatch.setattr(_lifecycle.os, "getpid", lambda: 100)
    live_sibling = _FakeProc(
        pid=200,
        ppid=300,
        cmdline=["python.exe", "-m", "truememory.mcp_server"],
        create_time=2000.0,  # child created at t=2000
    )
    _install_fake_psutil(
        monkeypatch,
        processes=[live_sibling],
        # PID 300 exists and was created BEFORE the child — real parent.
        live_pids={300: 1000.0},
    )

    swept = _lifecycle.sweep_orphan_siblings()

    assert swept == []
    assert live_sibling.killed is False


def test_sweep_spares_current_process(monkeypatch):
    """Case (d): the current process must always be spared, even if its
    own parent looks dead. ``start_parent_watchdog`` handles that case."""
    from truememory import _lifecycle

    my_pid = 100
    monkeypatch.setattr(_lifecycle.os, "getpid", lambda: my_pid)
    me = _FakeProc(
        pid=my_pid,
        ppid=9999,  # my own parent is "dead" — but I'm not a candidate
        cmdline=["python.exe", "-m", "truememory.mcp_server"],
        create_time=1000.0,
    )
    _install_fake_psutil(
        monkeypatch,
        processes=[me],
        live_pids={},
    )

    swept = _lifecycle.sweep_orphan_siblings()

    assert swept == []
    assert me.killed is False


def test_sweep_ignores_unrelated_processes(monkeypatch):
    """Non-truememory cmdlines must be ignored regardless of parent state."""
    from truememory import _lifecycle

    monkeypatch.setattr(_lifecycle.os, "getpid", lambda: 100)
    unrelated = _FakeProc(
        pid=200,
        ppid=9999,  # dead parent — but doesn't matter
        cmdline=["python.exe", "-m", "some_other_tool"],
        create_time=1000.0,
    )
    _install_fake_psutil(
        monkeypatch,
        processes=[unrelated],
        live_pids={},
    )

    swept = _lifecycle.sweep_orphan_siblings()

    assert swept == []
    assert unrelated.killed is False


def test_sweep_matches_entry_point_shim(monkeypatch):
    """``truememory-mcp.exe`` (the pyproject entry-point shim) must be
    recognized in addition to the ``-m truememory.mcp_server`` form."""
    from truememory import _lifecycle

    monkeypatch.setattr(_lifecycle.os, "getpid", lambda: 100)
    shim_orphan = _FakeProc(
        pid=200,
        ppid=9999,
        cmdline=["C:\\Python\\Scripts\\truememory-mcp.exe"],
        create_time=1000.0,
    )
    _install_fake_psutil(
        monkeypatch,
        processes=[shim_orphan],
        live_pids={},
    )

    swept = _lifecycle.sweep_orphan_siblings()

    assert swept == [200]
    assert shim_orphan.killed is True


def test_sweep_dry_run_does_not_kill(monkeypatch):
    """``dry_run=True`` must return the candidate PIDs but never call
    ``proc.kill()``. Used by the CLI for ``sweep --dry-run`` previews."""
    from truememory import _lifecycle

    monkeypatch.setattr(_lifecycle.os, "getpid", lambda: 100)
    orphan = _FakeProc(
        pid=200,
        ppid=9999,
        cmdline=["python.exe", "-m", "truememory.mcp_server"],
        create_time=1000.0,
    )
    _install_fake_psutil(
        monkeypatch,
        processes=[orphan],
        live_pids={},
    )

    swept = _lifecycle.sweep_orphan_siblings(dry_run=True)

    assert swept == [200]
    assert orphan.killed is False  # dry-run must NOT kill


def test_sweep_returns_empty_when_psutil_missing(monkeypatch):
    """Server boot must continue if ``psutil`` import fails. Sweep
    returns an empty list and never raises in that case."""
    from truememory import _lifecycle

    # Force the in-function ``import psutil`` to fail.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def bad_import(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("simulated missing psutil")
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "psutil", None)
    # Prevent the cached ``psutil`` module from being reused inside the
    # function: setitem(..., None) makes ``import psutil`` raise.
    swept = _lifecycle.sweep_orphan_siblings()
    assert swept == []


def test_sweep_survives_process_iter_exception(monkeypatch):
    """If ``psutil.process_iter`` itself raises, sweep returns ``[]`` and
    does NOT propagate the exception (boot must continue)."""
    import psutil
    from truememory import _lifecycle

    monkeypatch.setattr(_lifecycle.os, "getpid", lambda: 100)

    def broken_iter(attrs=None):
        raise OSError("simulated process_iter failure")

    monkeypatch.setattr(psutil, "process_iter", broken_iter)
    swept = _lifecycle.sweep_orphan_siblings()
    assert swept == []


# ---------------------------------------------------------------------------
# Integration test — start_parent_watchdog() on Windows
# ---------------------------------------------------------------------------
#
# POSIX has different parent-death semantics (the child gets re-parented
# to PID 1, so ``os.getppid()`` returns 1 rather than the original
# parent's PID — the watchdog's ``pid_exists(ppid)`` would stay True
# forever). On Windows, ``os.getppid()`` keeps returning the original
# parent's PID even after it dies, which is exactly what the watchdog
# needs to detect death.
#
# We use ``subprocess.Popen`` rather than ``multiprocessing.Process``
# for the parent shim because ``multiprocessing`` installs an atexit
# handler that joins managed child Processes, so the shim cannot die
# while its grandchild lives. ``subprocess.Popen``-spawned grandchildren
# have no such backreference — they're plain OS processes, which is
# exactly how Claude Code spawns the MCP server in production.


_GRANDCHILD_SCRIPT = r"""
import os, sys, time
sys.path.insert(0, %(repo_root)r)
from truememory._lifecycle import start_parent_watchdog
start_parent_watchdog(poll_interval_s=%(poll)f)
# Sleep far longer than the watchdog interval; the watchdog should
# os._exit(0) before this wakes up.
time.sleep(120)
"""


_PARENT_SHIM_SCRIPT = r"""
import os, subprocess, sys
sys.path.insert(0, %(repo_root)r)
grandchild = subprocess.Popen(
    [sys.executable, "-c", %(grandchild_code)r],
)
with open(%(ready_path)r, "w", encoding="utf-8") as f:
    f.write(str(grandchild.pid))
    f.flush()
# Parent shim exits immediately. Grandchild's parent is now dead.
"""


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="Windows-specific orphan reaper: POSIX re-parents to PID 1 so ppid never dies",
)
def test_watchdog_exits_when_parent_dies(tmp_path):
    """End-to-end: grandchild starts the watchdog, parent shim dies,
    grandchild must self-terminate within poll_interval_s plus slack.

    Uses a 5s poll interval so the test wraps in ~10s rather than the
    30s default — keeps CI fast while still exercising the real
    sleep/check/exit loop. Total bound: ~20s.
    """
    import subprocess
    from pathlib import Path

    repo_root = str(Path(__file__).resolve().parent.parent)
    ready_path = tmp_path / "grandchild_pid.txt"
    poll_interval_s = 5.0

    grandchild_code = _GRANDCHILD_SCRIPT % {
        "repo_root": repo_root,
        "poll": poll_interval_s,
    }
    parent_code = _PARENT_SHIM_SCRIPT % {
        "repo_root": repo_root,
        "grandchild_code": grandchild_code,
        "ready_path": str(ready_path),
    }

    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code],
        # Detach stdio so the grandchild doesn't inherit pytest's
        # captured pipes (which could keep the parent alive via
        # pipe-write).
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )

    # Wait for the parent shim to spawn the grandchild and write the PID.
    deadline = time.time() + 15
    while time.time() < deadline:
        if ready_path.exists() and ready_path.stat().st_size > 0:
            break
        time.sleep(0.1)
    else:
        try:
            parent.kill()
        except Exception:
            pass
        pytest.fail("parent shim did not write grandchild PID within 15s")

    grandchild_pid = int(ready_path.read_text(encoding="utf-8").strip())

    # Wait for the parent shim to exit.
    try:
        parent_rc = parent.wait(timeout=10)
    except subprocess.TimeoutExpired:
        parent.kill()
        pytest.fail("parent shim failed to exit within 10s")
    assert parent_rc == 0, f"parent shim exited non-zero: {parent_rc}"

    # Watchdog poll interval = 5s; allow up to ~3x for slack on a
    # contended box. Total wait bound ~20s.
    import psutil
    deadline = time.time() + (poll_interval_s * 3 + 5)
    while time.time() < deadline:
        if not psutil.pid_exists(grandchild_pid):
            break
        time.sleep(0.5)
    else:
        # Cleanup before failing.
        try:
            psutil.Process(grandchild_pid).kill()
        except Exception:
            pass
        pytest.fail(
            f"grandchild PID {grandchild_pid} still alive after parent died — "
            f"watchdog did not fire within {poll_interval_s * 3 + 5}s"
        )


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_help_returns_zero(capsys):
    """The CLI must return 0 on ``help`` / ``--help`` / ``-h``."""
    from truememory import _lifecycle

    assert _lifecycle._cli([]) == 0
    assert _lifecycle._cli(["--help"]) == 0
    assert _lifecycle._cli(["-h"]) == 0


def test_cli_unknown_command_returns_two():
    """The CLI must return exit code 2 on unknown subcommands."""
    from truememory import _lifecycle

    assert _lifecycle._cli(["bogus"]) == 2


def test_cli_dry_run_does_not_kill(monkeypatch, capsys):
    """``sweep --dry-run`` must report would-kill PIDs without killing."""
    from truememory import _lifecycle

    monkeypatch.setattr(_lifecycle.os, "getpid", lambda: 100)
    orphan = _FakeProc(
        pid=200,
        ppid=9999,
        cmdline=["python.exe", "-m", "truememory.mcp_server"],
        create_time=1000.0,
    )
    _install_fake_psutil(
        monkeypatch,
        processes=[orphan],
        live_pids={},
    )

    rc = _lifecycle._cli(["sweep", "--dry-run"])
    captured = capsys.readouterr()
    assert rc == 0
    assert orphan.killed is False
    assert "would kill" in captured.out
    assert "200" in captured.out
