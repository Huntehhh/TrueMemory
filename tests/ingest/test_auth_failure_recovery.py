"""Tests for opt-in auth-failure recovery in the ingest path.

Covers the four behaviours the feature promises:

1. ``_complete_claude_cli`` classifies dead-credential failures as
   :class:`LLMAuthError` (auth-pattern stderr OR a known dead-token exit
   code with empty stderr), keeps a plain :class:`LLMError` for non-auth
   nonzero exits, and still succeeds on a clean exit-0 response.
2. ``truememory-ingest ingest`` exits 2 and writes a backlog marker with
   ``reason="cli_auth_failure"`` when an ``LLMAuthError`` propagates.
3. ``auth_recovery.get_on_auth_failure_cmd`` reads config.json then the env
   var; ``run_auth_recovery`` is a no-op (False) when unset, True on a fake
   exit-0 command, False on nonzero, and never raises.
4. The retry guard: a marker already at ``retry_count >= MAX_AUTH_RETRIES``
   is not recovered again.

No real ``claude`` calls, no network. HOME / config / backlog dir are all
redirected through tmp_path + monkeypatch.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from truememory.ingest.models import (
    LLMConfig,
    LLMError,
    LLMAuthError,
    _complete_claude_cli,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_completed(returncode: int, stdout: str = "", stderr: str = ""):
    """Build a stand-in for subprocess.CompletedProcess."""
    return subprocess.CompletedProcess(
        args=["claude"], returncode=returncode, stdout=stdout, stderr=stderr,
    )


def _claude_cli_config() -> LLMConfig:
    return LLMConfig(provider="claude_cli", model="")


# ---------------------------------------------------------------------------
# 1. _complete_claude_cli classification
# ---------------------------------------------------------------------------

def test_auth_pattern_stderr_raises_llmautherror(monkeypatch):
    """A nonzero exit whose stderr matches an auth signal → LLMAuthError."""
    monkeypatch.setattr(
        "truememory.ingest.models._claude_cli_available", lambda: True
    )
    monkeypatch.setattr(
        "truememory.ingest.models.shutil.which", lambda _: "/usr/bin/claude"
    )

    def _fake_run(*args, **kwargs):
        return _fake_completed(
            returncode=1,
            stderr="Error: Invalid token. Please run `claude login` to authenticate.",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(LLMAuthError):
        _complete_claude_cli(_claude_cli_config(), "hi", "")


def test_dead_token_returncode_empty_stderr_raises_llmautherror(monkeypatch):
    """A known dead-token exit code (1) with EMPTY stderr → LLMAuthError."""
    monkeypatch.setattr(
        "truememory.ingest.models._claude_cli_available", lambda: True
    )
    monkeypatch.setattr(
        "truememory.ingest.models.shutil.which", lambda _: "/usr/bin/claude"
    )

    def _fake_run(*args, **kwargs):
        return _fake_completed(returncode=1, stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(LLMAuthError):
        _complete_claude_cli(_claude_cli_config(), "hi", "")


def test_windows_dead_token_returncode_raises_llmautherror(monkeypatch):
    """The Windows dead-token code (3221225794) with empty stderr → LLMAuthError."""
    monkeypatch.setattr(
        "truememory.ingest.models._claude_cli_available", lambda: True
    )
    monkeypatch.setattr(
        "truememory.ingest.models.shutil.which", lambda _: "claude"
    )

    def _fake_run(*args, **kwargs):
        return _fake_completed(returncode=3221225794, stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(LLMAuthError):
        _complete_claude_cli(_claude_cli_config(), "hi", "")


def test_non_auth_nonzero_exit_raises_plain_llmerror(monkeypatch):
    """A nonzero exit with a NON-auth stderr stays a plain LLMError.

    Uses exit code 2 (not in the dead-token set) with a content/runtime error
    message so neither the stderr nor the returncode heuristic fires.
    """
    monkeypatch.setattr(
        "truememory.ingest.models._claude_cli_available", lambda: True
    )
    monkeypatch.setattr(
        "truememory.ingest.models.shutil.which", lambda _: "/usr/bin/claude"
    )

    def _fake_run(*args, **kwargs):
        return _fake_completed(
            returncode=2, stderr="Error: rate limit exceeded, try again later",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(LLMError) as exc_info:
        _complete_claude_cli(_claude_cli_config(), "hi", "")
    # Must be a PLAIN LLMError, not the auth subclass.
    assert not isinstance(exc_info.value, LLMAuthError)


def test_dead_token_code_with_nonauth_stderr_stays_plain_error(monkeypatch):
    """A dead-token exit code BUT a non-auth, non-empty stderr → plain LLMError.

    This is the conservative guard: we only treat codes 1/129/... as auth when
    the stderr is empty or itself auth-ish.
    """
    monkeypatch.setattr(
        "truememory.ingest.models._claude_cli_available", lambda: True
    )
    monkeypatch.setattr(
        "truememory.ingest.models.shutil.which", lambda _: "/usr/bin/claude"
    )

    def _fake_run(*args, **kwargs):
        return _fake_completed(
            returncode=1, stderr="Error: model 'foo' does not exist",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(LLMError) as exc_info:
        _complete_claude_cli(_claude_cli_config(), "hi", "")
    assert not isinstance(exc_info.value, LLMAuthError)


def test_is_error_json_with_auth_message_raises_llmautherror(monkeypatch):
    """Exit 0 but is_error=true with an auth-ish result → LLMAuthError."""
    monkeypatch.setattr(
        "truememory.ingest.models._claude_cli_available", lambda: True
    )
    monkeypatch.setattr(
        "truememory.ingest.models.shutil.which", lambda _: "/usr/bin/claude"
    )

    envelope = json.dumps({
        "type": "result",
        "is_error": True,
        "result": "Authentication required: please run login",
    })

    def _fake_run(*args, **kwargs):
        return _fake_completed(returncode=0, stdout=envelope)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(LLMAuthError):
        _complete_claude_cli(_claude_cli_config(), "hi", "")


def test_clean_exit_zero_succeeds(monkeypatch):
    """A clean exit-0 JSON envelope returns the result string normally."""
    monkeypatch.setattr(
        "truememory.ingest.models._claude_cli_available", lambda: True
    )
    monkeypatch.setattr(
        "truememory.ingest.models.shutil.which", lambda _: "/usr/bin/claude"
    )

    envelope = json.dumps({
        "type": "result",
        "is_error": False,
        "result": "[]",
    })

    def _fake_run(*args, **kwargs):
        return _fake_completed(returncode=0, stdout=envelope)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    out = _complete_claude_cli(_claude_cli_config(), "hi", "")
    assert out == "[]"


# ---------------------------------------------------------------------------
# 2. CLI exits 2 + writes a cli_auth_failure backlog marker
# ---------------------------------------------------------------------------

def test_cli_ingest_exits_2_and_queues_on_auth_error(monkeypatch, tmp_path):
    """When ingest() raises LLMAuthError, the ingest command must exit 2 and
    write a backlog marker tagged reason='cli_auth_failure' with retry_count=0."""
    from truememory.ingest import cli as cli_mod
    from truememory.ingest.hooks import stop as stop_mod

    # Redirect the backlog dir the marker is written to.
    backlog = tmp_path / "backlog"
    monkeypatch.setattr(stop_mod, "BACKLOG_DIR", backlog)

    # A real, readable transcript file so preflight passes.
    transcript = tmp_path / "transcript.json"
    transcript.write_text("[]", encoding="utf-8")

    # Skip the model-server warmup (it's already wrapped in try/except, but be
    # explicit so the test doesn't depend on network/daemon state).
    monkeypatch.setattr(
        "truememory.model_client.ensure_server_running",
        lambda *a, **k: None,
        raising=False,
    )

    # Make the core ingest raise an auth error.
    def _boom(*args, **kwargs):
        raise LLMAuthError("claude CLI auth failure (exit 1): not logged in")

    monkeypatch.setattr(cli_mod, "ingest", _boom)

    # Build the args namespace the way argparse would.
    args = type("Args", (), {})()
    args.transcript = str(transcript)
    args.user = "alice"
    args.db = str(tmp_path / "mem.db")
    args.threshold = 0.30
    args.trace = None
    args.provider = "auto"
    args.model = ""
    args.session = "sess-auth-xyz"
    args.verbose = False

    with pytest.raises(SystemExit) as exc_info:
        cli_mod._run_ingest(args)
    assert exc_info.value.code == 2

    marker = backlog / "sess-auth-xyz.json"
    assert marker.exists(), "auth failure must write a backlog marker"
    data = json.loads(marker.read_text(encoding="utf-8"))
    assert data["reason"] == "cli_auth_failure"
    assert data["retry_count"] == 0
    assert data["session_id"] == "sess-auth-xyz"
    assert data["transcript_path"] == str(transcript)
    assert data["user_id"] == "alice"
    # Schema parity with the Stop hook's marker.
    required = {"transcript_path", "session_id", "user_id", "db_path", "queued_at", "reason"}
    assert required <= set(data.keys())


def test_cli_auth_failure_increments_retry_count_from_env(monkeypatch, tmp_path):
    """When re-run from a prior auth marker (TRUEMEMORY_AUTH_RETRY_COUNT set),
    a repeated auth failure stores retry_count = prior + 1."""
    from truememory.ingest import cli as cli_mod
    from truememory.ingest.hooks import stop as stop_mod

    backlog = tmp_path / "backlog"
    monkeypatch.setattr(stop_mod, "BACKLOG_DIR", backlog)
    monkeypatch.setenv("TRUEMEMORY_AUTH_RETRY_COUNT", "1")

    transcript = tmp_path / "t.json"
    transcript.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        "truememory.model_client.ensure_server_running",
        lambda *a, **k: None, raising=False,
    )
    monkeypatch.setattr(
        cli_mod, "ingest",
        lambda *a, **k: (_ for _ in ()).throw(LLMAuthError("unauthorized")),
    )

    args = type("Args", (), {})()
    args.transcript = str(transcript)
    args.user = ""
    args.db = None
    args.threshold = 0.30
    args.trace = None
    args.provider = "auto"
    args.model = ""
    args.session = "sess-retry"
    args.verbose = False

    with pytest.raises(SystemExit) as exc_info:
        cli_mod._run_ingest(args)
    assert exc_info.value.code == 2

    data = json.loads((backlog / "sess-retry.json").read_text(encoding="utf-8"))
    assert data["retry_count"] == 2


# ---------------------------------------------------------------------------
# 3. auth_recovery config + command runner
# ---------------------------------------------------------------------------

def test_get_on_auth_failure_cmd_unset_returns_none(monkeypatch, tmp_path):
    """No config key and no env var → None."""
    import truememory.ingest.auth_recovery as ar

    monkeypatch.setattr(ar, "_CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.delenv("TRUEMEMORY_ON_AUTH_FAILURE_CMD", raising=False)
    assert ar.get_on_auth_failure_cmd() is None


def test_get_on_auth_failure_cmd_reads_config(monkeypatch, tmp_path):
    """The config.json key is read (and wins when present)."""
    import truememory.ingest.auth_recovery as ar

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"tier": "base", "on_auth_failure_cmd": "my-reauth --headless"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(ar, "_CONFIG_PATH", config_path)
    monkeypatch.delenv("TRUEMEMORY_ON_AUTH_FAILURE_CMD", raising=False)
    assert ar.get_on_auth_failure_cmd() == "my-reauth --headless"


def test_get_on_auth_failure_cmd_reads_env_fallback(monkeypatch, tmp_path):
    """With no config key, the env var is used."""
    import truememory.ingest.auth_recovery as ar

    monkeypatch.setattr(ar, "_CONFIG_PATH", tmp_path / "config.json")  # missing
    monkeypatch.setenv("TRUEMEMORY_ON_AUTH_FAILURE_CMD", "env-reauth.sh")
    assert ar.get_on_auth_failure_cmd() == "env-reauth.sh"


def test_get_on_auth_failure_cmd_corrupt_config_falls_back_to_env(monkeypatch, tmp_path):
    """A corrupt config.json must not raise — fall through to the env var."""
    import truememory.ingest.auth_recovery as ar

    config_path = tmp_path / "config.json"
    config_path.write_text("{ this is not json", encoding="utf-8")
    monkeypatch.setattr(ar, "_CONFIG_PATH", config_path)
    monkeypatch.setenv("TRUEMEMORY_ON_AUTH_FAILURE_CMD", "fallback-cmd")
    assert ar.get_on_auth_failure_cmd() == "fallback-cmd"


def test_run_auth_recovery_noop_when_unset(monkeypatch, tmp_path):
    """No configured command → returns False (no-op) and does not run anything."""
    import truememory.ingest.auth_recovery as ar

    monkeypatch.setattr(ar, "_CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.delenv("TRUEMEMORY_ON_AUTH_FAILURE_CMD", raising=False)

    # Guard: subprocess.run must never be called in the no-op path.
    def _no_run(*args, **kwargs):
        raise AssertionError("run_auth_recovery must not run a command when unset")

    monkeypatch.setattr(subprocess, "run", _no_run)
    assert ar.run_auth_recovery() is False


def test_run_auth_recovery_true_on_exit_zero(monkeypatch, tmp_path):
    """A configured command that exits 0 → True."""
    import truememory.ingest.auth_recovery as ar

    monkeypatch.setattr(ar, "_CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setenv("TRUEMEMORY_ON_AUTH_FAILURE_CMD", "anything")

    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: _fake_completed(returncode=0, stdout="ok"),
    )
    assert ar.run_auth_recovery(timeout=5) is True


def test_run_auth_recovery_false_on_nonzero(monkeypatch, tmp_path):
    """A configured command that exits nonzero → False."""
    import truememory.ingest.auth_recovery as ar

    monkeypatch.setattr(ar, "_CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setenv("TRUEMEMORY_ON_AUTH_FAILURE_CMD", "anything")

    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: _fake_completed(returncode=7, stderr="login failed"),
    )
    assert ar.run_auth_recovery(timeout=5) is False


def test_run_auth_recovery_never_raises_on_exception(monkeypatch, tmp_path):
    """If the subprocess launch itself blows up, return False — never raise."""
    import truememory.ingest.auth_recovery as ar

    monkeypatch.setattr(ar, "_CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setenv("TRUEMEMORY_ON_AUTH_FAILURE_CMD", "anything")

    def _boom(*args, **kwargs):
        raise OSError("cannot spawn")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert ar.run_auth_recovery(timeout=5) is False


def test_run_auth_recovery_never_raises_on_timeout(monkeypatch, tmp_path):
    """A timeout is swallowed and returns False."""
    import truememory.ingest.auth_recovery as ar

    monkeypatch.setattr(ar, "_CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setenv("TRUEMEMORY_ON_AUTH_FAILURE_CMD", "anything")

    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="anything", timeout=1)

    monkeypatch.setattr(subprocess, "run", _timeout)
    assert ar.run_auth_recovery(timeout=1) is False


# ---------------------------------------------------------------------------
# 4. Retry guard — markers at/over MAX are not recovered again
# ---------------------------------------------------------------------------

def test_drain_skips_recovery_when_all_markers_over_cap(monkeypatch, tmp_path):
    """If every cli_auth_failure marker is at/over MAX_AUTH_RETRIES, the drain
    must NOT spawn the re-auth command."""
    import truememory.ingest.auth_recovery as ar
    from truememory.ingest.hooks import session_start as ss_mod

    backlog = tmp_path / "backlog"
    backlog.mkdir()
    # A marker that has already exhausted its retries.
    (backlog / "sess-dead.json").write_text(json.dumps({
        "transcript_path": "/x.json",
        "session_id": "sess-dead",
        "user_id": "",
        "db_path": "",
        "queued_at": "2026-01-01T00:00:00+00:00",
        "reason": "cli_auth_failure",
        "retry_count": ar.MAX_AUTH_RETRIES,
    }), encoding="utf-8")

    monkeypatch.setattr(ss_mod, "BACKLOG_DIR", backlog)
    # A command IS configured (so the only reason to skip is the retry cap).
    monkeypatch.setattr(ar, "get_on_auth_failure_cmd", lambda: "reauth-cmd")

    spawn_calls = {"n": 0}
    monkeypatch.setattr(
        ar, "spawn_auth_recovery",
        lambda *a, **k: spawn_calls.__setitem__("n", spawn_calls["n"] + 1) or True,
    )

    ss_mod._maybe_run_auth_recovery()
    assert spawn_calls["n"] == 0, "recovery must not run when all markers are over the cap"


def test_drain_runs_recovery_when_marker_under_cap(monkeypatch, tmp_path):
    """A cli_auth_failure marker under the retry cap triggers exactly one
    recovery spawn per drain."""
    import truememory.ingest.auth_recovery as ar
    from truememory.ingest.hooks import session_start as ss_mod

    backlog = tmp_path / "backlog"
    backlog.mkdir()
    (backlog / "sess-fresh.json").write_text(json.dumps({
        "transcript_path": "/x.json",
        "session_id": "sess-fresh",
        "user_id": "",
        "db_path": "",
        "queued_at": "2026-01-01T00:00:00+00:00",
        "reason": "cli_auth_failure",
        "retry_count": 0,
    }), encoding="utf-8")

    monkeypatch.setattr(ss_mod, "BACKLOG_DIR", backlog)
    monkeypatch.setattr(ar, "get_on_auth_failure_cmd", lambda: "reauth-cmd")

    spawn_calls = {"n": 0}
    monkeypatch.setattr(
        ar, "spawn_auth_recovery",
        lambda *a, **k: spawn_calls.__setitem__("n", spawn_calls["n"] + 1) or True,
    )

    ss_mod._maybe_run_auth_recovery()
    assert spawn_calls["n"] == 1, "exactly one recovery spawn when a marker is under the cap"


def test_drain_no_recovery_when_no_command_configured(monkeypatch, tmp_path):
    """Even with an eligible marker, no recovery runs if no command is set
    (recovery is strictly opt-in)."""
    import truememory.ingest.auth_recovery as ar
    from truememory.ingest.hooks import session_start as ss_mod

    backlog = tmp_path / "backlog"
    backlog.mkdir()
    (backlog / "sess.json").write_text(json.dumps({
        "transcript_path": "/x.json",
        "session_id": "sess",
        "reason": "cli_auth_failure",
        "retry_count": 0,
    }), encoding="utf-8")

    monkeypatch.setattr(ss_mod, "BACKLOG_DIR", backlog)
    monkeypatch.setattr(ar, "get_on_auth_failure_cmd", lambda: None)

    spawn_calls = {"n": 0}
    monkeypatch.setattr(
        ar, "spawn_auth_recovery",
        lambda *a, **k: spawn_calls.__setitem__("n", spawn_calls["n"] + 1) or True,
    )

    ss_mod._maybe_run_auth_recovery()
    assert spawn_calls["n"] == 0
