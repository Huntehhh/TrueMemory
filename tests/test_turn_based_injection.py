"""Tests for turn-based memory injection on UserPromptSubmit.

Covers the gate (turn count OR prompt length), marker-file dedup,
``INJECT_DISABLED`` kill switch, graceful degradation on bad inputs,
and the helpers in ``_shared.py`` (count_user_turns, already_injected,
mark_injected) plus ``user_prompt_submit._build_turn_based_query``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def truememory_home(monkeypatch, tmp_path):
    """Scope TURN_INJECTED_DIR and EXTRACTED_DIR into tmp_path.

    Critical: must monkeypatch BEFORE importing _shared so the module-level
    Path.home() / ".truememory" / "turn_injected" resolves under tmp_path.
    """
    home = tmp_path / "home"
    home.mkdir()
    fake_truememory = home / ".truememory"
    fake_truememory.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # Windows

    # Force reimport with patched home
    import importlib
    import truememory.ingest.hooks._shared as shared_mod
    importlib.reload(shared_mod)
    monkeypatch.setattr(shared_mod, "TURN_INJECTED_DIR", fake_truememory / "turn_injected")
    monkeypatch.setattr(shared_mod, "EXTRACTED_DIR", fake_truememory / "extracted")

    import truememory.ingest.hooks.user_prompt_submit as ups_mod
    importlib.reload(ups_mod)
    yield {"home": home, "truememory": fake_truememory, "shared": shared_mod, "ups": ups_mod}


def _make_transcript(path: Path, user_turn_count: int, msg_body: str = "hi") -> None:
    """Write a JSONL transcript with ``user_turn_count`` user entries."""
    lines = []
    for i in range(user_turn_count):
        lines.append(json.dumps({"type": "user", "content": f"{msg_body} {i}"}))
        # Add an assistant turn between users (not counted)
        lines.append(json.dumps({"type": "assistant", "content": "ok"}))
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# _shared.py helper tests
# ---------------------------------------------------------------------------


def test_count_user_turns_jsonl(truememory_home, tmp_path):
    shared = truememory_home["shared"]
    t = tmp_path / "transcript.jsonl"
    _make_transcript(t, user_turn_count=7)
    assert shared.count_user_turns(str(t)) == 7


def test_count_user_turns_empty_file(truememory_home, tmp_path):
    shared = truememory_home["shared"]
    t = tmp_path / "empty.jsonl"
    t.write_text("", encoding="utf-8")
    assert shared.count_user_turns(str(t)) == 0


def test_count_user_turns_missing_file_returns_zero(truememory_home, tmp_path):
    shared = truememory_home["shared"]
    assert shared.count_user_turns(str(tmp_path / "does-not-exist.jsonl")) == 0


def test_count_user_turns_json_array_format(truememory_home, tmp_path):
    """Some Claude Code versions serialize the transcript as a single JSON array."""
    shared = truememory_home["shared"]
    t = tmp_path / "array.json"
    arr = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]
    t.write_text(json.dumps(arr), encoding="utf-8")
    assert shared.count_user_turns(str(t)) == 2


def test_already_injected_round_trip(truememory_home):
    shared = truememory_home["shared"]
    sid = "session-abc-123"
    assert shared.already_injected(sid) is False
    shared.mark_injected(sid, {"trigger": "turns", "n_results": 5})
    assert shared.already_injected(sid) is True
    # Marker file content sanity check
    marker = truememory_home["truememory"] / "turn_injected" / "session-abc-123"
    data = json.loads(marker.read_text(encoding="utf-8"))
    assert data["trigger"] == "turns"
    assert data["n_results"] == 5
    assert "timestamp" in data


def test_mark_injected_unknown_session_no_op(truememory_home):
    shared = truememory_home["shared"]
    shared.mark_injected("unknown", {"trigger": "turns"})
    shared.mark_injected("", {"trigger": "turns"})
    # No marker dir created, no exception
    assert not (truememory_home["truememory"] / "turn_injected" / "unknown").exists()


def test_already_injected_unknown_session_is_false(truememory_home):
    shared = truememory_home["shared"]
    assert shared.already_injected("unknown") is False
    assert shared.already_injected("") is False


# ---------------------------------------------------------------------------
# is_subagent_invocation — guard semantics
# ---------------------------------------------------------------------------
#
# DESIGN NOTE for upstream reviewers
# ----------------------------------
# Per the canonical Claude Code hooks docs (code.claude.com/docs/en/hooks),
# sub-agents spawned via the Task tool fire SessionStart, UserPromptSubmit,
# and SessionEnd hooks in their own session context — same as the main
# thread. This is intentional behavior on Anthropic's side.
#
# This PR adds a runtime guard (`is_subagent_invocation` in `_shared.py`)
# that causes all three TrueMemory hooks to early-return when invoked
# inside a sub-agent. Two reasons:
#
#   1. Sub-agent prompts are orchestrator-generated, not real user input.
#      Ingesting them into TrueMemory pollutes the memory store with
#      task-context strings rather than user facts.
#   2. On Windows, every sub-agent spawn produces 1-3 hook subprocess
#      fires, each of which causes a console window flash (Claude Code's
#      Node spawn omits `windowsHide: true` and there is no settings.json
#      escape hatch). The guard collapses each fire into a <10ms no-op so
#      the user-visible noise is minimized.
#
# This is a Hunter-specific design preference and may disagree with
# upstream's intent. If you (upstream maintainer) want the canonical
# "sub-agents fire hooks normally" behavior, the guard can be made
# opt-in via a `TRUEMEMORY_SKIP_SUBAGENT_HOOKS` env var (default off).
# Happy to refactor as a follow-up commit if you'd prefer that shape.
#
# Detection uses two signals (either sufficient):
#   - `agent_id` field in hook stdin — the canonical sub-agent signal
#     per docs ("Present only when the hook fires inside a subagent call").
#   - `/subagents/` directory component in `transcript_path` — fallback
#     for older Claude Code versions (pre-v2.1.69) and any event where
#     `agent_id` might be absent.
#
# Crucially, `agent_type` is NOT used as a signal — it's also set for
# user-launched `--agent <name>` main sessions where the user IS the
# speaker. Using it would silently skip real user prompts. The tests
# below lock that decision in.


def test_subagent_guard_agent_id_present(truememory_home):
    """agent_id in stdin is the canonical sub-agent signal — must trigger guard."""
    shared = truememory_home["shared"]
    assert shared.is_subagent_invocation({"agent_id": "agent-xyz"}) is True


def test_subagent_guard_subagents_path(truememory_home, tmp_path):
    """transcript_path containing /subagents/ is the fallback signal — must trigger guard."""
    shared = truememory_home["shared"]
    sub_tp = str(tmp_path / "sess-abc" / "subagents" / "agent-y.jsonl")
    assert shared.is_subagent_invocation({}, transcript_path=sub_tp) is True
    # Windows backslash form must also match
    win_tp = r"C:\Users\x\.claude\projects\proj\sess\subagents\agent-y.jsonl"
    assert shared.is_subagent_invocation({}, transcript_path=win_tp) is True


def test_subagent_guard_agent_type_alone_is_NOT_subagent(truememory_home):
    """agent_type without agent_id and without /subagents/ in path is the
    `--agent <name>` main-session case. The user IS the speaker; their
    prompts are real user input and must be ingested. The guard must NOT
    trigger here."""
    shared = truememory_home["shared"]
    main_session_tp = r"C:\Users\x\.claude\projects\proj\sess-abc.jsonl"
    assert shared.is_subagent_invocation(
        {"agent_type": "novius-clinical-advisor"},
        transcript_path=main_session_tp,
    ) is False


def test_subagent_guard_main_session_is_not_subagent(truememory_home):
    """Plain main-thread fire with no agent_id, no agent_type, no /subagents/ path."""
    shared = truememory_home["shared"]
    assert shared.is_subagent_invocation({"session_id": "abc"}) is False
    assert shared.is_subagent_invocation(
        {"session_id": "abc", "transcript_path": "C:/proj/sess.jsonl"}
    ) is False


def test_subagent_guard_empty_input(truememory_home):
    """No input at all → safe default: not a sub-agent (don't false-positive
    on init failures that produce empty stdin)."""
    shared = truememory_home["shared"]
    assert shared.is_subagent_invocation() is False
    assert shared.is_subagent_invocation(None) is False
    assert shared.is_subagent_invocation({}) is False


# ---------------------------------------------------------------------------
# _build_turn_based_query tests
# ---------------------------------------------------------------------------


def test_build_query_takes_last_k_turns(truememory_home, tmp_path):
    ups = truememory_home["ups"]
    t = tmp_path / "t.jsonl"
    _make_transcript(t, user_turn_count=10, msg_body="msg")
    q = ups._build_turn_based_query(str(t), k=3)
    # last 3 user messages should be "msg 7", "msg 8", "msg 9"
    assert "msg 7" in q
    assert "msg 8" in q
    assert "msg 9" in q
    assert "msg 0" not in q  # earliest turns excluded


def test_build_query_truncates_each_turn(truememory_home, tmp_path, monkeypatch):
    ups = truememory_home["ups"]
    monkeypatch.setattr(ups, "INJECT_QUERY_TURN_CHARS", 10)
    t = tmp_path / "t.jsonl"
    huge = "X" * 5000
    t.write_text(json.dumps({"type": "user", "content": huge}), encoding="utf-8")
    q = ups._build_turn_based_query(str(t), k=6)
    # Truncated to 10 chars per turn
    assert q == "X" * 10


def test_build_query_empty_transcript_returns_empty(truememory_home, tmp_path):
    ups = truememory_home["ups"]
    t = tmp_path / "t.jsonl"
    t.write_text("", encoding="utf-8")
    assert ups._build_turn_based_query(str(t), k=6) == ""


# ---------------------------------------------------------------------------
# _try_turn_based_injection — the gate
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_memory_search(truememory_home, monkeypatch):
    """Stub Memory + _get_llm_fn so tests don't touch the real DB or LLM."""
    ups = truememory_home["ups"]

    class _StubMemory:
        def __init__(self, path=None):
            self.path = path

        def search_deep(self, query, user_id=None, limit=10, llm_fn=None):
            # Return a deterministic shape the formatter can ingest
            return [
                {"content": f"memory-{i} for query={query[:20]}"}
                for i in range(min(3, limit))
            ]

    import truememory.client as client_mod
    monkeypatch.setattr(client_mod, "Memory", _StubMemory)

    # Stub _get_llm_fn so we don't touch the real provider chain
    import truememory.mcp_server as ms
    monkeypatch.setattr(ms, "_get_llm_fn", lambda: (lambda p: "stub-hyde"))

    return ups


def test_gate_not_crossed_no_inject(mock_memory_search, tmp_path):
    """turns=12 + chars=100 → below both thresholds → no inject, no marker."""
    ups = mock_memory_search
    t = tmp_path / "t.jsonl"
    _make_transcript(t, user_turn_count=12)
    prompt = "x" * 100
    result = ups._try_turn_based_injection(
        prompt, "sess-1", str(t), user_id="hunter", db_path="",
    )
    assert result is None


def test_turn_trigger_fires_at_13_turns(mock_memory_search, tmp_path):
    """turns=13 + short prompt → turn trigger fires."""
    ups = mock_memory_search
    t = tmp_path / "t.jsonl"
    _make_transcript(t, user_turn_count=13)
    prompt = "x" * 100
    result = ups._try_turn_based_injection(
        prompt, "sess-2", str(t), user_id="hunter", db_path="",
    )
    assert result is not None
    assert "<truememory-context>" in result
    assert "trigger=turns" in result


def test_length_trigger_fires_at_334_chars(mock_memory_search, tmp_path):
    """turns=1 + chars=334 → length trigger fires before turn-count branch."""
    ups = mock_memory_search
    t = tmp_path / "t.jsonl"
    _make_transcript(t, user_turn_count=1)
    prompt = "x" * 334
    result = ups._try_turn_based_injection(
        prompt, "sess-3", str(t), user_id="hunter", db_path="",
    )
    assert result is not None
    assert "<truememory-context>" in result
    assert "trigger=length" in result


def test_marker_dedup_blocks_second_fire(mock_memory_search, tmp_path):
    """Once marker file exists, subsequent calls return None."""
    ups = mock_memory_search
    t = tmp_path / "t.jsonl"
    _make_transcript(t, user_turn_count=13)
    prompt = "x" * 100
    first = ups._try_turn_based_injection(prompt, "sess-4", str(t), "hunter", "")
    second = ups._try_turn_based_injection(prompt, "sess-4", str(t), "hunter", "")
    assert first is not None
    assert second is None


def test_inject_disabled_env_kill_switch(mock_memory_search, tmp_path, monkeypatch):
    """INJECT_DISABLED=1 short-circuits before any work."""
    ups = mock_memory_search
    monkeypatch.setattr(ups, "INJECT_DISABLED", True)
    t = tmp_path / "t.jsonl"
    _make_transcript(t, user_turn_count=13)
    result = ups._try_turn_based_injection(
        "x" * 334, "sess-5", str(t), "hunter", "",
    )
    assert result is None


def test_empty_transcript_graceful_no_inject(mock_memory_search, tmp_path):
    """Empty transcript can't trigger turn count → no inject from turn branch.
    Length branch can still fire — verify with a short prompt."""
    ups = mock_memory_search
    t = tmp_path / "t.jsonl"
    t.write_text("", encoding="utf-8")
    result = ups._try_turn_based_injection(
        "x" * 100, "sess-6", str(t), "hunter", "",
    )
    assert result is None


def test_missing_session_id_no_inject(mock_memory_search, tmp_path):
    ups = mock_memory_search
    t = tmp_path / "t.jsonl"
    _make_transcript(t, user_turn_count=13)
    result = ups._try_turn_based_injection(
        "x" * 100, "", str(t), "hunter", "",
    )
    assert result is None


def test_no_results_still_writes_marker(mock_memory_search, tmp_path, monkeypatch):
    """When search returns [], marker file is still written so we don't
    re-pay the search cost on every subsequent prompt."""
    ups = mock_memory_search

    class _EmptyMemory:
        def __init__(self, path=None):
            pass

        def search_deep(self, query, user_id=None, limit=10, llm_fn=None):
            return []

    import truememory.client as client_mod
    monkeypatch.setattr(client_mod, "Memory", _EmptyMemory)

    t = tmp_path / "t.jsonl"
    _make_transcript(t, user_turn_count=13)
    result = ups._try_turn_based_injection(
        "x" * 100, "sess-7", str(t), "hunter", "",
    )
    assert result is None
    # Marker should exist despite zero results
    import truememory.ingest.hooks._shared as shared
    assert shared.already_injected("sess-7") is True


def test_search_exception_does_not_crash(mock_memory_search, tmp_path, monkeypatch):
    """If Memory.search_deep raises, the function returns None (hook can't crash)."""
    ups = mock_memory_search

    class _BoomMemory:
        def __init__(self, path=None):
            pass

        def search_deep(self, *a, **kw):
            raise RuntimeError("boom")

    import truememory.client as client_mod
    monkeypatch.setattr(client_mod, "Memory", _BoomMemory)

    t = tmp_path / "t.jsonl"
    _make_transcript(t, user_turn_count=13)
    result = ups._try_turn_based_injection(
        "x" * 100, "sess-8", str(t), "hunter", "",
    )
    assert result is None
