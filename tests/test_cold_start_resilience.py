"""Regression locks for MCP cold-start resilience.

Three failure modes this file pins down, all of which would otherwise leave
the MCP server permanently hung or crashed at boot:

1. **Windows os.WNOHANG missing** — `_reap_children` calls `os.waitpid(-1,
   os.WNOHANG)`. WNOHANG is POSIX-only; on Windows the `os` module has no
   `WNOHANG` attribute and the backlog drainer thread crashes with
   AttributeError. Guard: hasattr check at top of `_reap_children`.

2. **Reranker preload stalls forever** — corrupt HuggingFace cache, blocked
   download, or Windows Defender ASR denying a sentencepiece shim leaves
   `CrossEncoder(...)` blocked indefinitely. Guard: watchdog thread with
   TRUEMEMORY_RERANKER_TIMEOUT_SEC (default 30s); on timeout, calls
   `reranker.mark_degraded(...)`.

3. **Rerank entrypoints block on stalled load** — once preload is hung,
   every `engine.search()` calls `rerank_with_modality_fusion()` which
   calls `rerank()` which calls `get_reranker()` which blocks on the same
   stalled load. Guard: rerank functions check `reranker.is_degraded()`
   and return original ordering instead.
"""
from __future__ import annotations

import os
import threading
import time

import pytest


# ---------------------------------------------------------------------------
# Bug #1: os.WNOHANG missing on Windows
# ---------------------------------------------------------------------------


def test_reap_children_no_crash_when_wnohang_missing(monkeypatch):
    """On Windows, os has no WNOHANG attribute. _reap_children must return
    cleanly instead of raising AttributeError, otherwise _backlog_drainer
    crashes on every boot for every Windows user.
    """
    from truememory import mcp_server as ms

    # Simulate the Windows environment: remove WNOHANG from os if present.
    monkeypatch.delattr(os, "WNOHANG", raising=False)

    # Must not raise.
    ms._reap_children()


# ---------------------------------------------------------------------------
# Bug #2 + #3: reranker degraded-mode fallback
# ---------------------------------------------------------------------------


@pytest.fixture
def _reset_reranker_degraded():
    """Reset the module-level degraded flag around each test so prior tests
    don't pollute state."""
    from truememory import reranker as rr
    original = rr._load_failed
    rr._load_failed = False
    yield
    rr._load_failed = original


def test_is_degraded_starts_false(_reset_reranker_degraded):
    from truememory import reranker as rr
    assert rr.is_degraded() is False


def test_mark_degraded_sets_flag(_reset_reranker_degraded):
    from truememory import reranker as rr
    rr.mark_degraded("test reason")
    assert rr.is_degraded() is True


def test_rerank_returns_original_ordering_when_degraded(_reset_reranker_degraded):
    """Once degraded, rerank() must NOT call get_reranker — that would
    block on the same stalled load that caused the degraded mark. It must
    return the candidates in their original order (truncated to top_k).
    """
    from truememory import reranker as rr

    candidates = [
        {"content": f"doc {i}", "rrf_score": 1.0 / (i + 1)}
        for i in range(5)
    ]
    rr.mark_degraded("simulated stall")

    out = rr.rerank("query", candidates, top_k=3)

    assert len(out) == 3, "top_k must be honored in degraded mode"
    assert [r["content"] for r in out] == ["doc 0", "doc 1", "doc 2"], (
        "Degraded mode must preserve original input ordering — the caller's "
        "RRF/vector ranking is the best signal available without a reranker."
    )
    # No rerank_score key must appear — proves get_reranker was never called.
    for r in out:
        assert "rerank_score" not in r


def test_rerank_with_modality_fusion_returns_original_when_degraded(
    _reset_reranker_degraded,
):
    from truememory import reranker as rr

    candidates = [
        {"content": "a", "modality": "conversation", "rrf_score": 0.9},
        {"content": "b", "modality": "episode", "rrf_score": 0.5},
        {"content": "c", "modality": "fact", "rrf_score": 0.3},
    ]
    rr.mark_degraded("simulated stall")

    out = rr.rerank_with_modality_fusion("why did X happen", candidates, top_k=2)

    assert len(out) == 2
    assert [r["content"] for r in out] == ["a", "b"]


# ---------------------------------------------------------------------------
# Bug #2: preload watchdog marks degraded on timeout
# ---------------------------------------------------------------------------


def test_preload_watchdog_marks_degraded_on_timeout(
    monkeypatch, _reset_reranker_degraded,
):
    """If get_reranker hangs longer than TRUEMEMORY_RERANKER_TIMEOUT_SEC,
    the watchdog must call mark_degraded so search calls fall back instead
    of blocking forever.

    Strategy: monkey-patch get_reranker with a function that sleeps past the
    timeout, set the timeout to a small value, call _preload_models, then
    poll is_degraded() until the watchdog fires.
    """
    from truememory import mcp_server as ms
    from truememory import reranker as rr

    # Force a very small timeout so the test finishes fast.
    monkeypatch.setattr(ms, "_RERANKER_LOAD_TIMEOUT_SEC", 1)

    # Replace get_reranker with a hang.
    def _hang(*_a, **_k):
        time.sleep(30)  # well past the 1s timeout

    monkeypatch.setattr(rr, "get_reranker", _hang)

    # Prevent the embedding-model branch from doing real I/O.
    monkeypatch.setenv("TRUEMEMORY_LAZY_MODELS", "")  # ensure preload runs

    # Stub the embedding loader so it returns immediately.
    import truememory.vector_search as vs
    monkeypatch.setattr(vs, "get_model", lambda *_a, **_k: None)
    monkeypatch.setattr(ms, "_get_memory", lambda: None)

    ms._preload_models()

    # Watchdog must fire within timeout + small margin.
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if rr.is_degraded():
            break
        time.sleep(0.05)

    assert rr.is_degraded(), (
        "Watchdog did not mark reranker degraded within 3s — the stalled "
        "preload would have blocked every subsequent search call indefinitely."
    )


def test_preload_watchdog_does_not_mark_degraded_on_fast_load(
    monkeypatch, _reset_reranker_degraded,
):
    """If get_reranker returns quickly, the watchdog must NOT mark degraded.
    Otherwise every successful boot would falsely report degraded mode.
    """
    from truememory import mcp_server as ms
    from truememory import reranker as rr

    monkeypatch.setattr(ms, "_RERANKER_LOAD_TIMEOUT_SEC", 5)
    monkeypatch.setattr(rr, "get_reranker", lambda *_a, **_k: None)

    import truememory.vector_search as vs
    monkeypatch.setattr(vs, "get_model", lambda *_a, **_k: None)
    monkeypatch.setattr(ms, "_get_memory", lambda: None)

    ms._preload_models()

    # Give the watchdog time to either fire (false positive) or finish cleanly.
    time.sleep(0.5)

    assert not rr.is_degraded(), (
        "Watchdog fired on a fast load — this would make every boot report "
        "degraded mode, defeating the purpose of the fallback."
    )
