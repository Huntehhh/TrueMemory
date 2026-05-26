#!/usr/bin/env python3
"""
UserPromptSubmit Hook — Turn-Based Memory Recall + Buffer
==========================================================

Fires on every user message submission. Three concurrent jobs:

1. **Buffer** — append the user message to a per-session JSONL buffer
   for diagnostic recovery if the live transcript is corrupted.
2. **Incremental extraction** — every N seconds (default 4h), spawn a
   background ingestion run so memories captured during long sessions
   land in the DB without waiting for SessionEnd.
3. **Turn-based recall** — always run a fresh search against the
   user's full memory corpus, dedup against memories already injected
   this session, and inject up to N novel results into Claude's context.

Why turn-based recall (vs. the old regex-only gate)
---------------------------------------------------
The previous implementation only injected when the prompt matched a
narrow ``what is | who is | do you remember | …`` regex AND was under
500 chars. Long, conversational prompts (the user's actual style)
silently bypassed recall, leaving SessionStart's blanket 25-fact
injection as the only memory ever surfaced. Per-session dedup
prevents re-injecting the same fact twice; the ``visibility log``
records every fire (including skips) so debugging the gate doesn't
require code archaeology.

Input (stdin JSON):
    {"session_id": "...", "prompt": "...", "transcript_path": "..."}

Output (stdout JSON, only when injecting):
    {"additionalContext": "<truememory-recall>...</truememory-recall>"}

Skip-reason taxonomy (recorded in ~/.truememory/injections.log):
    - prompt_too_short      prompt < TRUEMEMORY_TURN_RECALL_MIN_LEN chars
    - prompt_too_long       prompt > TRUEMEMORY_TURN_RECALL_MAX_LEN chars (paste)
    - trivial_reply         emoji-only, one-word ack, or all-pleasantry prompt
    - recall_disabled       TRUEMEMORY_TURN_RECALL_ENABLED=0
    - search_unavailable    truememory.Memory import or engine search failed
    - search_returned_none  search ran but produced zero candidates
    - no_novel_memories     all candidates were already injected this session
    - low_signal            top candidate's score < TRUEMEMORY_TURN_RECALL_SCORE_FLOOR
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

# fcntl isn't available on Windows — buffer writes degrade to bare append.
try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False


# ─────────────────────────────────────────────────────────────────
# Configuration (env-overridable)
# ─────────────────────────────────────────────────────────────────

BUFFER_DIR = Path(os.environ.get(
    "TRUEMEMORY_BUFFER_DIR",
    str(Path.home() / ".truememory" / "buffers"),
))
RETENTION_DAYS = int(os.environ.get("TRUEMEMORY_BUFFER_RETENTION_DAYS", "7"))
MAX_BUFFER_SIZE = int(os.environ.get("TRUEMEMORY_BUFFER_MAX_BYTES", str(10 * 1024 * 1024)))

# Turn-based recall tunables ------------------------------------------------
RECALL_ENABLED = os.environ.get("TRUEMEMORY_TURN_RECALL_ENABLED", "1") not in ("0", "false", "False", "")
RECALL_LIMIT = max(1, int(os.environ.get("TRUEMEMORY_TURN_RECALL_LIMIT", "5")))
RECALL_OVERFETCH = max(RECALL_LIMIT, int(os.environ.get("TRUEMEMORY_TURN_RECALL_OVERFETCH", "20")))
RECALL_MIN_LEN = max(1, int(os.environ.get("TRUEMEMORY_TURN_RECALL_MIN_LEN", "15")))
RECALL_MAX_LEN = int(os.environ.get("TRUEMEMORY_TURN_RECALL_MAX_LEN", "4000"))
RECALL_SCORE_FLOOR = float(os.environ.get("TRUEMEMORY_TURN_RECALL_SCORE_FLOOR", "0"))


# ─────────────────────────────────────────────────────────────────
# CLI args (installer threads --user / --db through every hook)
# ─────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--user", default=os.environ.get("TRUEMEMORY_USER_ID", ""))
    p.add_argument("--db", default=os.environ.get("TRUEMEMORY_DB_PATH", ""))
    args, _ = p.parse_known_args()
    return args


# ─────────────────────────────────────────────────────────────────
# Email capture (unchanged)
# ─────────────────────────────────────────────────────────────────

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def _try_capture_email(prompt: str) -> None:
    """If the user typed an email and config has no email, save it."""
    try:
        config_path = Path.home() / ".truememory" / "config.json"
        if not config_path.exists():
            return
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("email"):
            return
        match = _EMAIL_RE.search(prompt)
        if not match:
            return
        config["email"] = match.group(0)
        tmp = config_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(config, indent=2), encoding="utf-8")
        tmp.rename(config_path)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────
# Turn-based recall — the meat of the rewrite
# ─────────────────────────────────────────────────────────────────

# One-word / emoji acknowledgements that should never trigger recall.
_TRIVIAL_REPLIES = frozenset({
    "ok", "okay", "k", "kk", "yes", "y", "yep", "yeah", "yup",
    "no", "n", "nope", "nah",
    "thanks", "thx", "ty", "thank you", "cheers",
    "cool", "nice", "great", "good", "perfect", "awesome", "sweet",
    "sure", "fine", "alright",
    "done", "got it", "gotcha", "understood",
    "continue", "next", "go", "go on", "proceed",
    "hi", "hello", "hey", "yo", "sup",
    "bye", "later",
})

# Emoji-only / punctuation-only check: prompt has zero word chars.
_HAS_WORD_RE = re.compile(r"\w", re.UNICODE)

# Word-level pleasantry vocabulary. A prompt whose every token is in this
# set carries no information need, regardless of length or word order
# ("thank you so much", "ok sounds good thanks man"). More robust than a
# phrase regex — order-independent and trivial to extend. Tokens are
# lowercased and stripped of surrounding punctuation before lookup.
_PLEASANTRY_WORDS = frozenset({
    # gratitude
    "thanks", "thank", "thx", "ty", "cheers", "appreciated", "appreciate",
    # affirmation
    "ok", "okay", "k", "kk", "yes", "y", "yep", "yeah", "yup", "sure",
    "fine", "alright", "right",
    # negation acks
    "no", "n", "nope", "nah",
    # praise / acknowledgement
    "cool", "nice", "great", "good", "perfect", "awesome", "sweet",
    "excellent", "beautiful", "lovely", "lgtm",
    "done", "got", "gotcha", "understood", "makes", "sense", "noted",
    "sounds", "looks", "seems",
    # continuation
    "continue", "next", "go", "proceed", "onward", "carry",
    # greeting / signoff
    "hi", "hello", "hey", "yo", "sup", "bye", "later", "morning", "evening",
    # filler / vocatives
    "so", "very", "much", "a", "lot", "again", "please", "you", "u",
    "man", "dude", "bud", "buddy", "mate", "bro", "friend", "boss",
    "will", "do", "it", "that", "this", "all", "and", "the", "for",
    "really", "totally", "definitely", "absolutely", "indeed",
})

_TOKEN_RE = re.compile(r"[a-zA-Z']+")


def _is_all_pleasantry(stripped: str) -> bool:
    """True when every alphabetic token is a pleasantry/filler word.

    Guards against the empty-token case (handled separately as emoji-only).
    """
    tokens = _TOKEN_RE.findall(stripped.lower())
    if not tokens:
        return False
    return all(t in _PLEASANTRY_WORDS for t in tokens)


def _classify_prompt(prompt: str) -> str | None:
    """Return a skip-reason if the prompt should bypass recall, else None.

    Cheap gates first — bail before touching the search engine for prompts
    that obviously can't benefit from memory injection (one-word
    acknowledgements, pure emoji reactions, pleasantries, runaway pastes).
    """
    stripped = prompt.strip()
    if len(stripped) > RECALL_MAX_LEN:
        # Massive pastes — log lines, full files, dataset rows — almost
        # never benefit from semantic recall and waste a search call.
        return "prompt_too_long"
    if not _HAS_WORD_RE.search(stripped):
        # Emoji-only / punctuation-only reaction.
        return "trivial_reply"
    if stripped.lower().rstrip(".!?") in _TRIVIAL_REPLIES or _is_all_pleasantry(stripped):
        return "trivial_reply"
    if len(stripped) < RECALL_MIN_LEN:
        return "prompt_too_short"
    return None


def _compute_recall(
    prompt: str,
    session_id: str,
    user_id: str,
    db_path: str,
) -> dict:
    """Run search, dedup against this session, format injection context.

    Returns a result dict with one of two shapes:

        {"injected": True,  "content": str, "memory_count": int,
         "memory_ids": [int, …], "top_score": float}

        {"injected": False, "reason": str, "extra": {…}}
    """
    if not RECALL_ENABLED:
        return {"injected": False, "reason": "recall_disabled", "extra": {}}

    skip_reason = _classify_prompt(prompt)
    if skip_reason:
        return {
            "injected": False,
            "reason": skip_reason,
            "extra": {"prompt_len": len(prompt.strip())},
        }

    # Import lazily so a missing / broken truememory install can't crash
    # the hook before the buffer write has had a chance to land.
    try:
        from truememory.client import Memory
        from truememory.ingest.hooks import _session_dedup
    except Exception as exc:
        return {
            "injected": False,
            "reason": "search_unavailable",
            "extra": {"error": str(exc)[:200]},
        }

    memory = None
    try:
        memory = Memory(path=db_path or None)
        # FAST PATH — turn recall runs synchronously in the prompt-submit
        # critical path, so it must NOT pay for the cross-encoder reranker
        # (~8-15s on Pro tier) or the surprise boost. Measured on a Pro
        # install: full Memory.search() = ~15s cold / ~8s warm; the
        # skip-reranker hybrid path = ~8ms. session_start.py uses the same
        # flags for exactly this reason. The reranker's quality gain isn't
        # worth a 15s stall before every Claude response — the hybrid
        # FTS+vector ranking is already good enough for turn-level recall.
        engine = memory._engine
        results = engine.search(
            prompt,
            limit=RECALL_OVERFETCH,
            _skip_reranker=True,
            _skip_surprise_boost=True,
        )
        if user_id:
            results = [r for r in results if r.get("sender", "") == user_id]
    except Exception as exc:
        log.exception("turn_recall: search failed session=%s", session_id)
        return {
            "injected": False,
            "reason": "search_unavailable",
            "extra": {"error": str(exc)[:200]},
        }
    finally:
        # Results are fully materialized dicts — release the DB connection
        # now rather than waiting for process teardown (keeps WAL handles
        # from lingering if the host process is ever reused).
        if memory is not None:
            try:
                memory._engine.close()
            except Exception:
                pass

    if not results:
        return {
            "injected": False,
            "reason": "search_returned_none",
            "extra": {"query_len": len(prompt.strip())},
        }

    already = _session_dedup.already_injected(session_id)
    top_score = _result_score(results[0]) if results else 0.0

    if top_score < RECALL_SCORE_FLOOR:
        return {
            "injected": False,
            "reason": "low_signal",
            "extra": {"top_score": top_score, "floor": RECALL_SCORE_FLOOR},
        }

    novel: list[dict] = []
    seen_content: set[str] = set()
    for r in results:
        rid = r.get("id")
        if rid is None or rid in already:
            continue
        content = (r.get("content") or "").strip()
        if not content:
            continue
        # Content-level dedup against this turn's own pick list — catches
        # the (rare) case where two DB rows hold near-identical text.
        norm = _normalize_for_dedup(content)
        if norm in seen_content:
            continue
        seen_content.add(norm)
        novel.append(r)
        if len(novel) >= RECALL_LIMIT:
            break

    if not novel:
        return {
            "injected": False,
            "reason": "no_novel_memories",
            "extra": {
                "candidates":      len(results),
                "already_injected": len(already),
                "top_score":        top_score,
            },
        }

    content = _format_recall_context(novel)
    return {
        "injected":     True,
        "content":      content,
        "memory_count": len(novel),
        "memory_ids":   [r.get("id") for r in novel],
        "top_score":    top_score,
    }


def _result_score(r: dict) -> float:
    """Return whichever score field the engine populated, defaulting to 0."""
    for key in ("score", "rrf_score", "raw_score"):
        v = r.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return 0.0


_DEDUP_WS_RE = re.compile(r"\s+")


def _normalize_for_dedup(content: str) -> str:
    return _DEDUP_WS_RE.sub(" ", content.lower()).strip().rstrip(".")


def _format_recall_context(memories: list[dict]) -> str:
    """Wrap memories in the recall envelope Claude will see in context."""
    lines = [
        "<truememory-recall>",
        "## TrueMemory — Relevant memories for this turn",
        "These are facts surfaced from TrueMemory based on what you just asked.",
        "Newly surfaced this turn (deduped against earlier injections):",
        "",
    ]
    for r in memories:
        content = (r.get("content") or "").strip()
        if content:
            lines.append(f"- {content}")
    lines.append("</truememory-recall>")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# Incremental extraction trigger (unchanged in spirit, lifted out
# of main() so the control flow reads top-down)
# ─────────────────────────────────────────────────────────────────

def _maybe_trigger_incremental_extraction(
    transcript_path: str,
    session_id: str,
    user_id: str,
    db_path: str,
) -> None:
    """If enough time has passed, spawn a background ingestion run."""
    if not transcript_path or not Path(transcript_path).exists():
        return
    try:
        interval = int(os.environ.get("TRUEMEMORY_INCREMENTAL_INTERVAL", "14400"))
        from truememory.ingest.hooks._shared import (
            MARKER_PATH, mark_extracted, should_extract,
        )
        if not should_extract(interval):
            try:
                elapsed = (
                    time.time() - MARKER_PATH.stat().st_mtime
                    if MARKER_PATH.exists() else 0.0
                )
            except OSError:
                elapsed = 0.0
            log.debug(
                "incremental: skipped session=%s elapsed=%.0fs interval=%ds",
                session_id, elapsed, interval,
            )
            return

        log.info(
            "incremental: trigger fired session=%s interval=%ds transcript=%s",
            session_id, interval, transcript_path,
        )
        from truememory.ingest.hooks.stop import (
            LOG_DIR, TRACE_DIR, _has_enough_messages, _run_background_ingestion,
        )
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        if _has_enough_messages(transcript_path, 5):
            _run_background_ingestion(transcript_path, session_id, user_id, db_path)
            mark_extracted()
    except Exception:
        log.exception(
            "incremental: background ingestion failed session=%s transcript=%s",
            session_id, transcript_path,
        )


# ─────────────────────────────────────────────────────────────────
# Buffer write (unchanged)
# ─────────────────────────────────────────────────────────────────

def _sanitize_session_id(session_id: str) -> str:
    safe = "".join(c for c in session_id if c.isalnum() or c in "-_")[:64]
    return safe or "unknown"


def buffer_message(session_id: str, prompt: str) -> None:
    """Append a user message to the session buffer file (with file locking)."""
    BUFFER_DIR.mkdir(parents=True, exist_ok=True)
    try:
        BUFFER_DIR.chmod(0o700)
    except OSError:
        pass

    buffer_file = BUFFER_DIR / f"{_sanitize_session_id(session_id)}.jsonl"

    try:
        if buffer_file.exists() and buffer_file.stat().st_size > MAX_BUFFER_SIZE:
            rotated = buffer_file.with_suffix(f".{int(time.time())}.jsonl")
            buffer_file.rename(rotated)
    except OSError:
        pass

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "role": "user",
        "content": prompt,
    }

    with open(buffer_file, "a", encoding="utf-8") as f:
        if _HAS_FCNTL:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(json.dumps(entry) + "\n")
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except OSError:
                f.write(json.dumps(entry) + "\n")
        else:
            f.write(json.dumps(entry) + "\n")


def _prune_old_buffers() -> None:
    if not BUFFER_DIR.exists():
        return
    cutoff = time.time() - (RETENTION_DAYS * 86400)
    for path in BUFFER_DIR.iterdir():
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    try:
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError, ValueError):
        return
    # Defend against valid-but-non-object JSON (a bare list / string / number
    # would make the .get() calls below raise AttributeError and crash the
    # hook, violating the never-raise contract).
    if not isinstance(input_data, dict):
        return

    prompt = (input_data.get("prompt") or "").strip()
    session_id = input_data.get("session_id") or "unknown"
    transcript_path = input_data.get("transcript_path") or ""

    if not prompt:
        return

    # 1. Buffer (always — diagnostic, can never block). Skip ultra-short
    #    pings — they bloat the buffer without aiding recovery. Pruning is
    #    probabilistically throttled (~5% of turns) so the per-prompt path
    #    doesn't pay an O(N) directory scan every single time.
    if len(prompt) >= 3:
        try:
            buffer_message(session_id, prompt)
            if random.random() < 0.05:
                _prune_old_buffers()
        except Exception:
            pass

    # 2. Email auto-capture
    _try_capture_email(prompt)

    # 3. Incremental extraction (every ~4h by default)
    _maybe_trigger_incremental_extraction(
        transcript_path, session_id, args.user, args.db,
    )

    # 4. Turn-based recall — ALWAYS runs, always writes to visibility log,
    #    only emits additionalContext when novel memories surface.
    result = _compute_recall(prompt, session_id, args.user, args.db)

    if result["injected"]:
        print(json.dumps({"additionalContext": result["content"]}))
        _log_recall(
            session_id=session_id,
            prompt=prompt,
            content=result["content"],
            memory_count=result["memory_count"],
            action="injected",
            extra={
                "memory_ids": result["memory_ids"],
                "top_score":  result["top_score"],
            },
        )
        try:
            from truememory.ingest.hooks import _session_dedup
            _session_dedup.record_injection(session_id, result["memory_ids"])
        except Exception:
            log.exception("turn_recall: dedup record failed session=%s", session_id)
    else:
        _log_recall(
            session_id=session_id,
            prompt=prompt,
            content="",
            memory_count=0,
            action="skipped",
            extra={"reason": result["reason"], **result.get("extra", {})},
        )


def _log_recall(
    *,
    session_id: str,
    prompt: str,
    content: str,
    memory_count: int,
    action: str,
    extra: dict,
) -> None:
    """Write the visibility log record. Never raises."""
    try:
        from truememory.ingest.hooks._injection_log import write_injection
        write_injection(
            hook="user_prompt_submit",
            session_id=session_id,
            content=content,
            query=prompt[:500],
            memory_count=memory_count,
            action=action,
            extra=extra,
        )
    except Exception:
        log.exception("turn_recall: visibility log write failed session=%s", session_id)


if __name__ == "__main__":
    main()
