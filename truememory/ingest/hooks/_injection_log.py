"""Injection visibility log — captures what each hook injected into Claude's context.

Every hook fire writes two artifacts:

1. ``~/.truememory/last-injection.json`` — overwritten each call. Single source
   of truth for "what's the most recent thing TrueMemory put in context".
   Consumed by the ccstatusline widget and the /tm-log slash command.

2. ``~/.truememory/injections.log`` — append-only JSONL history. Rotated when
   it crosses ``INJECTIONS_LOG_MAX_BYTES`` (default 5 MB).

Schema (both files use the same record shape):

    {
      "hook": "session_start" | "user_prompt_submit" | "stop",
      "timestamp": "2026-05-21T19:30:00+00:00",
      "session_id": "abc-123",
      "char_count": 412,
      "memory_count": 5,
      "query": "user preferences favorites…",
      "preview": "Prefers bun over npm…",
      "full_content": "<truememory-recall>…</truememory-recall>",
      "action": "injected" | "queued_ingestion" | "skipped",
      "extra": {…}
    }

Designed to never raise — visibility is best-effort. A failure here must
never block the hook's primary job.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LAST_INJECTION_PATH = Path(os.environ.get(
    "TRUEMEMORY_LAST_INJECTION_PATH",
    str(Path.home() / ".truememory" / "last-injection.json"),
))
INJECTIONS_LOG_PATH = Path(os.environ.get(
    "TRUEMEMORY_INJECTIONS_LOG_PATH",
    str(Path.home() / ".truememory" / "injections.log"),
))
INJECTIONS_LOG_MAX_BYTES = int(os.environ.get(
    "TRUEMEMORY_INJECTIONS_LOG_MAX_BYTES",
    str(5 * 1024 * 1024),
))

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _make_preview(content: str, max_chars: int = 80) -> str:
    """Strip XML tags + collapse whitespace, return first ``max_chars``."""
    if not content:
        return ""
    stripped = _TAG_RE.sub("", content)
    stripped = _WS_RE.sub(" ", stripped).strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[:max_chars - 1].rstrip() + "…"


def _rotate_if_needed() -> None:
    try:
        if INJECTIONS_LOG_PATH.exists() and INJECTIONS_LOG_PATH.stat().st_size > INJECTIONS_LOG_MAX_BYTES:
            rotated = INJECTIONS_LOG_PATH.with_suffix(".log.1")
            if rotated.exists():
                rotated.unlink()
            INJECTIONS_LOG_PATH.rename(rotated)
    except OSError:
        pass


def write_injection(
    *,
    hook: str,
    session_id: str = "",
    content: str = "",
    query: str | None = None,
    memory_count: int | None = None,
    action: str = "injected",
    extra: dict[str, Any] | None = None,
) -> None:
    """Record one hook-fire event. Never raises."""
    try:
        LAST_INJECTION_PATH.parent.mkdir(parents=True, exist_ok=True)

        record: dict[str, Any] = {
            "hook": hook,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "session_id": session_id or "unknown",
            "char_count": len(content),
            "memory_count": memory_count,
            "query": query,
            "preview": _make_preview(content),
            "full_content": content,
            "action": action,
            "extra": extra or {},
        }

        # PID-suffix the tmp file so two concurrent hooks (e.g., sub-agent
        # session_start overlapping the parent's user_prompt_submit) don't
        # clobber each other's in-flight write. Last writer's rename wins
        # cleanly — no corruption, no partial-file ever published.
        # Derive the tmp name from the configured path so a custom
        # TRUEMEMORY_LAST_INJECTION_PATH keeps its sibling tmp file.
        tmp = LAST_INJECTION_PATH.with_name(f"{LAST_INJECTION_PATH.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
        tmp.replace(LAST_INJECTION_PATH)

        _rotate_if_needed()
        # Keep full_content in the rolling log so the history popup can show
        # the complete injected memories un-truncated (the user asked for no
        # truncation). The 5 MB rotation cap keeps total size bounded — at
        # ~6 KB per session_start record that's still ~800+ records of
        # history before the oldest rolls off.
        with INJECTIONS_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass
