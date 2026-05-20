# TrueMemory — Upstream Suggestions

Forward-looking ideas surfaced during Hunter's local fork work that may be worth considering for upstream. Not a roadmap commitment — just a brain-dump of architectural options + their tradeoffs so they're not lost.

Maintained on `local/instrumentation-diag` (Hunter's local-only branch). Pushed to fork for visibility on the GitHub UI.

---

## 1. `http` hook type as an alternative to subprocess-spawn

### What it is

Claude Code natively supports 5 hook types: `command`, `http`, `mcp_tool`, `prompt`, `agent`. TrueMemory currently uses `command` exclusively — every hook invocation spawns a fresh Python interpreter, imports the hook module, runs ~150-300ms of work, and exits.

The `http` type instead has Claude Code **POST the hook payload as JSON to a localhost URL**. The hook handler runs inside a long-lived service that owns the route. Zero subprocess. Zero Python startup cost per fire.

```jsonc
// command form (current)
{
  "type": "command",
  "command": "python.exe",
  "args": ["-m", "truememory.ingest.hooks.user_prompt_submit"]
}

// http form (proposed)
{
  "type": "http",
  "url": "http://localhost:8765/hook/user_prompt_submit"
}
```

### Cascading wins (honest framing)

1. **Windows terminal-flash elimination** — fundamental, not a workaround. Claude Code's Node spawn omits `windowsHide:true` (known issue, 8+ tickets, #17230 closed not-planned). HTTP hooks don't spawn anything. No flash possible.
2. **No Defender ASR risk** — no fresh-hash binary launches. Currently Hunter has to whitelist `headless-tty.exe` to bypass ASR rule 01443614. With HTTP, nothing to whitelist.
3. **~150-300ms saved per hook fire** — Python interpreter startup eliminated. Modest but real, multiplies across many hook fires per session.
4. **State persistence available IF needed** — long-lived handler could keep the embedder + reranker warm. This matters less than I initially thought: the hot path of `UserPromptSubmit` (buffer write + email regex + `_detect_recall` regex) doesn't touch the model. Model load only fires on explicit-recall matches OR turn-based injection (rare). So the "warm models" benefit is real for `search_deep`-heavy workloads, marginal for the common case.
5. **Single rate-limiter for quota** — IF the daemon does its own LLM calls, it can smooth quota usage across days instead of per-burst.

### Trade-offs

- **Real refactor cost.** All current hook scripts (`session_start.py`, `user_prompt_submit.py`, `stop.py`) become route handlers in a FastAPI / aiohttp / Flask app. Logic is reusable; transport is the change.
- **Process lifecycle management.** The daemon needs to start on boot (Windows: VBS launcher; macOS: LaunchAgent; Linux: systemd user unit), survive Claude Code restarts, and handle its own crash recovery.
- **Port allocation.** Default port + override mechanism (env var or config.json).
- **Auth.** Localhost-only is the natural posture; bearer-token + allowlist is the hardening path.
- **Backwards compat.** Existing `command`-form users keep working. New users get the `http` shape via an `--install-mode http` flag on `truememory-ingest install`, or a setting in `~/.truememory/config.json`.

### Urgency assessment (honest)

Not on fire. Hunter has worked around the practical blockers (Headless-TTY + ASR exclusion). The HTTP architecture is the **clean** fix, not the **urgent** fix. Worth considering when there's appetite for an infrastructure pass.

---

## 2. Per-call telemetry chokepoint

### Already shipped on `local/instrumentation-diag`

Commit `0c1e25d` — `_complete_claude_cli` in `models.py` writes a JSON line to `~/.truememory/claude-cli-usage.jsonl` on every invocation. Captures: timestamp, model, caller (stack-walked to first frame outside `models.py`), prompt/response chars, duration_ms, exit_code, PID, error string. Crash-tolerant (try/finally), never raises (broad except inside the logger).

This let Hunter pin the rate-limit source within minutes — turned out to be background ingestion calls (~338 sessions/day × 10.5 chunks each ≈ 3,500 Haiku calls/day routing through Claude CLI subscription).

### Worth upstreaming?

The JSONL is purely additive — opt-in via the existing call site, no new env vars required. Could be gated behind `TRUEMEMORY_USAGE_LOG=1` if upstream wants it off-by-default. The 10K-row-per-day volume isn't a disk concern for typical use.

---

## 3. `TRUEMEMORY_BACKLOG_BATCH_SIZE` env var (shipped)

Commit on `local/instrumentation-diag`. Renames hardcoded `_DRAIN_CAP = 3` in `session_start.py` to `_BACKLOG_BATCH_SIZE` and exposes via env var. Default 3 (unchanged behavior). Set to 1 to serialize backlog drain — useful when many SessionStarts arrive in rapid succession and the multiplicative drain fan-out (N × 3 × ~10 Haiku calls each) spikes the per-minute rate window.

Upstreamable cleanly — pure addition, default behavior preserved, single rename for clarity.

---

## 4. Hook stdin field that propagates main-session vs sub-agent context

Already implemented via `agent_id` field in stdin (canonical Claude Code feature). Hunter's local hooks check this + `/subagents/` in `transcript_path` to skip ingestion when invoked from a sub-agent context (orchestrator-generated prompts, not real user input).

PR #357 implements this in `_shared.is_subagent_invocation`. The design note in `tests/test_turn_based_injection.py` explains the rationale + offers an `TRUEMEMORY_SKIP_SUBAGENT_HOOKS` env var as a backout if upstream prefers canonical "sub-agents fire hooks normally" behavior.

---

## 5. Claude CLI as a priority-1 `_build_llm_fn` provider (PR #357)

In `mcp_server.py::_build_llm_fn`, Claude CLI now wins priority over the API-key providers. Zero cash spend for Pro-tier subscription users who have Claude Code installed. Opt-out via `TRUEMEMORY_DISABLE_CLAUDE_CLI=1` env var for users who want the API-key path (centralized billing, paid-plan consistency).

PR body has the full migration note + behavior-change call-out.

---

## Notes on local-only deviations from upstream

These are local-fork-only and explicitly NOT for upstream PR:

- **`local/haiku-default-claude-cli`** — defaults the Claude CLI model to `claude-haiku-4-5` instead of inheriting the user's Claude Code default (typically Opus 4.7[1m]). ~600× cheaper per extraction call. Hunter-specific choice that may not generalize.
- **`local/instrumentation-diag`** — Hunter's always-on diagnostic overlay. Heavy `_dlog` instrumentation across `engine.py`, `mcp_server.py`, `ingest/*`. Not for upstream — it's a debug aid.

---

---

## Hunter's open actions (not for Josh — personal tracker)

- [ ] **Send the Letta consolidation email to Josh.** Draft at [`letta-email-to-josh.md`](./letta-email-to-josh.md). Need to fill in Josh's email address, attach the 5 markdown files listed at the bottom of the draft (`~/.claude/_BEST-practices-GLOBAL/memory-systems/letta-consolidation-shippable-plan.md` + 4 `letta-impl-A/B/C/D-*.md` siblings), copy body into Gmail, send.
- [ ] **Set `TRUEMEMORY_BACKLOG_BATCH_SIZE=1`** in env (system or `~/.truememory/config.json`) to smooth cascade-drain bursts. See §3 above.

---

_Maintained by Hunter Casillas (`@Huntehhh`). Drop a comment in this file or open an issue on the fork if any of these are worth a real conversation._
