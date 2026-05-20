# Email to Josh — Letta-style nightly consolidation

**Action:** Copy the body below into a Gmail compose window. Attach the 5 markdown files listed at the bottom. Send.

---

**To:** _<Josh's email — fill in before sending>_
**Subject:** TrueMemory — Letta-style nightly consolidation: scoped plan + safety hole

---

## Body

Hey Josh —

Got pulled into a deep scope of what it'd take to add Letta-style nightly consolidation to TrueMemory. Half the infra is already there (`fact_timeline.superseded_by`, `summaries` table, `detect_contradictions()`, `build_summaries()`) — net new work is ~8-12 hrs, not the 1 week I initially guessed.

One thing worth flagging upfront: the current `detect_contradictions()` and `build_summaries()` both open with `DELETE FROM <table>` before rebuilding. Crash between DELETE and rebuild = silent permanent loss. One-line fix: `sqlite3 conn.backup()` snapshot before any DELETE. Plan attached spells it out.

4 supporting docs in the attachments — schema diff, pipeline integration, scheduling (SessionStart piggyback, no daemon/cron), eval + rollback safety. Strategic value: Letta has issue #3116 open since Dec 2025 proposing the same thing, hasn't shipped. TrueMemory could ship it first.

Separate but related: I just pushed `feat/turn-based-injection-and-claude-cli-llm` on my fork (`Huntehhh/TrueMemory`) — bundles turn-based memory injection on `UserPromptSubmit` + Claude CLI as priority-1 in `_build_llm_fn` so HyDE works on subscription auth without API keys. PR coming once you've had a chance to look at the consolidation plan, since the turn-based hook stack benefits from the consolidation pass running on the back end.

Happy to discuss / spin a branch myself if you want it off your plate — but figured this is core infrastructure that should sit with you. Up to you.

— Hunter

---

## Attachments to manually pick (5 files — all under `C:/Users/huntfat/.claude/_BEST-practices-GLOBAL/memory-systems/`)

1. `letta-consolidation-shippable-plan.md` — **primary plan, read this first**
2. `letta-impl-A-schema.md` — schema diff details + per-change migration risk
3. `letta-impl-B-pipeline.md` — file-level integration sketch with function signatures
4. `letta-impl-C-scheduling.md` — three scheduling options compared, recommend SessionStart piggyback
5. `letta-impl-D-eval-safety.md` — test fixture design + rollback recommendation
