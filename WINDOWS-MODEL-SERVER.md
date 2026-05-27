# Windows support for the shared model server

## What this is

TrueMemory runs a **shared model server** (`truememory/model_server.py`) that
loads the embedding + reranker models **once** and serves every Claude chat /
MCP process over a local socket. Without it, each process loads its own copy of
the ~149M-param reranker — which on Windows meant **~9-minute cold loads and
search timeouts**, so Windows was forced into per-process mode via
`TRUEMEMORY_NO_MODEL_SERVER=1`.

The blocker: the server's transport was a **Unix domain socket** (`AF_UNIX`),
and `hasattr(socket, "AF_UNIX")` is **`False`** on Windows Python. The server
crashed at `socket.socket(socket.AF_UNIX, ...)` the instant it tried to bind.

This change **ports the transport to TCP loopback on Windows** while keeping
POSIX (macOS / Linux) byte-for-byte identical.

## What changed

Two files, transport-only. The wire protocol (4-byte big-endian length prefix +
pickle) is **unchanged** — only the socket family/address differ.

### `truememory/model_server.py`

- New `_USE_UNIX = hasattr(socket, "AF_UNIX") and sys.platform != "win32"` —
  single source of truth for the transport family.
- `run()` branches:
  - **POSIX** — `AF_UNIX` bound to `~/.truememory/model.sock` (exactly as before).
  - **Windows** — `AF_INET` bound to **`127.0.0.1`** (loopback only, never
    `0.0.0.0`) on an **OS-assigned ephemeral port** (`bind` to port 0). The
    chosen port is written atomically to `~/.truememory/model_server.port`
    **before** the PID file, so any client that sees the PID is guaranteed to
    also see a readable port file.
- `_idle_checker()` self-connects on the active transport to wake `accept()`.
- `_cleanup()` removes the port file (Windows) alongside the sock + pid files.
- `main()` now uses a cross-platform `_pid_is_alive()` (psutil → `tasklist`
  fallback) instead of the POSIX-only `os.kill(pid, 0)` idiom, and clears stale
  port/sock files on a dead-PID restart. An `atexit` hook drops endpoint files
  even on abrupt teardown.

### `truememory/model_client.py`

- Mirrors `_USE_UNIX`, `PORT_PATH`, `_LOOPBACK_HOST`.
- `_read_port()` — reads the Windows port file (None on POSIX).
- `_pid_is_alive()` — cross-platform liveness (same logic as the server).
- `_connect()` — opens `AF_UNIX → model.sock` (POSIX) or
  `AF_INET → 127.0.0.1:<port>` (Windows); raises `ConnectionError` if the
  Windows port file is missing (callers then auto-start the server).
- `_server_is_alive()` / `use_model_server()` / `ensure_server_running()` —
  liveness is now "**endpoint file present AND PID alive**". On POSIX the
  endpoint file is `model.sock`; on Windows it's `model_server.port`. The
  observable contract is identical to before on POSIX.
- `_start_server()` spawns the server **headless on Windows** —
  `creationflags = CREATE_NO_WINDOW (0x08000000) | DETACHED_PROCESS (0x8) |
  CREATE_NEW_PROCESS_GROUP (0x200)` — **no console window flash**. POSIX keeps
  `start_new_session` (setsid). The readiness poll waits for the port file on
  Windows (the sock file on POSIX).

### Port-discovery approach (and why)

**Bind to port 0; server writes the chosen port to a file; client reads it.**

Chosen over a fixed high port (e.g. 47100) because:

- **Collision-free** — the OS guarantees a free port; no "address already in
  use" if something else grabbed 47100, and multiple TrueMemory installs /
  Python envs on one box won't fight over a port.
- **Same readiness semantics as POSIX** — on POSIX the sock file appears only
  after a successful `bind()`; the port file appears only after a successful
  `bind()` too. "File exists" means "listener is up" on both platforms, so the
  client's existing file-poll start logic works unchanged.

## How to ENABLE it (Windows)

The port only makes the server **capable** of running on Windows. It does NOT
flip any switch — your environment still has `TRUEMEMORY_NO_MODEL_SERVER=1`
set, which forces per-process mode.

To turn the shared server ON:

1. Open `~/.claude.json` (`C:\Users\huntfat\.claude.json`).
2. Find the `truememory` MCP server entry. Its `env` block contains:
   ```json
   "env": { "TRUEMEMORY_NO_MODEL_SERVER": "1", ... }
   ```
3. **Remove** the `"TRUEMEMORY_NO_MODEL_SERVER": "1"` line (or set it to `"0"`).
4. **Fully restart Claude Code** so the MCP server relaunches with the new env.
   On startup, `mcp_server.py` calls `ensure_server_running()`, which spawns the
   headless TCP server; the first search arrives warm.

> This branch only changes the two transport files. **It does not touch your
> env, `~/.claude.json`, or any settings** — enabling is a manual, reversible
> one-line edit you make when you're ready.

Verify it's running after restart:

```powershell
# Port file should exist and a python.exe should own a 127.0.0.1 listener:
Get-Content "$env:USERPROFILE\.truememory\model_server.port"
Get-NetTCPConnection -LocalPort (Get-Content "$env:USERPROFILE\.truememory\model_server.port") |
  Where-Object State -eq Listen
# Server's own log:
Get-Content "$env:USERPROFILE\.truememory\model_server.stderr" -Tail 5
# Expect a line like: Model server started: pid=NNNNN endpoint=127.0.0.1:PPPPP idle_timeout=300s
```

## Security note — localhost TCP

The server binds to **`127.0.0.1` only** (the loopback interface), **never
`0.0.0.0`**. That means **no other host on the network can reach it** — only
processes on this same machine can connect.

The wire protocol is **pickle**, so any local process that can connect could
send a crafted payload. On a **single-user box** (Hunter's) this is fine — it's
the same trust boundary as the POSIX Unix-domain-socket version, which is
likewise reachable by any local process owned by the user. On a shared /
multi-user Windows host you would want an auth token or a different IPC
mechanism; that's out of scope here and noted for completeness.

## Rollback

This change is fully self-contained in the two transport files and is **inert
until you remove `TRUEMEMORY_NO_MODEL_SERVER=1`**. To roll back:

- **Fastest (no code change):** leave the env var in place — with
  `TRUEMEMORY_NO_MODEL_SERVER=1` set, `use_model_server()` returns `False`
  immediately and the new transport code never runs. You're back to per-process
  mode.
- **Full code revert:** the changes live only on branch `local/win-fixes`.
  ```powershell
  git -C S:\OPEN-SOURCE-REPOSITORIES\TrueMemory-winfixes checkout -- `
    truememory/model_server.py truememory/model_client.py
  # or drop the whole commit:
  git -C S:\OPEN-SOURCE-REPOSITORIES\TrueMemory-winfixes reset --hard HEAD~1
  ```
- **Clean up any stray endpoint files** (harmless, self-healing on next start):
  ```powershell
  Remove-Item "$env:USERPROFILE\.truememory\model_server.port" -ErrorAction SilentlyContinue
  Remove-Item "$env:USERPROFILE\.truememory\model_server.pid"  -ErrorAction SilentlyContinue
  ```

POSIX users are unaffected by both the change and the rollback — their
`AF_UNIX` path is byte-for-byte what it was.
