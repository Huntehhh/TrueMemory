"""Client for the shared model server.

Provides drop-in replacements for get_model() and get_reranker() that
route inference to the shared model_server process. Auto-starts the server
on first request if not running.

The transport is platform-branched to match the server:

* POSIX (macOS / Linux) — connects to ~/.truememory/model.sock, a Unix
  domain socket (``AF_UNIX``). Unchanged from earlier versions.
* Windows — ``AF_UNIX`` is unavailable, so the client connects to a
  loopback TCP socket (``127.0.0.1``). It discovers the server's
  OS-assigned port by reading ~/.truememory/model_server.port, which the
  server writes on startup. On Windows the server is spawned headless
  (no console window).

Falls back to local model loading if the server cannot be reached.
Set TRUEMEMORY_NO_MODEL_SERVER=1 to force local loading.
"""

import logging
import os
import pickle
import platform
import plistlib
import shutil
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_TRUEMEMORY_DIR = Path.home() / ".truememory"
SOCK_PATH = _TRUEMEMORY_DIR / "model.sock"
PID_PATH = _TRUEMEMORY_DIR / "model_server.pid"
# Windows-only: server publishes its chosen loopback TCP port here. Must
# match model_server.PORT_PATH. Unused on POSIX.
PORT_PATH = _TRUEMEMORY_DIR / "model_server.port"

# Single source of truth for transport family — mirrors model_server.
_USE_UNIX = hasattr(socket, "AF_UNIX") and sys.platform != "win32"
_LOOPBACK_HOST = "127.0.0.1"

_HEADER_FMT = ">I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

_SERVER_START_TIMEOUT = 30.0
_REQUEST_TIMEOUT = 120.0

_APP_BUNDLE_PATH = _TRUEMEMORY_DIR / "TrueMemory.app"
_APP_EXECUTABLE = _APP_BUNDLE_PATH / "Contents" / "MacOS" / "TrueMemory"
_LSREGISTER = (
    "/System/Library/Frameworks/CoreServices.framework"
    "/Frameworks/LaunchServices.framework/Support/lsregister"
)


def _ensure_app_bundle() -> str | None:
    """Create a macOS .app bundle so Activity Monitor shows our icon.

    Returns the path to the .app executable, or None on failure.
    """
    if platform.system() != "Darwin":
        return None

    real_python = os.path.realpath(sys.executable)

    if _APP_EXECUTABLE.exists():
        try:
            if os.path.samefile(_APP_EXECUTABLE, real_python):
                return str(_APP_EXECUTABLE)
        except OSError:
            pass

    try:
        if _APP_BUNDLE_PATH.exists():
            shutil.rmtree(_APP_BUNDLE_PATH)

        contents = _APP_BUNDLE_PATH / "Contents"
        macos_dir = contents / "MacOS"
        resources_dir = contents / "Resources"
        macos_dir.mkdir(parents=True)
        resources_dir.mkdir(parents=True)

        os.link(real_python, _APP_EXECUTABLE)

        # @executable_path/../lib/libpython*.dylib needs this symlink
        python_root = Path(real_python).parent.parent
        lib_dir = python_root / "lib"
        if lib_dir.exists():
            os.symlink(lib_dir, contents / "lib")

        try:
            from importlib.resources import files
            icon_data = files("truememory.assets").joinpath("AppIcon.icns").read_bytes()
            (resources_dir / "AppIcon.icns").write_bytes(icon_data)
        except Exception:
            pass

        plist = {
            "CFBundleExecutable": "TrueMemory",
            "CFBundleIconFile": "AppIcon",
            "CFBundleIdentifier": "network.sauron.truememory",
            "CFBundleName": "TrueMemory",
            "CFBundleDisplayName": "TrueMemory",
            "CFBundlePackageType": "APPL",
            "LSBackgroundOnly": True,
            "LSUIElement": True,
        }
        with open(contents / "Info.plist", "wb") as f:
            plistlib.dump(plist, f)

        if os.path.exists(_LSREGISTER):
            subprocess.run(
                [_LSREGISTER, "-f", str(_APP_BUNDLE_PATH)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )

        return str(_APP_EXECUTABLE)
    except OSError as e:
        if e.errno == 18:
            log.debug("Cannot hardlink across devices, skipping app bundle")
        else:
            log.debug("Failed to create app bundle: %s", e)
        return None
    except Exception as e:
        log.debug("Failed to create app bundle: %s", e)
        return None


def _read_port() -> int | None:
    """Read the server's loopback TCP port from PORT_PATH (Windows only).

    Returns the port int, or None if the file is missing / unreadable /
    malformed. Always None on POSIX (the AF_UNIX path needs no port).
    """
    if _USE_UNIX:
        return None
    try:
        return int(PORT_PATH.read_text().strip())
    except (FileNotFoundError, ValueError, OSError):
        return None


def _pid_is_alive(pid: int) -> bool:
    """Cross-platform 'is this PID running?' check.

    POSIX uses ``os.kill(pid, 0)``. Windows has no signal-0 semantics, so
    we probe via psutil when present, falling back to a ``tasklist`` query.
    On uncertainty we return False so a stale PID never wedges a restart.
    """
    if sys.platform == "win32":
        try:
            import psutil
            return psutil.pid_exists(pid)
        except ImportError:
            pass
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            return str(pid) in out.stdout
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False


def _server_is_alive() -> bool:
    if not PID_PATH.exists():
        return False
    try:
        pid = int(PID_PATH.read_text().strip())
    except (ValueError, OSError):
        return False
    return _pid_is_alive(pid)


# The endpoint-readiness file the client polls after spawning the server:
# the AF_UNIX socket on POSIX, the published TCP port file on Windows. The
# server creates each only after a successful bind(), so its appearance is
# the "listener is up" signal on both platforms.
_READY_PATH = SOCK_PATH if _USE_UNIX else PORT_PATH


def _spawn_kwargs() -> dict:
    """Platform-specific subprocess.Popen kwargs to fully detach the server.

    POSIX: ``start_new_session`` (setsid) so the child outlives the parent
    and ignores the parent's SIGINT.
    Windows: ``CREATE_NO_WINDOW | DETACHED_PROCESS`` so no console window
    flashes and the child isn't tied to the parent's console — Hunter hates
    console flashes. CREATE_NEW_PROCESS_GROUP keeps Ctrl-C in the parent
    from propagating to the server.
    """
    if sys.platform == "win32":
        # 0x08000000 CREATE_NO_WINDOW | 0x00000008 DETACHED_PROCESS
        # | 0x00000200 CREATE_NEW_PROCESS_GROUP
        flags = 0x08000000 | 0x00000008 | 0x00000200
        return {"creationflags": flags}
    return {"start_new_session": hasattr(os, "setsid")}


def _start_server() -> bool:
    """Start the model server as a detached, headless subprocess."""
    _TRUEMEMORY_DIR.mkdir(parents=True, exist_ok=True)

    if not _server_is_alive():
        # Clear artifacts left by a crashed/killed server so they don't
        # masquerade as a live endpoint.
        SOCK_PATH.unlink(missing_ok=True)
        PORT_PATH.unlink(missing_ok=True)
        PID_PATH.unlink(missing_ok=True)
    else:
        return True

    log.info("Starting model server...")
    app_exe = _ensure_app_bundle()
    if app_exe:
        cmd = [app_exe, "-m", "truememory.model_server"]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(sys.path)
    else:
        cmd = [sys.executable, "-m", "truememory.model_server"]
        env = None

    spawn_kwargs = _spawn_kwargs()
    try:
        _stderr_path = _TRUEMEMORY_DIR / "model_server.stderr"
        _stderr_fh = open(_stderr_path, "a")
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=_stderr_fh,
            env=env,
            **spawn_kwargs,
        )
    except Exception as e:
        log.warning("Failed to start model server: %s", e)
        if app_exe:
            try:
                _stderr_fh2 = open(_stderr_path, "a")
                subprocess.Popen(
                    [sys.executable, "-m", "truememory.model_server"],
                    stdout=subprocess.DEVNULL,
                    stderr=_stderr_fh2,
                    **spawn_kwargs,
                )
            except Exception as e2:
                log.warning("Fallback launch also failed: %s", e2)
                return False
        else:
            return False

    deadline = time.time() + _SERVER_START_TIMEOUT
    while time.time() < deadline:
        if _READY_PATH.exists():
            time.sleep(0.2)
            return True
        time.sleep(0.1)

    log.warning("Model server did not start within %.0fs", _SERVER_START_TIMEOUT)
    return False


def _connect() -> socket.socket:
    """Open a connected socket to the model server on the active transport.

    POSIX: AF_UNIX → SOCK_PATH. Windows: AF_INET → 127.0.0.1:<port> read
    from PORT_PATH. Raises ConnectionError if the Windows port file is
    missing (so callers treat it like a down server and auto-start).
    """
    if _USE_UNIX:
        family, addr = socket.AF_UNIX, str(SOCK_PATH)
    else:
        port = _read_port()
        if port is None:
            raise ConnectionError("model server port file not found")
        family, addr = socket.AF_INET, (_LOOPBACK_HOST, port)
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.settimeout(_REQUEST_TIMEOUT)
    try:
        sock.connect(addr)
    except OSError:
        sock.close()
        raise
    return sock


def _send_request(request: dict) -> dict:
    """Send a request to the model server and return the response."""
    sock = _connect()
    try:
        data = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
        header = struct.pack(_HEADER_FMT, len(data))
        sock.sendall(header + data)

        resp_header = _recv_exact(sock, _HEADER_SIZE)
        if not resp_header:
            raise ConnectionError("Server closed connection")
        resp_len = struct.unpack(_HEADER_FMT, resp_header)[0]
        resp_data = _recv_exact(sock, resp_len)
        if not resp_data:
            raise ConnectionError("Incomplete response")
        return pickle.loads(resp_data)
    finally:
        sock.close()


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _request_with_autostart(request: dict) -> dict:
    """Send request, auto-starting server if needed."""
    try:
        return _send_request(request)
    except (ConnectionRefusedError, FileNotFoundError, socket.timeout, OSError):
        pass

    if not _start_server():
        raise ConnectionError("Cannot start model server")

    return _send_request(request)


class EmbeddingProxy:
    """Drop-in replacement for the embedding model with .encode() method."""

    def __init__(self, tier: str = ""):
        self._tier = tier

    def encode(self, texts, **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        resp = _request_with_autostart({
            "op": "embed",
            "texts": list(texts),
            "tier": self._tier,
        })
        if not resp.get("ok"):
            raise RuntimeError(f"Model server error: {resp.get('error', 'unknown')}")
        return resp["vectors"]


class RerankerProxy:
    """Drop-in replacement for CrossEncoder with .predict() method."""

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name

    def predict(self, pairs, **kwargs) -> np.ndarray:
        resp = _request_with_autostart({
            "op": "rerank",
            "pairs": list(pairs),
            "model_name": self._model_name,
        })
        if not resp.get("ok"):
            raise RuntimeError(f"Model server error: {resp.get('error', 'unknown')}")
        return resp["scores"]


def _endpoint_published() -> bool:
    """Has the server published a reachable endpoint?

    POSIX: the AF_UNIX socket file exists. Windows: the loopback port file
    exists (written by the server immediately after a successful bind()).
    """
    return _READY_PATH.exists()


def use_model_server() -> bool:
    """Check if the model server should be used.

    Returns True only if:
    1. TRUEMEMORY_NO_MODEL_SERVER is not set
    2. The server has published its endpoint (AF_UNIX socket file on POSIX,
       loopback TCP port file on Windows)
    3. The server process is alive

    Processes that want to ensure the server is running should call
    ensure_server_running() first (e.g., during MCP server startup).
    """
    if os.environ.get("TRUEMEMORY_NO_MODEL_SERVER", "") == "1":
        return False
    return _endpoint_published() and _server_is_alive()


def ensure_server_running() -> bool:
    """Start the model server if it's not already running.

    Call from MCP server startup or CLI to enable the shared model server.
    Returns True if server is running after this call.
    """
    if os.environ.get("TRUEMEMORY_NO_MODEL_SERVER", "") == "1":
        return False
    if _server_is_alive() and _endpoint_published():
        return True
    return _start_server()


def get_embedding_proxy(tier: str = "") -> EmbeddingProxy:
    """Get an embedding proxy connected to the model server."""
    return EmbeddingProxy(tier=tier)


def get_reranker_proxy(model_name: str | None = None) -> RerankerProxy:
    """Get a reranker proxy connected to the model server."""
    return RerankerProxy(model_name=model_name)


def ping() -> bool:
    """Check if model server is reachable."""
    try:
        resp = _send_request({"op": "ping"})
        return resp.get("ok", False)
    except Exception:
        return False
