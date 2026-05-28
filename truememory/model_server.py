"""Shared model server — loads embedding + reranker models once for all processes.

Run as: python -m truememory.model_server
Or auto-started by model_client on first request.

Transport is platform-branched:

* POSIX (macOS / Linux) — listens on ~/.truememory/model.sock, a Unix
  domain socket (``AF_UNIX``). Behavior is unchanged from earlier versions.
* Windows — ``AF_UNIX`` is unavailable, so the server listens on a TCP
  socket bound to the loopback address ``127.0.0.1`` (never ``0.0.0.0``).
  It binds to an OS-assigned ephemeral port (``bind`` to port 0) and writes
  the chosen port to ~/.truememory/model_server.port so the client can find
  it. Loopback-only means no other host on the network can reach it.

The wire protocol is identical on both transports: a 4-byte big-endian
length prefix followed by a pickled payload.

Auto-exits after idle timeout (default 300s, configurable via
TRUEMEMORY_MODEL_SERVER_IDLE env var).
"""

import os

try:
    import psutil
except ImportError:
    psutil = None


def _set_mps_memory_cap():
    """Set MPS memory cap and BLAS thread limits BEFORE torch is imported."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    if os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"):
        return
    if psutil is not None:
        total_gb = psutil.virtual_memory().total / (1024**3)
        ratio = min(0.08, 2.5 / total_gb) if total_gb >= 16 else 0.19
        ratio = str(max(ratio, 1.5 / total_gb))
    else:
        ratio = "0.19"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = ratio
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.0")


_set_mps_memory_cap()

import gc  # noqa: E402
import logging  # noqa: E402
import pickle  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import struct  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

log = logging.getLogger(__name__)

_TRUEMEMORY_DIR = Path.home() / ".truememory"
SOCK_PATH = _TRUEMEMORY_DIR / "model.sock"
PID_PATH = _TRUEMEMORY_DIR / "model_server.pid"
# Windows-only: the chosen loopback TCP port is written here so the client
# can discover it. Unused on POSIX (which addresses the AF_UNIX socket path).
PORT_PATH = _TRUEMEMORY_DIR / "model_server.port"
IDLE_TIMEOUT = int(os.environ.get("TRUEMEMORY_MODEL_SERVER_IDLE", "300"))

# Single source of truth for the transport family. AF_UNIX is absent on
# Windows Python; the sys.platform guard is belt-and-suspenders in case a
# future Python on Windows ever exposes a partial AF_UNIX symbol.
_USE_UNIX = hasattr(socket, "AF_UNIX") and sys.platform != "win32"
_LOOPBACK_HOST = "127.0.0.1"

_HEADER_FMT = ">I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically via a temp file + os.replace.

    Prevents a client from reading a half-written port file on Windows
    (no rename-into-place guarantee otherwise on a slow disk).
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


class ModelServer:
    """Serves embedding and reranking.

    Transport is platform-branched: a Unix domain socket on POSIX, a
    loopback TCP socket (127.0.0.1) on Windows. The chosen Windows port is
    captured in ``self._port`` after :meth:`run` binds, and persisted to
    ``PORT_PATH`` for the client to discover.
    """

    _SUSTAINED_THRESHOLD = 10
    _SUSTAINED_WINDOW = 30

    def __init__(self):
        self._embed_model = None
        self._embed_tier: str | None = None
        self._reranker = None
        self._reranker_name: str | None = None
        self._lock = threading.Lock()
        self._last_activity = time.time()
        self._running = True
        self._embed_timestamps: list[float] = []
        self._throttler = None
        self._throttler_active = False
        # Loopback TCP port the server is bound to (Windows only). None until
        # run() binds; stays None on POSIX where the AF_UNIX path is used.
        self._port: int | None = None

    def _get_embed_model(self, tier: str):
        if self._embed_model is not None and self._embed_tier == tier:
            return self._embed_model

        from truememory.vector_search import EMBEDDING_MODEL, set_embedding_model

        if tier and tier != EMBEDDING_MODEL:
            set_embedding_model(tier)

        resolved = EMBEDDING_MODEL if not tier else tier
        from truememory.vector_search import _TIER_ALIASES
        model_id = _TIER_ALIASES.get(resolved, resolved)

        if model_id == "model2vec":
            from model2vec import StaticModel
            self._embed_model = StaticModel.from_pretrained(
                "minishlab/potion-base-8M", force_download=False
            )
        elif model_id == "qwen3_256":
            from sentence_transformers import SentenceTransformer
            mkwargs = {}
            if sys.platform == "darwin":
                mkwargs["attn_implementation"] = "eager"
            self._embed_model = SentenceTransformer(
                "Qwen/Qwen3-Embedding-0.6B",
                truncate_dim=256,
                model_kwargs=mkwargs or None,
            )
        else:
            from model2vec import StaticModel
            self._embed_model = StaticModel.from_pretrained(
                "minishlab/potion-base-8M", force_download=False
            )

        self._embed_tier = tier
        log.info("Loaded embedding model for tier=%s", tier)
        return self._embed_model

    def _get_reranker(self, model_name: str | None = None):
        from truememory.reranker import get_current_reranker_name
        name = model_name or get_current_reranker_name()

        if self._reranker is not None and self._reranker_name == name:
            return self._reranker

        from sentence_transformers import CrossEncoder
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
        except ImportError:
            pass

        self._reranker = CrossEncoder(name, device=device)
        self._reranker_name = name
        log.info("Loaded reranker model=%s device=%s", name, device)
        return self._reranker

    def handle_request(self, request: dict) -> dict:
        self._last_activity = time.time()
        op = request.get("op")

        if op == "ping":
            return {"ok": True}

        if op == "embed":
            texts = request["texts"]
            tier = request.get("tier", "")

            now = time.time()
            self._embed_timestamps.append(now)
            self._embed_timestamps = [
                t for t in self._embed_timestamps
                if now - t < self._SUSTAINED_WINDOW
            ]

            if (len(self._embed_timestamps) >= self._SUSTAINED_THRESHOLD
                    and not self._throttler_active):
                self._activate_throttler()

            if self._throttler_active and self._throttler:
                self._throttler.before_batch()

            encode_start = time.time()
            with self._lock:
                model = self._get_embed_model(tier)
                vectors = model.encode(texts, show_progress_bar=False)
            encode_time = time.time() - encode_start

            if self._throttler_active and self._throttler:
                self._throttler.after_batch(len(texts), encode_time)
                if self._throttler.should_flush_cache():
                    self._flush_mps_cache()

            if self._throttler_active and len(self._embed_timestamps) < 3:
                self._deactivate_throttler()

            return {"ok": True, "vectors": np.asarray(vectors, dtype=np.float32)}

        if op == "rerank":
            pairs = request["pairs"]
            model_name = request.get("model_name")
            with self._lock:
                reranker = self._get_reranker(model_name)
                scores = reranker.predict(
                    pairs, batch_size=64, show_progress_bar=False
                )
            return {"ok": True, "scores": np.asarray(scores, dtype=np.float32)}

        return {"ok": False, "error": f"Unknown op: {op}"}

    def _activate_throttler(self):
        """Start adaptive throttling for sustained workload."""
        try:
            from truememory.tier_switch.throttler import DynamicThrottler
        except ImportError:
            log.warning("Cannot import DynamicThrottler — running without throttling")
            return
        device = "cpu"
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
        except ImportError:
            pass
        self._throttler = DynamicThrottler(device=device)
        self._throttler_active = True
        log.info(
            "Sustained workload detected (%d requests in %ds) — throttler activated",
            len(self._embed_timestamps), self._SUSTAINED_WINDOW,
        )

    def _deactivate_throttler(self):
        """Stop adaptive throttling — workload ended."""
        self._throttler = None
        self._throttler_active = False
        self._embed_timestamps.clear()
        log.info("Workload ended — throttler deactivated")

    def _flush_mps_cache(self):
        """Flush MPS cache — only called when throttler says to."""
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                torch.mps.synchronize()
        except Exception:
            pass
        gc.collect()

    def handle_client(self, conn: socket.socket):
        try:
            header = self._recv_exact(conn, _HEADER_SIZE)
            if not header:
                return
            length = struct.unpack(_HEADER_FMT, header)[0]
            data = self._recv_exact(conn, length)
            if not data:
                return

            request = pickle.loads(data)
            response = self.handle_request(request)
            self._send_response(conn, response)
        except Exception as e:
            try:
                self._send_response(conn, {"ok": False, "error": str(e)})
            except Exception:
                pass
        finally:
            conn.close()

    def _recv_exact(self, conn: socket.socket, n: int) -> bytes | None:
        buf = bytearray()
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def _send_response(self, conn: socket.socket, response: dict):
        data = pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
        header = struct.pack(_HEADER_FMT, len(data))
        conn.sendall(header + data)

    def _idle_checker(self):
        while self._running:
            time.sleep(60)
            if not self._running:
                break
            elapsed = time.time() - self._last_activity
            if elapsed >= IDLE_TIMEOUT:
                log.info(
                    "Idle timeout (%.0fs), shutting down model server", elapsed
                )
                self._running = False
                # Nudge the blocking accept() awake by self-connecting on the
                # same transport the listener is bound to.
                try:
                    if _USE_UNIX:
                        dummy = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                        dummy.connect(str(SOCK_PATH))
                    else:
                        dummy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        dummy.connect((_LOOPBACK_HOST, self._port))
                    dummy.close()
                except Exception:
                    pass
                break

    def run(self):
        _TRUEMEMORY_DIR.mkdir(parents=True, exist_ok=True)

        if _USE_UNIX:
            if SOCK_PATH.exists():
                SOCK_PATH.unlink()
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv.bind(str(SOCK_PATH))
            endpoint = str(SOCK_PATH)
        else:
            # Windows: loopback TCP. Bind to port 0 so the OS hands us a free
            # ephemeral port, then publish the chosen port for the client.
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((_LOOPBACK_HOST, 0))
            self._port = srv.getsockname()[1]
            # Write the port BEFORE listen()/PID so a client that sees the PID
            # is guaranteed to also see a readable port file.
            _atomic_write_text(PORT_PATH, str(self._port))
            endpoint = f"{_LOOPBACK_HOST}:{self._port}"

        # PID is written last so _server_is_alive()-style probes that key off
        # the PID only succeed once the listener address is fully published.
        PID_PATH.write_text(str(os.getpid()))

        srv.listen(16)
        srv.settimeout(2.0)

        idle_thread = threading.Thread(target=self._idle_checker, daemon=True)
        idle_thread.start()

        log.info(
            "Model server started: pid=%d endpoint=%s idle_timeout=%ds",
            os.getpid(), endpoint, IDLE_TIMEOUT,
        )

        try:
            while self._running:
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not self._running:
                    conn.close()
                    break
                t = threading.Thread(target=self.handle_client, args=(conn,), daemon=True)
                t.start()
        finally:
            srv.close()
            self._cleanup()

    def _cleanup(self):
        if SOCK_PATH.exists():
            SOCK_PATH.unlink(missing_ok=True)
        if PORT_PATH.exists():
            PORT_PATH.unlink(missing_ok=True)
        if PID_PATH.exists():
            PID_PATH.unlink(missing_ok=True)
        self._embed_model = None
        self._reranker = None
        gc.collect()
        log.info("Model server stopped")


def _handle_signal(signum, frame):
    log.info("Received signal %d, shutting down", signum)
    sys.exit(0)


def _pid_is_alive(pid: int) -> bool:
    """Cross-platform 'is this PID running?' check.

    POSIX uses the classic ``os.kill(pid, 0)`` probe. Windows has no
    signal-0 semantics (``os.kill`` there only delivers CTRL events or
    terminates), so we probe via psutil when available, falling back to a
    ``tasklist`` query. On any uncertainty we report False so a stale file
    never blocks a restart.
    """
    if sys.platform == "win32":
        if psutil is not None:
            return psutil.pid_exists(pid)
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


def _remove_stale_endpoint_files() -> None:
    """Drop sock/port artifacts left by a crashed or killed server."""
    if SOCK_PATH.exists():
        SOCK_PATH.unlink(missing_ok=True)
    if PORT_PATH.exists():
        PORT_PATH.unlink(missing_ok=True)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [model_server] %(levelname)s %(message)s",
    )

    try:
        import setproctitle
        setproctitle.setproctitle("TrueMemory")
    except ImportError:
        pass

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _handle_signal)

    if PID_PATH.exists():
        try:
            old_pid = int(PID_PATH.read_text().strip())
        except ValueError:
            PID_PATH.unlink(missing_ok=True)
            _remove_stale_endpoint_files()
        else:
            if _pid_is_alive(old_pid):
                log.error("Model server already running (pid=%d)", old_pid)
                sys.exit(1)
            PID_PATH.unlink(missing_ok=True)
            _remove_stale_endpoint_files()

    # Belt-and-suspenders: even if the accept loop's finally is skipped
    # (e.g. an abrupt interpreter teardown), drop our endpoint files so the
    # next start isn't blocked by a stale PID/port.
    import atexit
    atexit.register(_remove_stale_endpoint_files)

    server = ModelServer()
    server.run()


if __name__ == "__main__":
    main()
