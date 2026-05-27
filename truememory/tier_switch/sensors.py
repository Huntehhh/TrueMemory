"""Sensor stack for the adaptive MPS throttler.

Three independent monitoring channels: MPS memory level, memory growth
rate, and thermal pressure. Each returns a status dict usable by the
state machine in the throttler.
"""

import logging
import subprocess
import time

log = logging.getLogger(__name__)


def read_mps_memory(cap_gb: float) -> dict:
    """Read MPS driver-allocated memory and classify against cap."""
    try:
        import torch

        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return {"used_gb": 0.0, "ratio": 0.0, "status": "ok"}

        used_bytes = torch.mps.driver_allocated_memory()
        used_gb = used_bytes / (1024**3)
    except (ImportError, AttributeError, RuntimeError):
        return {"used_gb": 0.0, "ratio": 0.0, "status": "ok"}

    ratio = used_gb / cap_gb if cap_gb > 0 else 0.0

    if ratio >= 0.95:
        status = "critical"
    elif ratio >= 0.85:
        status = "warning"
    else:
        status = "ok"

    return {"used_gb": used_gb, "ratio": ratio, "status": status}


class GrowthRateTracker:
    """Tracks MPS memory growth rate between readings."""

    def __init__(self, cap_gb: float):
        self._cap_gb = cap_gb
        self._prev_gb: float = 0.0
        self._prev_time: float = 0.0

    def update(self, current_gb: float) -> dict:
        """Record a new reading and return growth rate status."""
        now = time.time()

        if self._prev_time == 0.0:
            self._prev_gb = current_gb
            self._prev_time = now
            return {"slope_gb_per_20s": 0.0, "slope_pct": 0.0, "status": "ok"}

        dt = now - self._prev_time
        if dt <= 0:
            return {"slope_gb_per_20s": 0.0, "slope_pct": 0.0, "status": "ok"}

        raw_slope = (current_gb - self._prev_gb) / dt
        slope_20s = raw_slope * 20.0
        slope_pct = (slope_20s / self._cap_gb * 100.0) if self._cap_gb > 0 else 0.0

        self._prev_gb = current_gb
        self._prev_time = now

        if slope_pct >= 10.0:
            status = "critical"
        elif slope_pct >= 5.0:
            status = "warning"
        else:
            status = "ok"

        return {"slope_gb_per_20s": slope_20s, "slope_pct": slope_pct, "status": status}


def read_thermal_pressure() -> dict:
    """Read macOS thermal pressure via pmset."""
    try:
        proc = subprocess.run(
            ["pmset", "-g", "therm"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = proc.stdout
        limit = 100
        for line in output.splitlines():
            if "CPU_Scheduler_Limit" in line or "CPU_Speed_Limit" in line:
                parts = line.split("=")
                if len(parts) >= 2:
                    limit = int(parts[-1].strip())
                    break
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError, ValueError):
        return {"scheduler_limit": 100, "status": "ok"}

    if limit <= 70:
        status = "critical"
    elif limit < 100:
        status = "warning"
    else:
        status = "ok"

    return {"scheduler_limit": limit, "status": status}
