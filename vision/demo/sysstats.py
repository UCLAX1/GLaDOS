"""
sysstats.py — dependency-free system usage for the Jetson (served on /status).

Reads /proc and sysfs directly so it works in the demo venv without psutil:
    cpu %      /proc/stat (delta between calls)
    ram        /proc/meminfo
    gpu %      Jetson GPU 'load' node in sysfs (0–1000 → %)
    temp °C    /sys/class/thermal/thermal_zone*/
    proc rss   /proc/self/status (this demo process)

Every field degrades to None if unavailable (e.g. running on a Mac).
"""

import glob
import threading
import time

_GPU_LOAD_GLOBS = [
    "/sys/devices/platform/bus@0/*.gpu/load",       # Orin (JetPack 6)
    "/sys/devices/platform/*.gpu/load",
    "/sys/devices/platform/*ga10b*/load",
    "/sys/devices/gpu.0/load",                       # Nano / Xavier
    "/sys/devices/platform/gpu.0/load",
]

_lock = threading.Lock()
_prev_cpu = None          # (idle, total) from the previous sample
_cache = (0.0, {})        # (timestamp, snapshot) — avoid re-reading at 5 Hz


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _cpu_percent():
    global _prev_cpu
    line = _read("/proc/stat")
    if not line:
        return None
    vals = [int(x) for x in line.splitlines()[0].split()[1:]]
    idle, total = vals[3] + (vals[4] if len(vals) > 4 else 0), sum(vals)
    prev, _prev_cpu = _prev_cpu, (idle, total)
    if prev is None or total == prev[1]:
        return None
    return round(100.0 * (1 - (idle - prev[0]) / (total - prev[1])), 1)


def _mem():
    txt = _read("/proc/meminfo")
    if not txt:
        return None, None
    kv = {}
    for line in txt.splitlines():
        k, _, v = line.partition(":")
        kv[k] = int(v.split()[0]) if v.split() else 0
    total = kv.get("MemTotal", 0) / 1048576
    avail = kv.get("MemAvailable", 0) / 1048576
    return round(total - avail, 2), round(total, 2)


def _gpu_percent():
    for pattern in _GPU_LOAD_GLOBS:
        for path in glob.glob(pattern):
            v = _read(path)
            if v and v.isdigit():
                return round(int(v) / 10.0, 1)
    return None


def _temps():
    out = {}
    for zone in glob.glob("/sys/class/thermal/thermal_zone*"):
        name, t = _read(f"{zone}/type"), _read(f"{zone}/temp")
        if name and t and t.lstrip("-").isdigit():
            c = int(t) / 1000.0
            if -40 < c < 150:
                out[name.lower()] = round(c, 1)
    if not out:
        return None, None
    cpu = next((v for k, v in out.items() if "cpu" in k), None)
    gpu = next((v for k, v in out.items() if "gpu" in k), None)
    return cpu, gpu if gpu is not None else max(out.values())


def _proc_rss_gb():
    txt = _read("/proc/self/status")
    if not txt:
        return None
    for line in txt.splitlines():
        if line.startswith("VmRSS:"):
            return round(int(line.split()[1]) / 1048576, 2)
    return None


def snapshot(min_interval: float = 1.0) -> dict:
    """Current usage; cached for min_interval seconds."""
    global _cache
    with _lock:
        now = time.time()
        if now - _cache[0] < min_interval and _cache[1]:
            return _cache[1]
        ram_used, ram_total = _mem()
        cpu_t, gpu_t = _temps()
        snap = {
            "cpu_pct":     _cpu_percent(),
            "gpu_pct":     _gpu_percent(),
            "ram_used_gb": ram_used,
            "ram_total_gb": ram_total,
            "cpu_temp_c":  cpu_t,
            "gpu_temp_c":  gpu_t,
            "demo_rss_gb": _proc_rss_gb(),
        }
        _cache = (now, snap)
        return snap
