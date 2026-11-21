#!/usr/bin/env python3
"""
run.py — the one command for GLaDOS, on every machine.

    python3 run.py          Mac / Linux / Jetson
    py run.py               Windows (or double-click run.py)

It works out what this machine is for, installs whatever is missing (only the
first time, or when requirements.txt changes), and starts it:

    Jetson              → camera + vision demo  (serves :8081 — start this first)
    Mac / Windows / PC  → LLM + voice + listening + sim + GUI
                          vision: the Jetson if it answers, else this machine's
                          webcam (if set up), else none

Options:
    --setup             reinstall / re-check everything, even if it looks done
    --vision            also set up webcam vision on this machine (Mac/PC), used
                        automatically when the Jetson is not reachable
    --role vision       run the camera/vision side here instead (any machine)
    --jetson IP         Jetson address (default 100.73.92.3; "none" = no vision)
    --no-start          set up only

Environment variables still work too: JETSON_IP, VISION_MODE (auto|jetson|local|
none), GLADOS_GAP_MS, LLAMA_ARGS, WEB_PORT.

Only the Python standard library is used here, so this runs before anything is
installed. Needs Python 3.10–3.12 somewhere on the machine for the venvs.
"""

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT     = Path(__file__).resolve().parent
WIN      = os.name == "nt"
MAC      = sys.platform == "darwin"
BRAIN_VENV  = ROOT / "listening" / "venv"
VISION_VENV = ROOT / "vision" / "demo" / ".venv"
GGUF     = ROOT / "central_ai" / "benchmarking_tools" / "Bonsai-8B-Q1_0.gguf"
LLAMA_DIR = ROOT / "tools" / "llama"
PRISM_TAG = "prism-b12128-d7b2896"     # PrismML llama.cpp build that runs Bonsai's 1-bit format
PRISM_URL = f"https://github.com/PrismML-Eng/llama.cpp/releases/download/{PRISM_TAG}"

os.chdir(ROOT)


# ── output helpers ────────────────────────────────────────────────────────────

def _supports_color():
    return sys.stdout.isatty() and (not WIN or os.environ.get("WT_SESSION"))

def say(msg):  print(f"\n\033[1m{msg}\033[0m" if _supports_color() else f"\n== {msg}")
def ok(msg):   print(f"  [ok] {msg}")
def warn(msg): print(f"  [!]  {msg}")
def step(msg): print(f"  ...  {msg}", flush=True)

def die(msg):
    print(f"\n  [x]  {msg}\n")
    if WIN:
        input("Press Enter to close…")
    sys.exit(1)

def run(cmd, **kw):
    """Run a command, streaming its output. Returns the exit code."""
    return subprocess.call([str(c) for c in cmd], **kw)

def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if WIN else "bin/python")


# ── which machine is this? ────────────────────────────────────────────────────

def is_jetson() -> bool:
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        return False
    if Path("/etc/nv_tegra_release").exists():
        return True
    try:
        return "jetson" in Path("/proc/device-tree/model").read_text(errors="ignore").lower()
    except OSError:
        return False


# ── a Python 3.10–3.12 to build venvs with ────────────────────────────────────

def find_python() -> list[str]:
    """Command for a Python 3.10–3.12 (MediaPipe / PyAudio have no newer wheels yet)."""
    def good(cmd):
        try:
            out = subprocess.run(cmd + ["-c", "import sys; print(sys.version_info[:2])"],
                                 capture_output=True, text=True, timeout=20).stdout
            return out.strip() in ("(3, 10)", "(3, 11)", "(3, 12)")
        except Exception:
            return False

    if (3, 10) <= sys.version_info[:2] <= (3, 12):
        return [sys.executable]
    cands = ([["py", f"-3.{v}"] for v in (12, 11, 10)] if WIN else
             [[f"python3.{v}"] for v in (12, 11, 10)]) + [["python3"], ["python"]]
    for c in cands:
        if good(c):
            return c
    die("Python 3.10–3.12 is needed to build the environments. "
        + ("Install Python 3.12 from python.org (tick 'Add to PATH'), then run again."
           if WIN else "Install it (e.g. `brew install python@3.12` or pyenv), then run again."))


# ── setup marker: skip the slow steps when nothing changed ────────────────────

def _req_hash(*files) -> str:
    h = hashlib.sha1()
    for f in files:
        p = ROOT / f
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]

def _marker(venv: Path) -> Path:
    return venv / ".glados_setup"

def setup_current(venv: Path, *req_files) -> bool:
    m = _marker(venv)
    return venv_python(venv).exists() and m.exists() and m.read_text().strip() == _req_hash(*req_files)

def mark_setup(venv: Path, *req_files):
    _marker(venv).write_text(_req_hash(*req_files))


# ── brain machine (Mac / Windows / Linux PC) ──────────────────────────────────

def has_nvidia() -> bool:
    return shutil.which("nvidia-smi") is not None

def find_llama_server() -> str | None:
    found = shutil.which("llama-server")
    if found:
        return found
    exe = "llama-server.exe" if WIN else "llama-server"
    for p in sorted(LLAMA_DIR.rglob(exe)) if LLAMA_DIR.exists() else []:
        return str(p)
    return None

def download(url: str, dest: Path):
    step(f"downloading {url.rsplit('/', 1)[-1]}")
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)

def install_llama_server():
    if find_llama_server():
        ok(f"llama-server ({find_llama_server()})")
        return
    if MAC:
        if shutil.which("brew"):
            step("installing llama.cpp with Homebrew")
            run(["brew", "install", "llama.cpp"])
        else:
            warn("Homebrew not found — install llama.cpp yourself (https://brew.sh, then `brew install llama.cpp`)")
    elif WIN:
        LLAMA_DIR.mkdir(parents=True, exist_ok=True)
        zips = ([f"llama-{PRISM_TAG}-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"]
                if has_nvidia() else [f"llama-{PRISM_TAG}-bin-win-cpu-x64.zip"])
        for z in zips:
            dest = LLAMA_DIR / z
            try:
                download(f"{PRISM_URL}/{z}", dest)
                with zipfile.ZipFile(dest) as zf:
                    zf.extractall(LLAMA_DIR)
            except Exception as e:
                warn(f"could not download {z}: {e}")
            finally:
                dest.unlink(missing_ok=True)
    else:
        warn("llama-server not found. Build the PrismML fork (it runs Bonsai's 1-bit format):\n"
             "         git clone https://github.com/PrismML-Eng/llama.cpp && cd llama.cpp && "
             "cmake -B build && cmake --build build -j --target llama-server\n"
             "       then put build/bin/llama-server on your PATH (or in tools/llama/).")
    if find_llama_server():
        ok(f"llama-server ({find_llama_server()})")
    else:
        warn("llama-server is still missing — GLaDOS cannot think without it")

def setup_brain(force: bool):
    say(f"Setup — {'Mac' if MAC else 'Windows' if WIN else 'Linux'} (brain, voice, GUI)")

    # System libraries (Mac): PortAudio for mic/speaker, espeak-ng for the voice
    if MAC and shutil.which("brew"):
        for pkg in ("portaudio", "espeak-ng"):
            if subprocess.run(["brew", "list", pkg], capture_output=True).returncode == 0:
                ok(pkg)
            else:
                step(f"installing {pkg}")
                run(["brew", "install", pkg])
    elif not WIN and not MAC and not shutil.which("espeak-ng"):
        warn("espeak-ng / PortAudio may be missing — sudo apt install espeak-ng portaudio19-dev")

    install_llama_server()

    # Python environment
    vpy = venv_python(BRAIN_VENV)
    if force or not setup_current(BRAIN_VENV, "requirements.txt"):
        if not vpy.exists():
            py = find_python()
            step(f"creating {BRAIN_VENV.relative_to(ROOT)} with {' '.join(py)}")
            if run(py + ["-m", "venv", str(BRAIN_VENV)]) != 0:
                die("could not create the virtual environment")
        run([vpy, "-m", "pip", "install", "-q", "--upgrade", "pip"])
        if WIN and has_nvidia():
            step("installing PyTorch with CUDA (NVIDIA GPU found)")
            run([vpy, "-m", "pip", "install", "-q", "torch", "torchaudio",
                 "--index-url", "https://download.pytorch.org/whl/cu124"])
        step("installing Python packages (first run: a few minutes)")
        if run([vpy, "-m", "pip", "install", "-q", "-r", "requirements.txt"]) != 0:
            die("pip install failed — see the messages above")
        mark_setup(BRAIN_VENV, "requirements.txt")
    ok(f"Python environment ({BRAIN_VENV.relative_to(ROOT)})")

    # Model
    if GGUF.exists():
        ok("Bonsai-8B model")
    else:
        step("downloading Bonsai-8B (1.1 GB)")
        run([vpy, "-m", "pip", "install", "-q", "huggingface_hub"])
        run([vpy, "-c", "from huggingface_hub import hf_hub_download as d; "
             "d('prism-ml/Bonsai-8B-gguf', 'Bonsai-8B-Q1_0.gguf', "
             "local_dir='central_ai/benchmarking_tools')"])
        if not GGUF.exists():
            die(f"model download failed — put Bonsai-8B-Q1_0.gguf at {GGUF.relative_to(ROOT)}")
        ok("Bonsai-8B model")


def setup_vision(force: bool):
    if force or not venv_python(VISION_VENV).exists():
        say("Setup — camera + vision")
        if run(find_python() + ["vision/demo/setup_demo.py"]) != 0:
            die("vision setup failed — see the messages above")
    ok(f"vision environment ({VISION_VENV.relative_to(ROOT)})")


# ── clean up leftovers from a run that did not exit cleanly ───────────────────

def kill_leftovers(patterns: list[str], exe_names: list[str] = ()):
    if WIN:
        for exe in exe_names:
            subprocess.run(["taskkill", "/F", "/IM", exe], capture_output=True)
        regex = "|".join(p.replace(".", r"\.") for p in patterns)
        ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and "
              f"$_.CommandLine -match '{regex}' }} | ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force }}")
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True)
    else:
        for p in patterns:
            subprocess.run(["pkill", "-f", p], capture_output=True)


# ── start ─────────────────────────────────────────────────────────────────────

def launch(cmd: list, env: dict):
    """Hand over to the program (replaces this process on Mac/Linux)."""
    cmd = [str(c) for c in cmd]
    if WIN:
        code = subprocess.call(cmd, env=env)
        sys.exit(code)
    os.execve(cmd[0], cmd, env)

def start_vision(port: str):
    kill_leftovers(["vision/demo/run_demo.py", "vision_server.py"])
    ip = ""
    try:
        ip = subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split()[0]
    except Exception:
        pass
    say("GLaDOS — vision")
    print(f"  video:      http://{ip or 'this-machine'}:{port}/stream")
    print(f"  perception: http://{ip or 'this-machine'}:{port}/status")
    print(f"  on the Mac/PC:  python3 run.py --jetson {ip or '<this IP>'}\n")
    launch([venv_python(VISION_VENV), "vision/demo/run_demo.py",
            "--source", "zed" if is_jetson() else "auto", "--no-window", "--web", port],
           dict(os.environ))

def start_brain(jetson: str | None):
    env = dict(os.environ)
    env["JETSON_IP"] = jetson if jetson is not None else env.get("JETSON_IP", "100.73.92.3")
    env.setdefault("VISION_DEMO_PORT", env.get("WEB_PORT", "8081"))
    env["PYTHONUTF8"] = "1"
    llama = find_llama_server()
    if llama:
        env["LLAMA_SERVER"] = llama
    kill_leftovers(["llama-server.*--port 8080", "speech/glados_daemon.py", "glados_daemon.py",
                    "run_demo.py.*--web 8091"], exe_names=["llama-server.exe"])
    say(f"GLaDOS — {'Mac' if MAC else 'PC'}   (vision: {env.get('VISION_MODE', 'auto')}, "
        f"Jetson {env['JETSON_IP'] or 'none'})")
    launch([venv_python(BRAIN_VENV), "gui_launch.py"], env)


def main():
    ap = argparse.ArgumentParser(description="Set up (if needed) and start GLaDOS on this machine.")
    ap.add_argument("--setup", action="store_true", help="reinstall / re-check everything")
    ap.add_argument("--vision", action="store_true", help="also set up webcam vision here (Mac/PC)")
    ap.add_argument("--role", choices=["auto", "brain", "vision"], default="auto")
    ap.add_argument("--jetson", metavar="IP", help='Jetson address, or "none"')
    ap.add_argument("--no-start", action="store_true", help="set up only")
    a = ap.parse_args()

    role = a.role if a.role != "auto" else ("vision" if is_jetson() else "brain")
    port = os.environ.get("WEB_PORT", "8081")

    try:
        if role == "vision":
            setup_vision(a.setup)
            if not a.no_start:
                start_vision(port)
        else:
            setup_brain(a.setup)
            if a.vision:
                setup_vision(a.setup)
            elif venv_python(VISION_VENV).exists():
                ok("webcam vision (used when the Jetson is not reachable)")
            else:
                print("  [i]  no webcam vision here — without the Jetson, GLaDOS runs without "
                      "vision.  Add it with:  run.py --vision")
            if not a.no_start:
                start_brain(a.jetson)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
