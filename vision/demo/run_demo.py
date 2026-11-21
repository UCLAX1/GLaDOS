#!/usr/bin/env python3
"""
run_demo.py — every GLaDOS vision feature at once, on one camera, in one window.

One process opens the camera (camera.py). Each feature runs on its own thread
at its own speed against the newest frame (workers.py, modules/), and the
dashboard (display.py) fuses the results per tracked person:
    1  person tracking    YOLO26n + ByteTrack            (object_tracker.py)
    2  gestures           MediaPipe pose + hands rules   (gestures_revised.py)
    3  face recognition   InsightFace + known_faces/     (face_memory.py)
    4  expression         MediaPipe face + HSEmotion     (expression2.py)
    5  depth distance     ZED SDK depth per person       (depth/zed_depth.py)
A feature that fails to load is shown as unavailable; the rest keep running.
Auto-idle (presence.py) pauses the heavy features while nobody is close.

Usage (from the repo root, after `python vision/demo/setup_demo.py`):
    vision/demo/.venv/bin/python vision/demo/run_demo.py                  # auto camera, window + web
    ... run_demo.py --source webcam:0                                     # laptop webcam
    ... run_demo.py --source zed                                          # ZED SDK, adds depth
    ... run_demo.py --source vision/gesture_eval/recordings/<run>/video.avi  # no camera needed
    ... run_demo.py --modules track,gesture --fullscreen
    ... run_demo.py --source zed --rotate 180                             # camera upside down
    ... run_demo.py --no-window                                           # browser only
    ... run_demo.py --no-idle                                             # everything always on

Keys: 1-5 toggle a feature, i auto-idle on/off, r register a face (type name, Enter; only
      that person in view), x cancel registering, f fullscreen, q quit.
Browser: http://<this machine>:8080/  (same view, buttons for the same controls)
"""

import argparse
import os
import signal
import socket
import sys
import time
import traceback
from pathlib import Path

# A sourced ROS workspace puts its packages on PYTHONPATH, ahead of the venv's;
# restart without it so only the demo's own packages are imported.
if os.environ.get("PYTHONPATH"):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    os.execve(sys.executable, [sys.executable] + sys.argv, env)

DEMO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO_DIR))

import cv2  # noqa: E402

from camera import FrameSource  # noqa: E402
from display import Compositor, WebServer  # noqa: E402
from presence import Presence  # noqa: E402
from modules.depth import DepthWorker  # noqa: E402
from modules.expression import ExpressionWorker  # noqa: E402
from modules.faces import FaceWorker  # noqa: E402
from modules.gestures import GestureWorker  # noqa: E402
from modules.tracking import TrackerWorker  # noqa: E402

ALL = [TrackerWorker, GestureWorker, FaceWorker, ExpressionWorker, DepthWorker]
WINDOW = "GLaDOS Vision Demo"
DISPLAY_HZ = 30


def local_ips():
    ips = set()
    try:                                           # every interface, incl. Tailscale (100.x)
        import subprocess
        ips |= set(subprocess.run(["hostname", "-I"], capture_output=True, text=True,
                                  timeout=2).stdout.split())
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))           # no packet is sent
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        ips |= {a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}
    except OSError:
        pass
    ips = [i for i in ips if "." in i and not i.startswith("127.")]   # IPv4 only
    return sorted(ips, key=lambda i: not i.startswith("100."))     # Tailscale first


def report(source, workers, presence=None) -> int:
    """--check: one line per module; failure if the camera or any module didn't run."""
    print(f"\n{'module':<18} {'state':<12} {'load s':>7} {'Hz':>6} {'ms':>6}  note")
    print(f"{'camera':<18} {'ok' if source.fps else 'NO FRAMES':<12} {'':>7} {source.fps:6.1f} {'':>6}  {source.status}")
    if presence is not None:
        print(f"{'presence':<18} {presence.status()['mode']:<12} {'':>7} {'':>6} {'':>6}  {presence.line()}")
    bad = source.fps == 0
    for w in workers:
        ok = w.state in ("running",) and w.hz > 0
        # depth only matters with a ZED; its absence elsewhere is expected
        expected_missing = w.key == "depth" and not source.has_depth
        bad |= not ok and not expected_missing
        print(f"{w.label:<18} {w.state:<12} {w.load_s:7.1f} {w.hz:6.1f} {w.ms:6.0f}  {w.message}")
    print("\nRESULT:", "FAIL" if bad else "OK")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", default="auto", help="auto | zed | webcam:N | path to a video file")
    ap.add_argument("--modules", default=",".join(w.key for w in ALL),
                    help="comma list of " + ",".join(w.key for w in ALL))
    ap.add_argument("--web", type=int, default=8080, help="browser view port (0 = off)")
    ap.add_argument("--no-window", action="store_true", help="no OpenCV window (browser only)")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="degrees clockwise, for a camera mounted sideways or upside down")
    ap.add_argument("--height", type=int, default=720, help="camera height per lens (720 or 376/480)")
    ap.add_argument("--no-idle", action="store_true",
                    help="keep every feature running even when nobody is around (also DEMO_IDLE=0)")
    ap.add_argument("--check", type=int, metavar="SECONDS",
                    help="headless self-test: run everything this long, print a table, exit 1 on failure")
    args = ap.parse_args()
    if args.check:
        args.no_window, args.web = True, 0

    wanted = set(args.modules.split(","))
    source = FrameSource(args.source, width=args.height * 16 // 9, height=args.height,
                         rotate=args.rotate).start()
    print(f"[demo] camera: {source.spec}")
    hub = {}
    for cls in ALL:
        if cls.key in wanted:
            hub[cls.key] = cls(source, hub)
    # load one model at a time: loading in parallel thrashes memory on the Jetson
    # and makes every module slow to appear. The window is up meanwhile.
    workers = list(hub.values())
    # the self-test measures every module, so it runs with everything on
    idle = not (args.no_idle or args.check or os.environ.get("DEMO_IDLE", "1") == "0")
    presence = Presence(source, hub, enabled=idle).start()
    comp = Compositor(source, hub, presence)

    def register(name):
        face = hub.get("face")
        msg = face.register(name) if face else "face recognition is not running"
        comp.say(msg)
        return msg

    web = None
    if args.web:
        for port in range(args.web, args.web + 10):    # next port if one is taken
            try:
                web = WebServer(comp, port, register)
                break
            except OSError as e:
                print(f"[demo] port {port}: {e}")
        if web:
            for ip in local_ips() or ["localhost"]:
                print(f"[demo] browser view: http://{ip}:{port}/", flush=True)
        else:
            print("[demo] web view off: no free port")

    running = True

    def stop(*_):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    show = not args.no_window
    if show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        if args.fullscreen:
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    fullscreen = args.fullscreen

    pending = list(workers)
    crashed = False
    t_loaded = None                                    # when every module finished loading
    try:
        while running:
            if t_loaded is None and not pending and all(w.state != "loading" for w in workers):
                t_loaded = time.time()
                print("[demo] all modules loaded")
            if args.check and t_loaded and time.time() - t_loaded > args.check:
                break
            # start the next worker once the previous one has finished loading
            if pending and all(w.state != "loading" for w in workers if w._thread):
                pending.pop(0).start()

            t0 = time.time()
            img = comp.render()
            if web:
                web.publish(img)
            if show:
                cv2.imshow(WINDOW, img)
                key = cv2.waitKey(1) & 0xFF
                if comp.typing is not None:            # typing a name to register
                    if key in (13, 10):
                        register(comp.typing)
                        comp.typing = None
                    elif key == 27:
                        comp.typing = None
                    elif key in (8, 127):
                        comp.typing = comp.typing[:-1]
                    elif 32 < key < 127 and len(comp.typing) < 20:
                        comp.typing += chr(key)
                elif key == ord("q") or key == 27:
                    break
                elif key == ord("i"):
                    comp.say(presence.toggle())
                elif key == ord("r"):
                    comp.typing = ""
                elif key == ord("x"):
                    face = hub.get("face")
                    comp.say(face.cancel_register() if face and hasattr(face, "cancel_register")
                             else "face recognition is not running")
                elif key == ord("f"):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                elif chr(key) in {w.hotkey for w in workers}:
                    w = next(w for w in workers if w.hotkey == chr(key))
                    w.toggle()
                    comp.say(f"{w.label}: {w.state}")
                try:
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break                          # window closed with the X button
                except cv2.error:
                    break
            time.sleep(max(0.0, 1 / DISPLAY_HZ - (time.time() - t0)))
    except Exception:
        # os._exit below would swallow the traceback: print it first
        traceback.print_exc()
        crashed = True
    finally:
        print("[demo] stopping")
        presence.stop()
        code = report(source, workers, presence) if args.check else 0
        code = 1 if crashed else code
        for w in workers:
            w.stop()
        source.stop()                                  # releases the camera
        if web:
            web.close()
        if show:
            cv2.destroyAllWindows()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)                                    # don't wait on model threads mid-call


if __name__ == "__main__":
    main()
