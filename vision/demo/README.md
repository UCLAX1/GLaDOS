# Vision demo (PDR)

All vision features run at once on one camera and show in one dashboard:

| key | feature | from |
|---|---|---|
| 1 | Person tracking (YOLO26n + ByteTrack, IDs) | `object_tracker.py` |
| 2 | Gestures for each arm (wave, raise, point, thumbs, peace) | `gestures_revised.py` |
| 3 | Face recognition (names from `known_faces/`) | `face_memory.py` |
| 4 | Expression (8 emotions) | `expression2.py` |
| 5 | Distance to each person in metres (ZED SDK only) | `depth/zed_depth.py` |

One process opens the camera. Each feature runs on its own thread at its own speed on the newest frame. Results are matched to tracked people, so each person gets a box with their ID, name and distance, plus their gestures and expression. If a feature fails to load it shows as *unavailable* and the others keep running.

## Setup (once per machine, needs internet)

```bash
cd GLaDOS                       # repo root, branch vision-demo
python3 vision/demo/setup_demo.py
```

Setup needs Python 3.9–3.12, because MediaPipe has no 3.13 wheels. On a Mac, run `brew install python@3.12`, then use `python3.12 vision/demo/setup_demo.py`.

Setup creates `vision/demo/.venv`, installs everything and downloads all models, so the demo itself works offline. It finishes with a 30-second self-test that prints **GO** or **NOT READY**, with frame rate and time per frame for each feature.

* **insightface needs a C/C++ compiler.** On a Mac run `xcode-select --install`. On Windows install "Build Tools for Visual Studio". If it still fails, the demo runs without face recognition.
* **Faces:** setup doesn't copy registered faces. Copy `vision/known_faces/*.npy` from another machine, or register people live with `r`.

## Run

```bash
vision/demo/.venv/bin/python vision/demo/run_demo.py            # Windows: vision\demo\.venv\Scripts\python
```

| option | what it does |
|---|---|
| `--source auto` | ZED SDK if present, otherwise webcam 0 (default) |
| `--source webcam:1` | another camera; a ZED without the SDK is cropped to its left lens |
| `--source zed` | ZED SDK: adds depth distance (Jetson) |
| `--source path/to/video.avi` | loops a recording instead of the camera (backup) |
| `--modules track,gesture,face` | run only these features |
| `--rotate 180` | camera mounted upside down (90 / 270 for sideways); depth turns with it |
| `--fullscreen` / `--no-window` | projector / browser only |
| `--web 8080` | browser view port, 0 = off |

**Keys:** `1`–`5` turn a feature on or off, `f` toggles fullscreen and `q` quits. To register a face, press `r`, type the name and press Enter (`x` cancels).

**Browser view:** go to `http://<machine-ip>:8080/` from any device on the same network. Tailscale's `http://100.73.92.3:8080/` works from anywhere on the tailnet; campus Wi-Fi often blocks device-to-device traffic. The terminal prints every address at startup. The page shows the same dashboard plus:
* a button for each feature, to turn it on or off
* **Register** / **Cancel**, a ✕ on each known name to forget that face, and **Forget all**

**Registering faces:**
* Only the person being registered may be in view. Sampling pauses while 2+ faces are visible, so a bystander can't be recorded under someone else's name.
* Turn your head slowly until it reaches 15/15. That takes about 20–80 s on the Jetson and gives up after 90 s.
* A face that's already registered under another name is refused, and the message says whose it is.
* Names: each face labels one person only, a name belongs to one person at a time, and two "unknown" results in a row clear a wrong name.
* Faces are saved to `vision/known_faces/<name>.npy`, which is gitignored.

## Demo-day checklist

1. Run the self-test the day before on the machine you'll present from: `python3 vision/demo/setup_demo.py --check-only`. Sit in front of the camera while it runs.
2. Record a backup clip with `vision/gesture_eval/record_gestures.py` and test `--source <clip>`.
3. **Laptop (main):** plug HDMI into the projector, then `run_demo.py --fullscreen`.
4. **Jetson (backup, adds depth):** bring the HDMI monitor, keyboard and mouse. Run `run_demo.py --source zed --fullscreen`.
   * Close other GPU users first (LLM, ZED viewers), because only one program can open the ZED.
   * For a laptop view without the venue Wi-Fi, connect the Jetson and laptop to a phone hotspot and open the browser view.
5. If the frame rate gets low, turn off a feature with its number key. The others speed up right away.

## Measured speed

Each feature runs at its own rate. Jetson numbers were measured with everything running at once (recording, 1 person). Laptop numbers are estimates until the self-test runs there.

| | Jetson Orin Nano (CPU) | laptop (expected) |
|---|---|---|
| tracking | 6 Hz (ONNX, 320 px) | 15–30 Hz (torch, 640 px; MPS on a Mac) |
| gestures | 11–12 Hz (lite models, capped at 12) | 15–25 Hz (full models) |
| faces | ~0.4 Hz, plus at once for a new person | 1 Hz |
| expression | ~5 Hz | 6 Hz (capped) |

On the Jetson, each library's CPU threads are capped (`workers.onnx_threads`, torch 2 threads). Without the caps the libraries fought over 6 cores and tracking fell to about 1 Hz.

Environment overrides: `GESTURE_COMPLEXITY=0|1`, `DEMO_YOLO_IMGSZ=320`, `DEMO_FACE_PERIOD=1.0`, `DEMO_EXPR_HZ=6`, `ZED_DEPTH_MODE=PERFORMANCE`.
