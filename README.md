# X1 GLaDOS Robot

UCLA X1's 2026-2027 GLaDOS project.

## Running the full pipeline

Three terminals, in order:

**Terminal 1 — Brain (Bonsai-8B via llama.cpp)**
```bash
llama-server -m central_ai/benchmarking_tools/Bonsai-8B-Q1_0.gguf --port 8080
```

**Terminal 2 — Speech daemon**
```bash
python3 speech/glados_daemon.py
```

**Terminal 3 — Main pipeline (mic → brain → speech + gesture)**
```bash
source listening/venv/bin/activate
python3 run_glados.py
```

Speak into the mic. glados responds with audio and a MuJoCo gesture.

## Structure

```
run_glados.py           full pipeline entry point
│
├── listening/
│   └── listener.py     mic → VAD → faster-whisper → text callback
│
├── central_ai/
│   ├── brain.py        Bonsai-8B brain — returns {speech, gesture, look_at, mood}
│   └── benchmarking_tools/   LLM eval suite + Bonsai-8B model file
│
├── speech/
│   ├── glados_tts.py   Kokoro TTS inference (auto-downloads weights from HF)
│   ├── glados_daemon.py    speech daemon — load once, speak on demand via JSON
│   └── voice_training/     training pipeline for the GLaDOS voice model
│
├── actions/
│   ├── run_action.py   run any motor action (sim or hardware)
│   ├── sequence.py     backend-agnostic pose sequencer
│   └── scripts/        individual action files (nod, scan, look_away, …)
│
├── control/
│   ├── control_interface.py    abstract joint interface
│   └── mujoco_control.py       MuJoCo backend
│
├── sim/
│   ├── runner.py       MuJoCo sim setup
│   └── model/          glados.xml (arm), push.xml (head mechanism prototype)
│
└── vision/             (WIP)
```

## Running actions in sim

```bash
# Loop (interactive)
mjpython actions/run_action.py nod

# Play once and exit
mjpython actions/run_action.py nod --once
```

## Speech model

Weights: [`yifanfang/glados-kokoro`](https://huggingface.co/yifanfang/glados-kokoro) (private — request access from Yifan)

Downloaded automatically on first run. See `speech/README.md` for details.

## LLM

Bonsai-8B Q1_0 at `central_ai/benchmarking_tools/Bonsai-8B-Q1_0.gguf` — 1.15 GB, runs fully local.
Benchmarked at 100% persona consistency, 89% JSON format compliance, 1.52s avg latency.
See `central_ai/README.md` and `central_ai/benchmarking_tools/` for the full eval suite.
