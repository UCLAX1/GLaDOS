# Speech

GLaDOS voice synthesis for the X1 robot. Fine-tuned [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (StyleTTS2-based) on Portal 1, Portal 2, and Poker Night 2 dialogue — 82M params, ~487MB RAM, ~0.02 real-time factor.

Model weights: [`yifanfang/glados-kokoro`](https://huggingface.co/yifanfang/glados-kokoro) (private — request access from Yifan)

## Quick start

```bash
pip install kokoro "misaki[en]" inflect soundfile huggingface_hub
brew install espeak-ng          # Mac
sudo apt-get install espeak-ng  # Linux / Jetson

python3 speech/glados_tts.py "The cake is a lie."
```

Model files download automatically from HuggingFace on first run.

## Robot integration — speech daemon

For the robot, run the daemon once at startup. It loads the model, warms up, then listens for speech requests with minimal latency:

```bash
python3 speech/glados_daemon.py
```

Any other process triggers speech by appending to the queue:

```python
from speech.speech_queue import enqueue
enqueue("Firing neurotoxin in 3, 2, 1.", speed=0.95)
```

The daemon takes the oldest pending utterance, speaks it, and moves to the next.
Utterances queue rather than overwrite, so a reply streamed sentence by sentence
plays in order and nothing is dropped mid-playback.

The queue is a directory of JSON files (`/tmp/glados_queue/` by default,
`GLADOS_QUEUE_DIR` to override) written atomically, so the daemon never reads a
half-written request. `speech_queue.py` has no heavy dependencies — importing it
does not pull in torch or kokoro.

The older single-file interface still works for one-off requests:

```python
import json
from pathlib import Path
Path("/tmp/glados_request.json").write_text(json.dumps({"text": "The cake is a lie."}))
```

Note that a request left there is spoken whenever the daemon next starts, even
days later — clear the file if you abort a session mid-utterance.

Playback streams segment by segment as Kokoro produces them, through a single
PortAudio stream so consecutive segments run together with no seam. If PortAudio
is unavailable the daemon falls back to `afplay`/`aplay` per segment. Typical
latency after warmup: 0.3–0.7x real-time on Mac CPU, faster on Jetson CUDA.

## Structure

```
speech/
├── glados_tts.py           main inference script (auto-downloads weights)
├── glados_daemon.py        robot speech daemon — load once, speak on demand
├── speech_queue.py         atomic utterance queue (no heavy deps)
├── setup_inference.sh      one-time setup (Mac, Linux, Jetson)
├── config_kokoro.json      Kokoro architecture config
│
└── voice_training/
    ├── deploy_to_gpu.sh    sync everything to a GPU and start training
    ├── data_pipeline/      scrape + process + sort training audio
    ├── training/           GPU fine-tuning scripts and configs
    ├── notebooks/          experiment notebooks
    ├── prepared_data/      train/val splits used for the current model
    └── dataset_manifests/  record of what the model was trained on
```

## Deploying to a new GPU

```bash
# 1. Copy your sorted audio to the GPU first
rsync -avz speech/voice_training/sorted/robotic/ gpu_zflow:~/voice_training/data/wavs_robotic/
rsync -avz speech/voice_training/sorted/human/   gpu_zflow:~/voice_training/data/wavs_human/

# 2. Deploy scripts and launch training
bash speech/voice_training/deploy_to_gpu.sh [gpu_host]
```
