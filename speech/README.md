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

Any other process triggers speech by writing JSON to `/tmp/glados_request.json`:

```python
import json
from pathlib import Path
Path("/tmp/glados_request.json").write_text(
    json.dumps({"text": "Firing neurotoxin in 3, 2, 1.", "speed": 0.95})
)
```

The daemon consumes the file, speaks, then goes back to waiting. Typical latency after warmup: 0.3–0.7x real-time on Mac CPU, faster on Jetson CUDA.

## Structure

```
speech/
├── glados_tts.py           main inference script (auto-downloads weights)
├── glados_daemon.py        robot speech daemon — load once, speak on demand
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
