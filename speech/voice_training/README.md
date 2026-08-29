# GLaDOS Voice Training Pipeline

Scrapes GLaDOS voice lines from the Portal Wiki, transcribes them, and sorts them by emotion for Kokoro/StyleTTS2 fine-tuning.

## Setup

```bash
cd voice_training
pip install -r requirements.txt
```

## Step 1 — Scrape audio from Portal Wiki

```bash
python scrape.py
```

Downloads all GLaDOS `.ogg` files from Portal, Portal 2, and Coop into `raw_audio/`. Also saves `raw_audio/metadata.json` with filenames + wiki transcripts.

## Step 2 — Transcribe + tag emotions

Requires llama.cpp server running on port 8080 (for emotion tagging):

```bash
llama-server -m ../central_ai/benchmarking_tools/Bonsai-8B-Q1_0.gguf --port 8080 -ngl 99
```

Then run:

```bash
python process.py
```

This:
1. Re-transcribes each file with faster-whisper (more accurate than wiki text)
2. Classifies each line into one of 7 emotions using Bonsai 8B
3. Copies files into `processed_audio/<emotion>/` subfolders
4. Saves `processed_audio/dataset.json`

Test with a small batch first:

```bash
python process.py --limit 50
```

Skip transcription (use wiki text) or skip emotion tagging:

```bash
python process.py --no-transcribe
python process.py --no-tag
```

## Output structure

```
processed_audio/
  sarcastic/       ← most lines (~60%)
  cold/
  angry/
  triumphant/
  distressed/
  sincere/
  neutral/
  dataset.json     ← full manifest with transcripts + emotion labels
```

## Emotions

| Emotion | Description | Example |
|---|---|---|
| sarcastic | Dry, contemptuous, superior | Most test chamber dialogue |
| cold | Flat, menacing, clinical | "You will be baked, and then there will be cake." |
| angry | Frustrated, hostile | Wheatley takeover scenes |
| triumphant | Smug, victorious | Final boss moments |
| distressed | Panicked, desperate | End of Portal 1 |
| sincere | Genuine, unguarded | "Caroline" scenes in Portal 2 |
| neutral | Informational, no affect | Tutorial lines |

## Training

Fine-tune Kokoro/StyleTTS2 on the processed audio:

```bash
bash deploy_to_gpu.sh    # sync scripts to GPU and launch training
bash monitor_kokoro.sh   # check training status
```

Best checkpoint: Stage 2 epoch 4, validation loss 0.346 (82M params, ~487MB peak RAM, RTF 0.019 on RTX 5090).
Model weights and voicepack are hosted at `yifanfang/glados-kokoro` on HuggingFace.

Dataset manifests (what the model was trained on) are in `dataset_manifests/`.

## Inference

Run locally on Mac or Jetson:

```bash
bash setup_mac_inference.sh   # one-time setup (pulls weights from HF)
python3 glados_tts.py "The cake is a lie."
```
