"""
GLaDOS voice training pipeline — Step 2: Transcribe + Emotion Tag

Reads raw_audio/ from scrape.py, then:
  1. Transcribes each file with faster-whisper (validates/corrects wiki transcripts)
  2. Tags emotion using two signals:
       - Audio: wav2vec2 speech emotion recognition (acoustic tone)
       - Text:  Bonsai 8B via llama.cpp (semantic content)
     Both signals are combined — audio wins for tonal emotions (cold, angry,
     distressed), text wins for semantic ones (sarcastic, triumphant, sincere).
  3. Outputs processed_audio/ organized by emotion + dataset.json

Emotions:
  sarcastic   — dry, contemptuous, superior tone (most GLaDOS lines)
  cold        — flat, menacing, matter-of-fact
  angry       — frustrated, hostile
  triumphant  — smug, victorious
  distressed  — panicked, desperate (rare)
  sincere     — genuine, unguarded (very rare)
  neutral     — informational, no strong affect

Usage:
  python process.py
  python process.py --limit 50          # process first 50 files only (test run)
  python process.py --no-transcribe     # skip whisper, use wiki transcripts
  python process.py --no-tag            # skip all emotion tagging
  python process.py --audio-tag         # enable audio-based SER (downloads ~300MB model on first run)
  python process.py --text-only         # use only Bonsai text classification (no audio model)
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import requests

RAW_DIR = Path("raw_audio")
OUT_DIR = Path("processed_audio")
LLAMACPP_URL = "http://localhost:8080/v1/chat/completions"
EMOTIONS = ["sarcastic", "cold", "angry", "triumphant", "distressed", "sincere", "neutral"]

# Weights for combining audio vs text signals per emotion.
AUDIO_WEIGHT = {
    "sarcastic":  0.3,   # delivery flat, sarcasm mostly in words
    "cold":       0.7,   # very tonal — flat delivery, low energy
    "angry":      0.7,   # high energy / harsh acoustic signal
    "triumphant": 0.4,   # smug tone detectable but words matter more
    "distressed": 0.6,   # panic/fear detectable acoustically
    "sincere":    0.5,
    "neutral":    0.5,
}

# Mapping from superb/wav2vec2-base-superb-er labels → our taxonomy
# Labels: neu, hap, ang, sad
SER_TO_EMOTION = {
    "neu": "neutral",
    "hap": "triumphant",   # GLaDOS "happy" = smug/victorious
    "ang": "angry",
    "sad": "sincere",      # subdued, genuine affect
}

EMOTION_PROMPT = """You are classifying GLaDOS voice lines by emotional tone for TTS training.

Classify the following line into EXACTLY ONE of these emotions:
- sarcastic: dry, contemptuous, superior, ironic
- cold: flat, menacing, matter-of-fact, clinical
- angry: frustrated, hostile, aggressive
- triumphant: smug, victorious, self-satisfied
- distressed: panicked, desperate, fearful
- sincere: genuine, unguarded, emotionally honest
- neutral: informational, no strong affect

Reply with ONLY the emotion word, nothing else.

Line: "{transcript}"
"""

_ser_pipeline = None
_whisper_model = None


def get_ser_pipeline():
    """Lazy-load the SER model (superb/wav2vec2-base-superb-er, ~300MB)."""
    global _ser_pipeline
    if _ser_pipeline is None:
        from transformers import pipeline
        print("Loading speech emotion recognition model (~300MB download on first run)...")
        _ser_pipeline = pipeline(
            "audio-classification",
            model="superb/wav2vec2-base-superb-er",
        )
        print("SER model loaded.")
    return _ser_pipeline


def get_whisper_model():
    """Lazy-load faster-whisper once for the whole run."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        print("Loading Whisper tiny model...")
        _whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("Whisper loaded.")
    return _whisper_model


def transcribe(audio_path: Path) -> str:
    """Transcribe a single audio file with faster-whisper."""
    try:
        model = get_whisper_model()
        segments, _ = model.transcribe(str(audio_path), language="en")
        return " ".join(s.text.strip() for s in segments).strip()
    except Exception as e:
        return ""


def tag_emotion_text(transcript: str) -> dict:
    """Call llama.cpp to classify emotion. Returns {emotion: confidence}."""
    if not transcript:
        return {"neutral": 1.0}
    try:
        resp = requests.post(
            LLAMACPP_URL,
            json={
                "messages": [{"role": "user", "content": EMOTION_PROMPT.format(transcript=transcript)}],
                "temperature": 0.0,
                "max_tokens": 10,
            },
            timeout=30,
        )
        resp.raise_for_status()
        tag = resp.json()["choices"][0]["message"]["content"].strip().lower()
        tag = re.sub(r"[^a-z]", "", tag)
        emotion = tag if tag in EMOTIONS else "neutral"
        return {emotion: 1.0}
    except Exception:
        return {"neutral": 1.0}


def tag_emotion_audio(audio_path: Path) -> dict:
    """Run SER on audio. Returns {emotion: confidence}."""
    try:
        pipe = get_ser_pipeline()
        results = pipe(str(audio_path), top_k=None)
        scores = {e: 0.0 for e in EMOTIONS}
        for item in results:
            our_label = SER_TO_EMOTION.get(item["label"].lower(), "neutral")
            scores[our_label] += item["score"]
        return scores
    except Exception:
        return {"neutral": 1.0}


def combine_signals(audio_scores: dict, text_scores: dict) -> str:
    """Blend audio and text scores with per-emotion weights."""
    combined = {}
    for emotion in EMOTIONS:
        aw = AUDIO_WEIGHT[emotion]
        combined[emotion] = (
            audio_scores.get(emotion, 0.0) * aw
            + text_scores.get(emotion, 0.0) * (1.0 - aw)
        )
    return max(combined, key=lambda e: combined[e])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-transcribe", action="store_true")
    parser.add_argument("--no-tag", action="store_true")
    parser.add_argument("--audio-tag", action="store_true",
                        help="Enable audio SER (~300MB model download on first run)")
    parser.add_argument("--text-only", action="store_true",
                        help="Text classification only, skip audio SER")
    args = parser.parse_args()

    use_audio = args.audio_tag and not args.text_only

    meta_path = RAW_DIR / "metadata.json"
    if not meta_path.exists():
        print("Run scrape.py first to download audio files.")
        return

    with open(meta_path) as f:
        entries = json.load(f)

    if args.limit:
        entries = entries[: args.limit]

    for emotion in EMOTIONS:
        (OUT_DIR / emotion).mkdir(parents=True, exist_ok=True)

    # Preload models upfront
    if not args.no_transcribe:
        get_whisper_model()
    if use_audio:
        get_ser_pipeline()

    dataset = []

    for i, entry in enumerate(entries):
        src = RAW_DIR / entry["filename"]
        if not src.exists():
            continue

        transcript = entry.get("transcript", "")

        if not args.no_transcribe:
            whisper_text = transcribe(src)
            if whisper_text:
                transcript = whisper_text

        emotion = "neutral"
        if not args.no_tag:
            text_scores = tag_emotion_text(transcript) if transcript else {"neutral": 1.0}

            if use_audio:
                audio_scores = tag_emotion_audio(src)
                emotion = combine_signals(audio_scores, text_scores)
            else:
                emotion = max(text_scores, key=lambda e: text_scores[e])

        dest = OUT_DIR / emotion / entry["filename"]
        shutil.copy2(src, dest)

        dataset.append({
            "filename": entry["filename"],
            "path": str(dest),
            "transcript": transcript,
            "emotion": emotion,
            "source": entry.get("source", ""),
        })

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(entries)}] last: {emotion} — {transcript[:60]}")

    dataset_path = OUT_DIR / "dataset.json"
    with open(dataset_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nDone. {len(dataset)} files processed → {OUT_DIR}/")
    counts: dict = {}
    for r in dataset:
        counts[r["emotion"]] = counts.get(r["emotion"], 0) + 1
    for emotion in EMOTIONS:
        print(f"  {emotion:12s}: {counts.get(emotion, 0)}")
    print(f"\nDataset saved to {dataset_path}")


if __name__ == "__main__":
    main()
