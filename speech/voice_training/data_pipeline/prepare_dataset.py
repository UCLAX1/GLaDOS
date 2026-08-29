"""
Prepare Kokoro/StyleTTS2 training datasets from auto_sorted/ voice files.

For each voice type in auto_sorted/ (robotic, human, glitchy, uncertain):
  1. Looks up the wiki transcript from raw_audio/metadata.json
  2. Resamples audio to 22050Hz mono wav
  3. Writes dataset/<voice_type>/wavs/ + dataset/<voice_type>/metadata.csv

metadata.csv format (LJSpeech-style):
  filename|transcript

Usage:
  pip install librosa soundfile
  python prepare_dataset.py
  python prepare_dataset.py --voice-types robotic human   # specific types only
  python prepare_dataset.py --min-duration 1.0            # skip clips shorter than 1s
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

AUTO_DIR = Path("auto_sorted")
RAW_DIR = Path("raw_audio")
OUT_DIR = Path("dataset")
TARGET_SR = 22050


def load_transcript_map() -> dict[str, str]:
    """Load filename → transcript from raw_audio/metadata.json."""
    meta_path = RAW_DIR / "metadata.json"
    if not meta_path.exists():
        print("Warning: raw_audio/metadata.json not found — transcripts will be empty")
        return {}
    with open(meta_path) as f:
        entries = json.load(f)
    return {e["filename"]: e.get("transcript", "") for e in entries}


def resample_and_save(src: Path, dest: Path, target_sr: int) -> float | None:
    """Resample audio to target_sr mono, save as wav. Returns duration in seconds."""
    try:
        import librosa
        import soundfile as sf
        y, sr = librosa.load(str(src), sr=target_sr, mono=True)
        sf.write(str(dest), y, target_sr)
        return len(y) / target_sr
    except Exception as e:
        print(f"  ERROR processing {src.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice-types", nargs="+", default=None,
                        help="Voice types to process (default: all in auto_sorted/)")
    parser.add_argument("--min-duration", type=float, default=0.5,
                        help="Skip clips shorter than this many seconds (default: 0.5)")
    args = parser.parse_args()

    try:
        import librosa  # noqa
        import soundfile  # noqa
    except ImportError:
        print("Install dependencies: pip install librosa soundfile")
        return

    if not AUTO_DIR.exists():
        print(f"{AUTO_DIR}/ not found — run classify_voice.py first")
        return

    transcript_map = load_transcript_map()

    # Discover voice type folders
    voice_dirs = [d for d in AUTO_DIR.iterdir() if d.is_dir()]
    if args.voice_types:
        voice_dirs = [d for d in voice_dirs if d.name in args.voice_types]

    if not voice_dirs:
        print(f"No voice type folders found in {AUTO_DIR}/")
        return

    total_stats = {}

    for voice_dir in sorted(voice_dirs):
        voice_type = voice_dir.name
        wavs_in = list(voice_dir.glob("*.wav"))

        if not wavs_in:
            continue

        print(f"\n[{voice_type}] {len(wavs_in)} files")

        out_wavs = OUT_DIR / voice_type / "wavs"
        out_wavs.mkdir(parents=True, exist_ok=True)

        rows = []
        skipped = 0
        total_duration = 0.0

        for wav in sorted(wavs_in):
            transcript = transcript_map.get(wav.name, "").strip()
            if not transcript:
                skipped += 1
                continue

            dest = out_wavs / wav.name
            duration = resample_and_save(wav, dest, TARGET_SR)

            if duration is None:
                skipped += 1
                continue

            if duration < args.min_duration:
                dest.unlink(missing_ok=True)
                skipped += 1
                continue

            rows.append((wav.stem, transcript))
            total_duration += duration

        # Write metadata.csv
        if rows:
            meta_path = OUT_DIR / voice_type / "metadata.csv"
            with open(meta_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter="|")
                for row in rows:
                    writer.writerow(row)

        mins = total_duration / 60
        print(f"  kept: {len(rows)}  skipped: {skipped}  duration: {mins:.1f} min")
        print(f"  → {OUT_DIR}/{voice_type}/")

        total_stats[voice_type] = {
            "files": len(rows),
            "skipped": skipped,
            "duration_min": round(mins, 1),
        }

    # Summary
    print(f"\n{'='*50}")
    print("Dataset summary:")
    for vt, stats in total_stats.items():
        flag = ""
        if stats["files"] < 50:
            flag = " ⚠ low (aim for 100+)"
        elif stats["files"] >= 100:
            flag = " ✓"
        print(f"  {vt:12s}: {stats['files']:4d} files  {stats['duration_min']:5.1f} min{flag}")

    print(f"\nDatasets saved to {OUT_DIR}/")
    print("Next: upload to RunPod/Colab and fine-tune Kokoro from hexgrad/Kokoro-82M")

    # Save summary json
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(total_stats, f, indent=2)


if __name__ == "__main__":
    main()
