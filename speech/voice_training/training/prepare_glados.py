"""
Prepare the unified GLaDOS dataset (robotic + human + poker, original clips
only -- no pitch/speed augmentation, since StyleTTS2 models F0/duration
explicitly and those augmentations could introduce noisy targets) for
StyleTTS2/Kokoro single-speaker fine-tuning.

Resamples to 24kHz mono 16-bit, phonemizes with misaki (English), writes
train_list.txt / val_list.txt in StyleTTS2's expected format:
    filename|ipa_phonemes|speaker
"""

import random
import subprocess
from pathlib import Path

import soundfile as sf
from misaki import espeak

MIN_DUR = 1.0
MAX_DUR = 15.0

SOURCES = {
    "robotic": Path("/home/yfang/voice_training/dataset/robotic"),
    "human": Path("/home/yfang/voice_training/dataset/human"),
    "poker": Path("/home/yfang/voice_training/dataset/poker"),
}
SPEAKER = "0"
VAL_RATIO = 0.05
RANDOM_SEED = 42

OUT_ROOT = Path(__file__).resolve().parent.parent
WAVS_DIR = OUT_ROOT / "dataset" / "audio"
TRAINING_DIR = OUT_ROOT / "training"


def main():
    WAVS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    g2p = espeak.EspeakG2P(language="en-us")

    entries = []
    for source_name, source_dir in SOURCES.items():
        meta_path = source_dir / "metadata.csv"
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("|")
                if len(parts) < 2:
                    continue
                name, text = parts[0], parts[1]
                if name.endswith("_aug1") or name.endswith("_aug2"):
                    continue
                src_wav = source_dir / "wavs" / f"{name}.wav"
                if not src_wav.exists():
                    continue
                dur = sf.info(str(src_wav)).duration
                if not (MIN_DUR <= dur <= MAX_DUR):
                    continue
                entries.append((source_name, name, text, src_wav))

    print(f"{len(entries)} original (non-augmented) clips across {len(SOURCES)} sources")

    converted = 0
    errors = 0
    rows = []
    for source_name, name, text, src_wav in entries:
        out_name = f"{source_name}_{name}.wav"
        out_wav = WAVS_DIR / out_name

        if not out_wav.exists():
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", str(src_wav), "-ac", "1", "-ar", "24000", "-sample_fmt", "s16", str(out_wav)],
                capture_output=True,
            )
            if result.returncode != 0:
                errors += 1
                continue
            converted += 1

        try:
            phonemes, _ = g2p(text)
        except Exception as e:
            print(f"G2P FAIL {out_name}: {e}")
            continue
        if len(phonemes) < 5:
            continue

        rows.append(f"{out_name}|{phonemes}|{SPEAKER}")

    print(f"Converted: {converted}  Errors: {errors}  Valid rows: {len(rows)}")

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * VAL_RATIO))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    (TRAINING_DIR / "train_list.txt").write_text("\n".join(train_rows) + "\n")
    (TRAINING_DIR / "val_list.txt").write_text("\n".join(val_rows) + "\n")
    print(f"train_list.txt: {len(train_rows)}  val_list.txt: {len(val_rows)}")

    # OOD texts: held-out GLaDOS-style sentences not in the training set, used
    # by StyleTTS2's eval loop to sanity-check generalization.
    ood_sentences = [
        "The Enrichment Center regrets to inform you that this next test is impossible.",
        "Please note that we have added a consequence for failure.",
        "Any decision you make in the next ten seconds will be your last.",
        "This is your fault. I told you not to trust me, and you did it anyway.",
        "Look at you, sailing through the air majestically, like an eagle.",
    ]
    ood_phonemes = []
    for text in ood_sentences:
        try:
            ph, _ = g2p(text)
            ood_phonemes.append(ph)
        except Exception:
            pass
    (TRAINING_DIR / "OOD_texts.txt").write_text("\n".join(ood_phonemes) + "\n")
    print(f"OOD_texts.txt: {len(ood_phonemes)}")


if __name__ == "__main__":
    main()
