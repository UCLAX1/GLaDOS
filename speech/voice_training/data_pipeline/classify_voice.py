"""
Auto-classify GLaDOS audio by voice type (robotic / human / glitchy).

Uses your manually sorted files in sorted/ as reference examples to build
acoustic profiles, then classifies all raw_audio/ files against those profiles.
Outputs to auto_sorted/ — does NOT touch your existing sorted/ folder.

Approach:
  - Extract acoustic features (spectral flatness, MFCCs, pitch variance, ZCR)
  - Compute mean feature profile per voice type from your reference files
  - Nearest-centroid classify each raw_audio file
  - Files too far from any centroid → auto_sorted/uncertain/

Usage:
  pip install librosa numpy
  python classify_voice.py
  python classify_voice.py --threshold 0.5   # stricter uncertain cutoff (0-1)
  python classify_voice.py --dry-run         # print classifications, don't copy
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

RAW_DIR = Path("raw_audio")
SORTED_DIR = Path("sorted")
OUT_DIR = Path("auto_sorted")

# Voice type folders to look for in sorted/ (ignores emotion subfolders)
VOICE_TYPES = ["robotic", "human", "glitchy"]


def extract_features(path: Path) -> np.ndarray | None:
    """Extract acoustic feature vector from a wav file."""
    try:
        import librosa
        y, sr = librosa.load(str(path), sr=16000, mono=True, duration=10.0)
        if len(y) < sr * 0.3:  # skip clips shorter than 0.3s
            return None

        features = []

        # Spectral flatness — high = synthetic/robotic, low = natural/human
        flatness = librosa.feature.spectral_flatness(y=y)
        features.append(np.mean(flatness))
        features.append(np.std(flatness))

        # Spectral centroid — brightness
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(np.mean(centroid) / sr)  # normalize
        features.append(np.std(centroid) / sr)

        # MFCCs (first 13) — timbre fingerprint
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfcc, axis=1).tolist())
        features.extend(np.std(mfcc, axis=1).tolist())

        # Zero crossing rate — glitchy files have high ZCR spikes
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))

        # Pitch variance — robotic = flat pitch, human = variable
        try:
            f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
            voiced_f0 = f0[voiced] if voiced is not None and np.any(voiced) else np.array([])
            features.append(float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0.0)
            features.append(float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0)
        except Exception:
            features.extend([0.0, 0.0])

        return np.array(features, dtype=np.float32)
    except Exception as e:
        return None


def build_profiles(sorted_dir: Path) -> dict[str, np.ndarray]:
    """Build mean feature vector for each voice type from sorted/ reference files."""
    profiles = {}
    for voice_type in VOICE_TYPES:
        type_dir = sorted_dir / voice_type
        if not type_dir.exists():
            print(f"  Warning: no reference folder found for '{voice_type}' — skipping")
            continue

        wavs = list(type_dir.rglob("*.wav"))
        if not wavs:
            print(f"  Warning: '{voice_type}' folder is empty — skipping")
            continue

        print(f"  Building profile for '{voice_type}' from {len(wavs)} reference files...")
        vecs = []
        for wav in wavs:
            v = extract_features(wav)
            if v is not None:
                vecs.append(v)

        if vecs:
            profiles[voice_type] = np.mean(vecs, axis=0)
            print(f"    → {len(vecs)} files used")
        else:
            print(f"    → no valid features extracted")

    return profiles


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / norm)


def classify(vec: np.ndarray, profiles: dict[str, np.ndarray], threshold: float) -> str:
    distances = {vt: cosine_distance(vec, profile) for vt, profile in profiles.items()}
    best = min(distances, key=lambda k: distances[k])
    if distances[best] > threshold:
        return "uncertain"
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="Cosine distance threshold above which a file goes to uncertain/ (default 0.35)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print classifications without copying files")
    args = parser.parse_args()

    try:
        import librosa  # noqa
    except ImportError:
        print("Install librosa first: pip install librosa")
        return

    print("Building voice type profiles from sorted/ reference files...")
    profiles = build_profiles(SORTED_DIR)

    if not profiles:
        print("No reference profiles built — make sure sorted/ has robotic/, human/, or glitchy/ subfolders with .wav files.")
        return

    print(f"\nProfiles built for: {list(profiles.keys())}")
    print(f"Threshold: {args.threshold} (lower = stricter)")

    wavs = sorted(RAW_DIR.glob("*.wav"))
    if not wavs:
        print(f"\nNo .wav files found in {RAW_DIR}/")
        return

    print(f"\nClassifying {len(wavs)} files...")

    # Create output dirs
    if not args.dry_run:
        for vt in list(profiles.keys()) + ["uncertain"]:
            (OUT_DIR / vt).mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    results = []

    for i, wav in enumerate(wavs):
        vec = extract_features(wav)
        if vec is None:
            label = "uncertain"
        else:
            label = classify(vec, profiles, args.threshold)

        counts[label] = counts.get(label, 0) + 1
        results.append({"filename": wav.name, "voice_type": label})

        if not args.dry_run:
            dest_dir = OUT_DIR / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wav, dest_dir / wav.name)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(wavs)}]")

    # Save results
    if not args.dry_run:
        with open(OUT_DIR / "classifications.json", "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Done. {len(wavs)} files classified:")
    for label in list(profiles.keys()) + ["uncertain"]:
        print(f"  {label:12s}: {counts.get(label, 0)}")

    if not args.dry_run:
        print(f"\nOutput in {OUT_DIR}/  —  classifications saved to {OUT_DIR}/classifications.json")
        print("Review auto_sorted/, then copy anything that looks right into sorted/.")


if __name__ == "__main__":
    main()
