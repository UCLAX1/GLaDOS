"""
Clean up noisy transcripts in dataset/ metadata.csv files.

Removes wiki formatting artifacts like:
  - Trailing ' " | |"' noise from the scraper
  - Leading/trailing quotes and whitespace
  - Sound effect annotations like [bzzzzzt], [music], etc.
  - Lines that are too short or mostly noise

Usage:
  python clean_transcripts.py           # clean all voice types
  python clean_transcripts.py --dry-run # preview changes only
"""

import argparse
import csv
import re
from pathlib import Path

DATASET_DIR = Path("dataset")


def clean(text: str) -> str:
    # Remove wiki table separators that leaked through
    text = re.sub(r'"\s*\|\s*\|"?$', '', text)
    text = re.sub(r'\s*"\s*\|\s*\|"?\s*$', '', text)
    # Remove surrounding quotes
    text = text.strip().strip('"').strip("'").strip()
    # Remove sound effect annotations
    text = re.sub(r'\[.*?\]', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove trailing punctuation noise
    text = text.rstrip('-').rstrip().rstrip('--').strip()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for meta_path in sorted(DATASET_DIR.glob("*/metadata.csv")):
        voice_type = meta_path.parent.name
        rows = []
        skipped = 0

        with open(meta_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            for row in reader:
                if len(row) < 2:
                    skipped += 1
                    continue
                filename = row[0].strip()
                transcript = clean("|".join(row[1:]))  # rejoin in case transcript had |
                if len(transcript) < 3:
                    skipped += 1
                    continue
                rows.append((filename, transcript))

        print(f"[{voice_type}] {len(rows)} kept, {skipped} skipped")
        if args.dry_run:
            print("  Sample cleaned transcripts:")
            for fn, tr in rows[:3]:
                print(f"    {fn}: {tr[:80]}")
            continue

        with open(meta_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="|")
            for row in rows:
                writer.writerow(row)

    if args.dry_run:
        print("\nDry run — no files written.")
    else:
        print("\nDone. metadata.csv files cleaned.")


if __name__ == "__main__":
    main()
