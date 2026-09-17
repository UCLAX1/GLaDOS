"""
GLaDOS audio sorting UI — keyboard-driven, uses macOS afplay.

Keys:
  r — robotic
  h — human
  o — other
  v — dota
  SPACE — repeat clip
  s — skip
  d — delete (noise/unusable)
  t — toggle speed (1x / 2x)
  q — quit and save progress

Usage:
  python sort_ui.py                        # interactive folder picker
  python sort_ui.py --src sorted/robotic   # jump straight to a folder
"""

import argparse
import json
import shutil
import subprocess
import sys
import termios
import tty
from pathlib import Path

SORTED_DIR = Path("sorted")
DELETE_DIR = Path("sorted/deleted")
CATEGORIES = {
    "r": "robotic",
    "h": "human",
    "o": "other",
    "v": "dota",   # v for dota (d is taken by delete)
}


# ── terminal helpers ──────────────────────────────────────────────────────────

def flush_stdin():
    import select
    while select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.read(1)


def getch():
    flush_stdin()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def play(path: Path, speed: float):
    subprocess.run(
        ["afplay", "-r", str(speed), str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ── progress ──────────────────────────────────────────────────────────────────

def load_progress(f: Path) -> set:
    return set(json.load(open(f))) if f.exists() else set()


def save_progress(f: Path, done: set):
    json.dump(list(done), open(f, "w"))


# ── folder picker ─────────────────────────────────────────────────────────────

def pick_folder() -> Path:
    """Show a numbered menu of available folders and return the chosen one."""
    candidates = []

    raw = Path("raw_audio")
    if raw.exists():
        candidates.append(raw)

    if SORTED_DIR.exists():
        for sub in sorted(SORTED_DIR.iterdir()):
            if sub.is_dir() and sub.name != "deleted":
                candidates.append(sub)

    if not candidates:
        print("No folders found (expected raw_audio/ or sorted/*).")
        sys.exit(1)

    print("\n=== Pick a folder to sort ===")
    for i, folder in enumerate(candidates):
        wav_count = len(list(folder.rglob("*.wav")))
        print(f"  [{i}] {folder}  ({wav_count} wavs)")
    print()

    while True:
        sys.stdout.write("Enter number: ")
        sys.stdout.flush()
        # read normally (cooked mode) for the picker
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
        if line.strip().isdigit():
            idx = int(line.strip())
            if 0 <= idx < len(candidates):
                return candidates[idx]
        print("  Invalid — try again.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=None, help="Folder to sort (skip picker)")
    args = parser.parse_args()

    if args.src:
        src_dir = Path(args.src)
        if not src_dir.exists():
            print(f"Not found: {src_dir}")
            sys.exit(1)
    else:
        src_dir = pick_folder()

    wavs = sorted(src_dir.rglob("*.wav"))
    if not wavs:
        print(f"No wav files in {src_dir}")
        sys.exit(1)

    # When re-sorting from a sorted/ subfolder, move files; otherwise copy.
    is_resort = src_dir.parts[0] == "sorted" or (
        len(src_dir.parts) > 1 and src_dir.parts[-2] == "sorted"
    )
    transfer = shutil.move if is_resort else shutil.copy2

    for folder in CATEGORIES.values():
        (SORTED_DIR / folder).mkdir(parents=True, exist_ok=True)
    DELETE_DIR.mkdir(parents=True, exist_ok=True)

    progress_file = Path(f".sort_progress_{src_dir.name}.json")
    done = load_progress(progress_file)
    remaining = [w for w in wavs if w.name not in done]
    total = len(wavs)
    speed = 2.0

    mode = "RESORT (files will move)" if is_resort else "SORT (files will copy)"
    print(f"\n{'='*55}")
    print(f"  {mode}")
    print(f"  {len(remaining)} files remaining (of {total}) in {src_dir}/")
    print(f"  [r] robotic  [h] human  [o] other  [v] dota")
    print(f"  [SPACE] repeat  [t] toggle speed  [s] skip  [d] delete  [q] quit")
    print(f"{'='*55}\n")

    i = 0
    while i < len(remaining):
        wav = remaining[i]
        sorted_count = len(done)
        print(f"[{sorted_count+1}/{total}] {wav.name}  [{speed}x]", end="  ", flush=True)
        play(wav, speed)

        while True:
            key = getch()

            if key == " ":
                print("↺ ", end="", flush=True)
                play(wav, speed)

            elif key == "t":
                speed = 1.0 if speed == 2.0 else 2.0
                print(f"[{speed}x] ", end="", flush=True)
                play(wav, speed)

            elif key in CATEGORIES:
                folder = CATEGORIES[key]
                dest = SORTED_DIR / folder / wav.name
                if dest != wav:  # skip if already in the right place
                    transfer(str(wav), str(dest))
                print(f"→ {folder}")
                done.add(wav.name)
                save_progress(progress_file, done)
                i += 1
                break

            elif key == "s":
                print("→ skip")
                done.add(wav.name)
                save_progress(progress_file, done)
                i += 1
                break

            elif key == "d":
                transfer(str(wav), str(DELETE_DIR / wav.name))
                print("→ deleted")
                done.add(wav.name)
                save_progress(progress_file, done)
                i += 1
                break

            elif key == "q":
                print("\nQuitting — progress saved.")
                save_progress(progress_file, done)
                sys.exit(0)

    print(f"\nAll done! {len(done)} files sorted.")
    if progress_file.exists():
        progress_file.unlink()


if __name__ == "__main__":
    main()
