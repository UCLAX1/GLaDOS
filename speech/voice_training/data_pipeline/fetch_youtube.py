#!/usr/bin/env python3
"""
fetch_youtube.py — download a YouTube video, VAD-split into chunks, transcribe with Whisper.

Usage:
  python fetch_youtube.py --url URL --out dataset/poker

Outputs:
  dataset/poker/wavs/*.wav   (2-15s chunks)
  dataset/poker/metadata.csv (filename|transcript, LJSpeech format)
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

MIN_DUR = 2.0   # seconds
MAX_DUR = 15.0

def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", default="dataset/poker")
    parser.add_argument("--whisper_model", default="medium")
    args = parser.parse_args()

    out = Path(args.out)
    wavs = out / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)

    # 1. Download
    print("Downloading...")
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.wav"
        run(["yt-dlp", "-x", "--audio-format", "wav",
             "--postprocessor-args", "ffmpeg:-ar 22050 -ac 1",
             "-o", str(raw), args.url])

        # 2. VAD split via ffmpeg silence detection
        print("Splitting on silence...")
        result = subprocess.run(
            ["ffmpeg", "-i", str(raw), "-af",
             "silencedetect=noise=-35dB:d=0.4", "-f", "null", "-"],
            capture_output=True, text=True
        )
        output = result.stderr

        # parse silence intervals
        starts = [float(m) for m in re.findall(r"silence_start: ([\d.]+)", output)]
        ends   = [float(m) for m in re.findall(r"silence_end: ([\d.]+)", output)]

        # build speech segments from gaps between silences
        duration_line = re.search(r"Duration: (\d+):(\d+):([\d.]+)", output)
        total = 0.0
        if duration_line:
            h, m, s = duration_line.groups()
            total = int(h)*3600 + int(m)*60 + float(s)

        # pair silence_end[i] → silence_start[i+1]
        speech_starts = [0.0] + ends
        speech_ends   = starts + ([total] if total else [])

        segments = []
        for ss, se in zip(speech_starts, speech_ends):
            dur = se - ss
            if MIN_DUR <= dur <= MAX_DUR:
                segments.append((ss, se))

        print(f"  {len(segments)} segments in [{MIN_DUR}s, {MAX_DUR}s] range")

        # 3. Extract chunks
        chunk_paths = []
        for i, (ss, se) in enumerate(segments):
            out_wav = wavs / f"poker_{i:04d}.wav"
            run(["ffmpeg", "-y", "-ss", str(ss), "-to", str(se),
                 "-i", str(raw), str(out_wav)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            chunk_paths.append(out_wav)

        # 4. Transcribe with Whisper
        print(f"Transcribing {len(chunk_paths)} chunks with whisper ({args.whisper_model})...")
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel(args.whisper_model, device="cuda", compute_type="float16")
        except Exception:
            from faster_whisper import WhisperModel
            model = WhisperModel(args.whisper_model, device="cpu", compute_type="int8")

        rows = []
        for i, wav_path in enumerate(chunk_paths):
            segments_w, _ = model.transcribe(str(wav_path), language="en")
            text = " ".join(s.text.strip() for s in segments_w).strip()
            if len(text.split()) >= 4:   # skip near-empty transcripts
                rows.append(f"{wav_path.stem}|{text}")
            else:
                wav_path.unlink()   # drop useless chunk
            if (i+1) % 25 == 0:
                print(f"  [{i+1}/{len(chunk_paths)}]")

        meta = out / "metadata.csv"
        meta.write_text("\n".join(rows) + "\n")
        print(f"\nDone. {len(rows)} clips → {out}/")
        print(f"Metadata: {meta}")

if __name__ == "__main__":
    main()
