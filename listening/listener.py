#!/usr/bin/env python3
"""
listener.py — GLaDOS speech listener.

Importable module. Uses Silero VAD + faster-whisper.
Endpointing uses a gap counter (not a wall-clock timeout) — speech ends
after GAP_CHUNKS consecutive silent chunks (~640ms by default).

Usage:
    from listening.listener import SpeechListener

    def on_text(text):
        print("Heard:", text)

    listener = SpeechListener(on_transcription=on_text)
    listener.start()   # blocks until Ctrl-C or listener.stop()

Run standalone to test:
    cd ~/GitHub/X1_GLaDOS
    source listening/venv/bin/activate
    python3 listening/listener.py
"""

import collections
import queue
import threading
import time

import numpy as np
import pyaudio
import torch
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad

# ── Audio config ───────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000
CHUNK_FRAMES  = 512                           # ~32ms per chunk at 16kHz
CHUNK_MS      = CHUNK_FRAMES * 1000 // SAMPLE_RATE   # 32

BUFFER_CHUNKS = 800  // CHUNK_MS   # pre-speech ring buffer: ~800ms = 25 chunks
GAP_CHUNKS    = 640  // CHUNK_MS   # silence → end of speech: ~640ms = 20 chunks
SPEC_CHUNKS   = 160  // CHUNK_MS   # silence → speculative transcribe: ~160ms = 5 chunks


def _preprocess(text: str) -> str:
    text = text.lstrip().lstrip("...")
    if text:
        text = text[0].upper() + text[1:]
    return text.strip()


class SpeechListener:
    """
    Captures mic audio, runs VAD, transcribes completed utterances.

    Parameters
    ----------
    on_transcription : callable(str)
        Called on the main listener thread whenever a complete utterance is
        transcribed. Keep it fast (e.g. put onto a queue) — it blocks the
        transcription worker while running.
    on_realtime : callable(str) or None
        Called with in-progress text while the user is still speaking.
        Uses a lighter model; accuracy is lower than final transcription.
    model_final : str
        faster-whisper model for final transcription. Default distil-large-v3.
    model_realtime : str
        faster-whisper model for realtime (only loaded if on_realtime is set).
    language : str or None
        ISO 639-1 code ("en"). None = auto-detect.
    device : str or None
        "cuda" / "cpu". None = auto-detect.
    speech_threshold : float
        Silero VAD confidence threshold (0–1). Default 0.5.
    gap_chunks : int
        Consecutive silent chunks before speech is considered finished.
        Default GAP_CHUNKS (~640ms). Reduce for snappier endpointing.
    speculative : bool
        Transcribe early, after spec_chunks of silence, instead of waiting out
        the full gap. The audio in between is silence by definition, so the
        early result is normally identical — it just arrives sooner. If speech
        resumes, the speculative result is discarded and the utterance is
        transcribed again in full. Costs one extra transcription per false
        endpoint; set False on a CPU-bound host.
    spec_chunks : int
        Silent chunks before speculating. Default SPEC_CHUNKS (~160ms).
    """

    def __init__(
        self,
        on_transcription,
        on_realtime=None,
        model_final="distil-large-v3",
        model_realtime="distil-small.en",
        language="en",
        device=None,
        speech_threshold=0.5,
        gap_chunks=GAP_CHUNKS,
        speculative=True,
        spec_chunks=SPEC_CHUNKS,
    ):
        self._on_transcription = on_transcription
        self._on_realtime      = on_realtime
        self._language         = language
        self._threshold        = speech_threshold
        self._gap_chunks       = gap_chunks
        self._speculative      = speculative
        self._spec_chunks      = min(spec_chunks, gap_chunks)
        self._stop             = threading.Event()

        # Device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device  = device
        compute_type  = "float16" if device == "cuda" else "int8"

        print(f"[listener] device: {device}")
        print(f"[listener] loading VAD ...")
        self._vad = load_silero_vad()

        print(f"[listener] loading transcriber ({model_final}) ...")
        self._transcriber = WhisperModel(
            model_final, device=device, compute_type=compute_type
        )

        self._realtime_model = None
        if on_realtime:
            print(f"[listener] loading realtime transcriber ({model_realtime}) ...")
            self._realtime_model = WhisperModel(
                model_realtime, device=device, compute_type=compute_type
            )

        # Audio state (touched only in the PyAudio callback thread)
        self._pre_buffer   = collections.deque(maxlen=BUFFER_CHUNKS)
        self._speech_chunks: list[bytes] = []
        self._speaking     = False
        self._gap_count    = 0
        # Bumped whenever speech resumes or an utterance ends, so a speculative
        # transcript from a superseded moment can be recognised and dropped.
        self._epoch        = 0
        self._spec_sent    = False

        # Per-utterance stage timings, refreshed just before on_transcription
        # fires. Safe to read from that callback — one worker thread writes it.
        self.last_timing: dict = {}

        # Work queues ((chunks, t_speech_end) tuples)
        self._final_q    = queue.Queue()
        self._realtime_q = queue.Queue(maxsize=1)   # drop if worker is busy

    # ── Transcription helpers ──────────────────────────────────────────────────

    def _transcribe(self, chunks: list[bytes], model=None) -> str:
        if not chunks:
            return ""
        m = model or self._transcriber
        audio = (
            np.frombuffer(b"".join(chunks), dtype=np.int16)
            .astype(np.float32) / 32768.0
        )
        segments, _ = m.transcribe(
            audio,
            language=self._language,
            condition_on_previous_text=False,
        )
        return "".join(s.text for s in segments)

    # ── Worker threads ─────────────────────────────────────────────────────────

    def _transcription_worker(self):
        # Result of the most recent speculative pass, valid only for its epoch.
        spec_epoch, spec_text, spec_stt = None, None, 0.0

        while not self._stop.is_set():
            try:
                kind, epoch, chunks, t_end = self._final_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # Speculative pass: transcribe now, hold the result for the endpoint.
            if kind == "spec":
                t0 = time.time()
                spec_text  = _preprocess(self._transcribe(chunks))
                spec_stt   = time.time() - t0
                spec_epoch = epoch
                continue

            t_pickup = time.time()
            if epoch == spec_epoch:
                # No speech since we speculated — the extra audio was silence,
                # so that transcript still stands. Nothing to compute.
                text, speculative = spec_text, True
            else:
                text, speculative = _preprocess(self._transcribe(chunks)), False
            t_done = time.time()
            spec_epoch, spec_text = None, None

            self.last_timing = {
                # Audio captured, including pre-roll buffer and trailing silence.
                "speech_dur": len(chunks) * CHUNK_MS / 1000,
                # Fixed cost of deciding speech ended — floor on responsiveness.
                "vad_gap":    self._gap_chunks * CHUNK_MS / 1000,
                # Backlog: >0 means the previous turn was still being handled.
                "q_wait":     t_pickup - t_end,
                "stt":        t_done - t_pickup,
                # True when the transcript came free from the gap.
                "speculative": speculative,
                "t_speech_end": t_end,
            }

            if not text:
                continue
            try:
                self._on_transcription(text)
            except Exception as e:
                # A failing callback must not kill this thread — the mic would
                # stay open and the listener would go silently deaf.
                print(f"[listener] on_transcription raised: {e!r}")

    def _realtime_worker(self):
        while not self._stop.is_set():
            try:
                chunks = self._realtime_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if self._realtime_model and self._on_realtime:
                text = _preprocess(self._transcribe(chunks, self._realtime_model))
                if text:
                    try:
                        self._on_realtime(text)
                    except Exception as e:
                        print(f"[listener] on_realtime raised: {e!r}")

    # ── PyAudio callback (runs in its own thread) ──────────────────────────────

    def _audio_callback(self, in_data, frame_count, time_info, status):
        # VAD
        audio_f32 = (
            np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        )
        tensor = torch.from_numpy(audio_f32).unsqueeze(0)   # (1, 512)
        with torch.no_grad():
            prob = self._vad(tensor, SAMPLE_RATE).item()
        is_speech = prob > self._threshold

        if not self._speaking:
            # Always keep rolling pre-speech buffer
            self._pre_buffer.append(in_data)
            if is_speech:
                # Speech started — prepend buffer so word starts aren't clipped
                self._speech_chunks = list(self._pre_buffer)
                self._speaking  = True
                self._gap_count = 0
                self._spec_sent = False
        else:
            self._speech_chunks.append(in_data)

            if is_speech:
                if self._gap_count:
                    # Resumed mid-pause: anything already speculated is stale.
                    self._epoch += 1
                    self._spec_sent = False
                self._gap_count = 0
            else:
                self._gap_count += 1

                # Speculate well before the endpoint, so transcription runs
                # inside the gap instead of after it.
                if (self._speculative and not self._spec_sent
                        and self._gap_count == self._spec_chunks):
                    self._spec_sent = True
                    self._final_q.put(
                        ("spec", self._epoch, list(self._speech_chunks), time.time())
                    )

                if self._gap_count >= self._gap_chunks:
                    # Speech ended
                    chunks = list(self._speech_chunks)
                    self._final_q.put(("final", self._epoch, chunks, time.time()))
                    self._epoch += 1
                    self._spec_sent = False

                    # Drop any stale realtime work
                    try:
                        self._realtime_q.get_nowait()
                    except queue.Empty:
                        pass

                    self._speech_chunks = []
                    self._speaking      = False
                    self._gap_count     = 0

            # Feed realtime worker periodically while speaking
            if self._on_realtime and is_speech:
                if len(self._speech_chunks) % 10 == 0:
                    try:
                        self._realtime_q.put_nowait(list(self._speech_chunks))
                    except queue.Full:
                        pass   # worker still busy — skip this update

        return (None, pyaudio.paContinue)

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        """Open the mic and block until stop() or KeyboardInterrupt."""
        threading.Thread(target=self._transcription_worker, daemon=True).start()
        if self._on_realtime:
            threading.Thread(target=self._realtime_worker, daemon=True).start()

        pa     = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_FRAMES,
            stream_callback=self._audio_callback,
        )
        stream.start_stream()
        print("[listener] listening — press Ctrl-C to stop")

        try:
            while not self._stop.is_set() and stream.is_active():
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    def stop(self):
        """Signal the listener to shut down."""
        self._stop.set()


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    def on_text(text):
        m = listener.last_timing
        print(f"\n>>> {text}")
        print(f"    [timing] speech {m['speech_dur']:.1f}s  vad_gap {m['vad_gap']:.2f}s  "
              f"q_wait {m['q_wait']:.2f}s  stt {m['stt']:.2f}s\n")

    def on_realtime(text):
        print(f"\r... {text}", end="", flush=True)

    listener = SpeechListener(
        on_transcription=on_text,
        on_realtime=on_realtime,
    )
    listener.start()
