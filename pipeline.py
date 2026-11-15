#!/usr/bin/env python3
"""
pipeline.py — turn manager: decides WHAT GLaDOS responds to and WHEN.

No Qt, no MuJoCo, no audio — just the logic, so the GUI (gui_launch.py) and the
terminal runner (run_glados.py) share it and it can be tested on its own.

    listener ──speculative / final / cancelled──►┐
    camera   ──person_entered / gesture─────────►│ TurnManager ──► brain ──► speak()
    admin    ──typed text / visual check────────►┘      ▲
    speech daemon ──speak-start / speak-end─────────────┘

Latency + "retroactive" behaviour
---------------------------------
* Early start. The listener reports a transcript ~160 ms into a pause, long
  before the end-of-speech timeout. The LLM starts on it immediately; its
  output is held back until the listener confirms the utterance really ended.
  If the speaker carries on instead, the held reply is thrown away.
* Retroactive merge. If more speech arrives before GLaDOS has made a sound for
  the previous utterance, the two were one thought split by a pause: the
  unfinished reply is cancelled and she answers the combined text instead.
* Barge-in. If she is already talking, new speech stops her (audio included)
  and starts a fresh turn.
* A reply that was cancelled before being heard leaves no trace in the
  conversation history.
* Inputs are handled first-in-first-out; camera events only run when idle.
"""

import collections
import threading
import time

from central_ai.brain import build_turn

import re

# Sounds that are not a request. Dropping them before the LLM saves a whole turn
# and stops her answering every "um" (or the speech recogniser's guess at noise).
_FILLER = {"um", "umm", "uh", "uhh", "uhm", "hmm", "hm", "mm", "mmm", "mhm", "er", "erm",
           "ah", "eh", "oh", "huh", "you"}


def is_filler(text: str) -> bool:
    words = re.findall(r"[a-z']+", text.lower())
    return not words or all(w in _FILLER for w in words)


MERGE_WINDOW = 8.0    # s — how long an unspoken utterance can still absorb a follow-up
HELD_TIMEOUT = 3.0    # s — a speculative reply nobody confirmed is dropped after this


class Turn:
    def __init__(self, heard: str, event: str, held: bool = False,
                 raw: str = "", base: str = ""):
        self.heard     = heard          # full text GLaDOS answers (may include merged speech)
        self.event     = event          # speech | person_entered | gesture | visual_check
        self.held      = held           # speculative: generate, but don't speak yet
        self.raw       = raw            # the listener's transcript for this utterance alone
        self.base      = base           # earlier unspoken speech merged in front of it
        self.release   = threading.Event()
        self.cancel    = threading.Event()
        self.dead      = False          # cancelled — never (further) spoken
        self.done      = False          # LLM finished and the turn is in history
        self.spoken    = False          # at least one chunk has started playing
        self.announced = False
        self.dispatched = 0
        self.buffer: list[str] = []
        self.user_json = None
        self.spec_hit  = False
        self.t_created     = time.time()
        self.t_voice_end   = None       # when the human actually stopped talking
        self.t_llm_start   = None
        self.t_first_chunk = None
        self.t_first_say   = None
        self.t_first_sound = None


class TurnManager:
    """
    Parameters
    ----------
    brain          : GladosBrain
    vision         : object with .snapshot() -> dict | None
    speak          : callable(text) — queue one chunk for the speech daemon
    cancel_speech  : callable() — drop queued speech and stop current playback
    on_event       : callable(name, **data) — UI hook (called from worker threads):
                       input(json, turn)   chunk(text)     queue(items)
                       speaking(text)      spoke_done()    silent(raw)     llm_timing(**numbers)
                       status(stage)       gesture(gesture) latency(**numbers)
                       log(text)
    get_notes      : callable() -> list[str] — operator notes for the next turn
    """

    def __init__(self, brain, vision, speak, cancel_speech,
                 on_event=None, get_notes=None):
        self.brain         = brain
        self.vision        = vision
        self._speak        = speak
        self._cancel_speech = cancel_speech
        self._on_event     = on_event or (lambda name, **kw: None)
        self._get_notes    = get_notes or (lambda: [])

        self.deafened       = threading.Event()   # ignore the mic
        self.auto_interrupt = threading.Event()   # speech stops her mid-sentence
        self.auto_interrupt.set()

        self._q        = collections.deque()      # FIFO of Turn
        self._cv       = threading.Condition()
        self._st       = threading.Lock()         # guards Turn.dead/done + mirror
        self._active: Turn | None = None          # turn the brain is on / was last on
        self._spec:   Turn | None = None          # current speculative (held) turn
        self._carry    = None                     # (text, t) unspoken speech awaiting merge
        self._mirror   = collections.deque()      # (text, turn) queued in the daemon, unplayed
        self._speaking = False
        self._discards: list[str] = []            # user_json of turns to drop from history

    # ── UI helpers ────────────────────────────────────────────────────────────

    def _emit(self, name, **kw):
        try:
            self._on_event(name, **kw)
        except Exception as e:           # a UI bug must never stall the pipeline
            print(f"[pipeline] on_event({name}) raised: {e!r}")

    def _emit_queue(self):
        with self._st:
            items = [t for t, _ in self._mirror]
        self._emit("queue", items=items)

    def is_speaking(self) -> bool:
        return self._speaking or bool(self._mirror)

    # ── Inputs ────────────────────────────────────────────────────────────────

    def speech_speculative(self, text: str):
        """Listener: a pause was transcribed; the utterance is probably over."""
        text = (text or "").strip()
        if not text or self.deafened.is_set() or is_filler(text):
            return
        now = time.time()
        with self._cv:
            self._drop_spec_locked()
            base = self._absorb_unspoken_locked(now)
            turn = Turn((base + " " + text).strip(), "speech", held=True, raw=text, base=base)
            self._spec = turn
            self._q.append(turn)
            self._cv.notify()

    def speculation_cancelled(self):
        """Listener: the speaker carried on — the speculative transcript is stale."""
        with self._cv:
            self._drop_spec_locked()

    def speech_final(self, text: str, t_voice_end: float | None = None, from_mic: bool = True):
        """Listener (or admin): a complete utterance."""
        text = (text or "").strip()
        now  = time.time()
        with self._cv:
            if not text or (from_mic and (self.deafened.is_set() or is_filler(text))):
                self._drop_spec_locked()
                return
            spec = self._spec
            if spec is not None and not spec.dead and spec.raw == text:
                # The early start was right: let the held reply out.
                self._spec = None
                spec.t_voice_end = t_voice_end or now
                spec.spec_hit    = True
                self._barge_in_locked(keep=spec)
                spec.held = False
                spec.release.set()
            else:
                self._drop_spec_locked()
                base = self._absorb_unspoken_locked(now)
                self._barge_in_locked()
                turn = Turn((base + " " + text).strip(), "speech", raw=text, base=base)
                turn.t_voice_end = t_voice_end or now
                self._q.append(turn)
                self._cv.notify()
        self._emit("status", stage="thinking")

    def camera_event(self, event: str):
        """Vision trigger / 'input visuals' button. Only runs when nothing else is going on."""
        with self._cv:
            act = self._active
            busy = (self._q or self.is_speaking()
                    or (act is not None and not act.done and not act.dead))
            if busy and event != "visual_check":
                return
            turn = Turn("", event)
            turn.t_voice_end = time.time()
            self._q.append(turn)
            self._cv.notify()
        self._emit("status", stage="thinking")

    def interrupt(self):
        """Manual interrupt / mute: stop talking and drop the reply in progress."""
        with self._cv:
            act = self._active
            if act is not None and not act.done:
                self._kill_locked(act)
            self._stop_audio_locked()

    def say_direct(self, text: str):
        """Speak text verbatim, bypassing the LLM (admin force-speak)."""
        self._speak(text)
        with self._st:
            self._mirror.append((text, None))
        self._emit_queue()

    # ── Internals (call with self._cv held) ───────────────────────────────────

    def _kill_locked(self, turn: Turn):
        with self._st:
            turn.dead = True
            done = turn.done
            mine = [m for m in self._mirror if m[1] is turn]
            for m in mine:
                self._mirror.remove(m)
        turn.cancel.set()
        if mine:
            self._cancel_speech()           # its chunks are queued in the daemon
        if done and turn.user_json:
            self._discards.append(turn.user_json)   # never heard → forget it

    def _drop_spec_locked(self):
        spec, self._spec = self._spec, None
        if spec is None:
            return
        if spec.base:                        # give back what it had absorbed
            self._carry = (spec.base, time.time())
        if spec in self._q:
            self._q.remove(spec)
        self._kill_locked(spec)

    def _absorb_unspoken_locked(self, now: float) -> str:
        """
        Collect speech GLaDOS has not answered out loud yet, cancelling the
        replies that were in progress for it. Returns the text to put in front
        of the new utterance.
        """
        base = ""
        if self._carry and now - self._carry[1] < MERGE_WINDOW:
            base = self._carry[0]
        self._carry = None

        act = self._active
        if (act is not None and act is not self._spec and not act.dead
                and not act.spoken and now - act.t_created < MERGE_WINDOW):
            if act.event == "speech":
                base = act.heard             # already contains anything older
            self._kill_locked(act)           # camera turns are simply superseded

        for t in list(self._q):              # queued but not started yet
            if t is self._spec:
                continue
            self._q.remove(t)
            if t.event == "speech":
                base = (base + " " + t.heard).strip()
        return base

    def _stop_audio_locked(self):
        with self._st:
            self._mirror.clear()
        self._speaking = False
        self._cancel_speech()
        self._emit("queue", items=[])
        self._emit("speaking", text="")

    def _barge_in_locked(self, keep: Turn | None = None):
        if not self.auto_interrupt.is_set() or not self.is_speaking():
            return
        act = self._active
        if act is not None and act is not keep and not act.done:
            self._kill_locked(act)
        self._stop_audio_locked()

    # ── Speech daemon feedback ────────────────────────────────────────────────

    def tts_event(self, tag: str, payload: dict):
        """Feed '[speak-start] {...}' / '[speak-end] {...}' / '[cancelled] {...}' here."""
        if tag == "speak-start":
            text, turn = payload.get("text", ""), None
            with self._st:
                for m in self._mirror:
                    if m[0] == text:
                        turn = m[1]
                        self._mirror.remove(m)
                        break
            self._speaking = True
            self._emit("status", stage="speaking")
            self._emit("speaking", text=text)
            self._emit_queue()
            if turn is not None:
                turn.spoken = True
                if turn.t_first_sound is None:
                    turn.t_first_sound = time.time()
                    self._report_latency(turn)
        elif tag == "speak-end":
            self._speaking = False
            self._emit("spoke_done")
            if not self._mirror:
                self._emit("status", stage="listening")
        elif tag == "cancelled":
            self._speaking = False

    def _report_latency(self, t: Turn):
        ref = t.t_voice_end or t.t_created
        self._emit(
            "latency",
            total=t.t_first_sound - ref,                               # end of speech → first sound
            llm=(t.t_first_chunk - ref) if t.t_first_chunk else None,  # … → first words from the LLM
            tts=(t.t_first_sound - t.t_first_say) if t.t_first_say else None,
            head_start=(ref - t.t_llm_start) if t.t_llm_start else None,
            speculative=t.spec_hit,
            event=t.event,
        )

    # ── Brain loop ────────────────────────────────────────────────────────────

    def run(self):
        """Blocking worker loop — run in a daemon thread."""
        while True:
            with self._cv:
                while not self._q:
                    self._cv.wait()
                turn = self._q.popleft()
                self._active = turn
            if not turn.dead:
                try:
                    self._process(turn)
                except Exception as e:
                    print(f"[pipeline] turn failed: {e!r}")
                    self._emit("status", stage="listening")

    def _announce(self, turn: Turn):
        if not turn.announced:
            turn.announced = True
            self._emit("input", json=turn.user_json, turn=turn)

    def _say(self, turn: Turn, chunk: str):
        self._announce(turn)
        if turn.t_first_say is None:
            turn.t_first_say = time.time()
            self._emit("status", stage="tts")
        with self._st:
            if turn.dead:
                return
            self._mirror.append((chunk, turn))
            self._speak(chunk)      # inside the lock: a concurrent kill then sees it and cancels it
        turn.dispatched += 1
        self._emit("chunk", text=chunk)
        self._emit_queue()

    def _flush(self, turn: Turn):
        buffered, turn.buffer = turn.buffer, []
        for chunk in buffered:
            self._say(turn, chunk)

    def _process(self, turn: Turn):
        with self._cv:
            discards, self._discards = self._discards, []
        for uj in discards:
            self.brain.discard(uj)

        turn.user_json = build_turn(
            heard=turn.heard,
            vision=self.vision.snapshot() if self.vision else None,
            event=turn.event,
            notes=self._get_notes() or None,
        )
        if not turn.held:
            self._announce(turn)

        turn.t_llm_start = time.time()
        exhausted = False
        gen = self.brain.respond_stream(turn.user_json)
        try:
            for chunk in gen:
                if turn.cancel.is_set():
                    break
                if turn.t_first_chunk is None:
                    turn.t_first_chunk = time.time()
                if turn.held and not turn.release.is_set():
                    turn.buffer.append(chunk)       # speculative: hold it back
                    continue
                self._flush(turn)
                self._say(turn, chunk)
            else:
                exhausted = True
        finally:
            gen.close()                             # closes the HTTP stream → server stops generating

        if exhausted and turn.held and not turn.release.is_set():
            # Reply is ready before the listener confirmed the utterance ended.
            deadline = time.time() + HELD_TIMEOUT
            while (not turn.release.is_set() and not turn.cancel.is_set()
                   and time.time() < deadline):
                turn.release.wait(0.01)
            if not turn.release.is_set() and not turn.cancel.is_set():
                with self._cv:
                    if self._spec is turn:
                        self._spec = None
                    self._kill_locked(turn)

        with self._st:
            if exhausted and not turn.dead:
                turn.done = True
            dead = turn.dead

        if dead:
            if exhausted:
                self.brain.discard(turn.user_json)  # generated but never heard
            self._emit("log", text=f"[turn] cancelled: {turn.heard!r}")
            return

        self._flush(turn)
        self._announce(turn)
        response = self.brain.last_response or {}
        if turn.dispatched == 0:
            self._emit("silent", raw=response)
            if not self.is_speaking():
                self._emit("status", stage="listening")
        self._emit("gesture", gesture=response.get("gesture", "idle"))
        self._emit("llm_timing", **(self.brain.last_timing or {}))
