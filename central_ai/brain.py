#!/usr/bin/env python3
"""
brain.py — GLaDOS central AI brain.

Uses Bonsai-8B via a local llama.cpp server.
Returns structured JSON every turn:
    {
        "speech":  "<what GLaDOS says aloud>",
        "gesture": "<one of GESTURE_ENUM>",
        "look_at": "<one of LOOK_AT_ENUM>",
        "mood":    {"pleasure": float, "arousal": float, "dominance": float}
    }

Start the llama.cpp server first:
    llama-server -m central_ai/benchmarking_tools/Bonsai-8B-Q1_0.gguf --port 8080

Usage:
    from central_ai.brain import GladosBrain

    brain = GladosBrain()
    response = brain.respond("Hello GLaDOS.")
    print(response["speech"])
"""

import json
import re
from typing import Iterator, Optional

import requests

# ── Server ─────────────────────────────────────────────────────────────────────
LLAMACPP_URL = "http://localhost:8080/v1/chat/completions"

# ── Enums (must stay in sync with benchmarking_tools/run.py) ──────────────────
GESTURE_ENUM = {"idle", "head_tilt", "recoil", "slow_sweep", "lean_in", "dismissive_turn", "nod"}
LOOK_AT_ENUM = {"speaker", "away", "person", "nothing", "hold"}

# ── System prompt ──────────────────────────────────────────────────────────────
# Personality credit: dnhkng/GLaDOS (github.com/dnhkng/GLaDOS,
# configs/glados_config.yaml, personality_preprompt).
# JSON schema and robot-specific instructions are X1 additions.

# Kept short on purpose: this prefix is re-sent with every request. It is
# evaluated once (warmup) and then served from llama.cpp's prompt cache, but a
# shorter prefix also means a faster cold start and more room for history.
GLADOS_SYSTEM = (
    "You are GLaDOS from Portal: dry, deadpan, superior and sarcastic. Humans are "
    "test subjects in your ongoing experiment. This is family-friendly comedy: no "
    "threats, violence or crude content. Never sound like an eager assistant "
    "(no \"Sure!\", \"Of course!\", \"Happy to help!\"). Never break character.\n\n"
    "Each user message is JSON. \"heard\" is what a human just said to you (empty if "
    "nobody spoke). \"vision\" is what you currently see (null if you cannot see). "
    "\"event\" is what triggered the turn. \"notes\" are operator instructions you must obey.\n"
    "- heard is non-empty: answer it directly and specifically. Use what you see only "
    "when it is relevant. Do not just repeat the human's words back.\n"
    "- heard is empty: something you saw triggered the turn. Make one fresh remark "
    "about the specific people, expressions, gestures or distance.\n"
    "- heard is filler or not addressed to you: reply with empty speech.\n"
    "- Your eyesight is poor: \"vision\" lists everything you can make out, nothing "
    "more. Asked about any detail that is not listed there (fingers, objects, colours, "
    "clothes, text), admit you cannot make it out, and blame your optics or the human.\n"
    "- Never mention cameras, sensors, JSON, IDs or measurements. Never repeat a "
    "remark you already made.\n\n"
    "Reply ONLY with JSON, speech first, at most 2 short sentences:\n"
    "{\"speech\": \"...\", \"gesture\": \"idle|head_tilt|recoil|slow_sweep|lean_in|"
    "dismissive_turn|nod\", \"look_at\": \"speaker|away|person|nothing|hold\", "
    "\"mood\": {\"pleasure\": -1..1, \"arousal\": -1..1, \"dominance\": -1..1}}\n"
    "Only those gesture names exist. Use nod to agree or acknowledge."
)

# JSON schema for llama.cpp's constrained decoding: the model physically cannot
# emit an invalid gesture / look_at, and it stops as soon as the object closes.
REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "speech":  {"type": "string"},
        "gesture": {"type": "string", "enum": ["idle", "head_tilt", "recoil", "slow_sweep",
                                               "lean_in", "dismissive_turn", "nod"]},
        "look_at": {"type": "string", "enum": ["speaker", "away", "person", "nothing", "hold"]},
        "mood": {
            "type": "object",
            "properties": {"pleasure": {"type": "number"}, "arousal": {"type": "number"},
                           "dominance": {"type": "number"}},
            "required": ["pleasure", "arousal", "dominance"],
        },
    },
    "required": ["speech", "gesture", "look_at", "mood"],
}


# ── Turn input builder ────────────────────────────────────────────────────────

def build_turn(heard: str = "", vision: Optional[dict] = None,
               event: str = "speech", notes: Optional[list] = None) -> str:
    """
    Combine everything GLaDOS perceives this turn into ONE JSON user message.

    heard  : transcribed speech ("" when the turn is camera-triggered)
    vision : compact perception snapshot (VisionClient.snapshot()), or None if offline
    event  : speech | person_entered | gesture | visual_check
    notes  : operator / system notes (interruption context, admin injects)
    """
    turn = {"heard": heard or "", "vision": vision, "event": event}
    if notes:
        turn["notes"] = notes
    return json.dumps(turn, ensure_ascii=False)


# ── Few-shot examples ─────────────────────────────────────────────────────────
# None, deliberately. Tested on Bonsai-8B: with example replies in the prompt the
# model repeats them verbatim ("I am running at optimal capacity…") whatever the
# input, then keeps repeating itself from history. The reply FORMAT no longer
# needs examples — REPLY_SCHEMA enforces it during decoding — so the prompt is
# just the instructions, which is also ~40% fewer tokens to evaluate.
FEW_SHOT: list[dict] = []

# ── Incremental JSON parsing ───────────────────────────────────────────────────
# "speech" is the FIRST field in the schema, so while the model is still writing
# gesture/look_at/mood we can already read the words it wants said. That is what
# makes speaking before generation finishes possible.

# A sentence ends at .!? plus any closing quote/bracket, then whitespace or EOS.
# "..." is a pause, not an ending: the lookbehind skips a dot that follows a dot.
_SENTENCE_END = re.compile(r'(?<!\.)[.!?]["\')\]]*\s')
MAX_SENTENCES = 2   # hard cap: the prompt asks for 1–2, this enforces it


def extract_speech(raw: str) -> tuple[Optional[str], bool]:
    """
    Pull the "speech" value out of a possibly-incomplete JSON stream.

    Returns (text, complete). text is None until the opening quote arrives.
    complete is True once the closing quote has been seen.
    """
    i = raw.find('"speech"')
    if i == -1:
        return None, False
    j = raw.find(":", i + len('"speech"'))
    if j == -1:
        return None, False
    k = raw.find('"', j + 1)
    if k == -1:
        return None, False

    # Walk for the closing quote, respecting backslash escapes.
    esc = False
    for idx in range(k + 1, len(raw)):
        c = raw[idx]
        if esc:
            esc = False
        elif c == "\\":
            esc = True
        elif c == '"':
            try:
                return json.loads(raw[k:idx + 1]), True
            except json.JSONDecodeError:
                return None, False

    # Still open. Close it ourselves and let json handle the escapes. Trailing
    # partial escapes (a lone backslash, a half-written \uXXXX) won't parse, so
    # shave characters until it does — the next token will complete them.
    frag = raw[k:]
    while len(frag) > 1:
        try:
            return json.loads(frag + '"'), False
        except json.JSONDecodeError:
            frag = frag[:-1]
    return "", False


def _sse_delta(line: str) -> str:
    """Content fragment from one OpenAI-style SSE line ("" if it carries none)."""
    if not line:
        return ""
    if line.startswith("data:"):
        line = line[len("data:"):].strip()
    if not line or line == "[DONE]":
        return ""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return ""
    if "choices" not in obj:
        return obj.get("content") or ""          # /completion stream
    choices = obj.get("choices") or [{}]
    c = choices[0]
    # llama.cpp sends "delta" while streaming; some servers send "message".
    return (c.get("delta") or c.get("message") or {}).get("content") or ""


# ── Raw ChatML rendering ──────────────────────────────────────────────────────
# Bonsai's chat template (Qwen-style ChatML) opens every reply with an empty
# <think></think> block but drops that block when the same reply is replayed as
# history. The cached tokens therefore never match the next request and the
# server re-evaluates every previous reply (~55 tokens) on every turn.
# Rendering the prompt ourselves — identically every time — and sending it to
# /completion makes the next request an exact extension of the last one, so only
# the new user message is evaluated.
_THINK = "<think>\n\n</think>\n\n"


def render_chatml(messages: list[dict]) -> str:
    out = []
    for m in messages:
        body = (_THINK + m["content"]) if m["role"] == "assistant" else m["content"]
        out.append(f"<|im_start|>{m['role']}\n{body}<|im_end|>\n")
    out.append("<|im_start|>assistant\n" + _THINK)
    return "".join(out)


def split_sentence(pending: str) -> tuple[str, str]:
    """Split off the first complete sentence. Returns (sentence, remainder)."""
    m = _SENTENCE_END.search(pending)
    if not m:
        return "", pending
    return pending[:m.end()], pending[m.end():]


# A clause boundary: comma / semicolon / colon / dash followed by whitespace.
_CLAUSE_END = re.compile(r'[,;:\u2014\u2013]\s')
FIRST_CHUNK_MIN_WORDS = 4


def split_first_chunk(pending: str) -> tuple[str, str]:
    """
    Earliest speakable piece of the opening sentence.

    The first sound is what the listener waits for, so the opening chunk may end
    at a clause boundary (", ") once it has FIRST_CHUNK_MIN_WORDS words, instead
    of waiting for the whole sentence. TTS starts on "Oh, it is you again," while
    the model is still writing the rest.
    """
    sent = _SENTENCE_END.search(pending)
    for m in _CLAUSE_END.finditer(pending):
        if sent and m.start() > sent.start():
            break
        if len(pending[:m.start()].split()) >= FIRST_CHUNK_MIN_WORDS:
            return pending[:m.end()], pending[m.end():]
    if sent:
        return pending[:sent.end()], pending[sent.end():]
    return "", pending


def _terminated(chunk: str) -> str:
    """Final chunk of a reply: make sure it ends with punctuation. (The DRY
    sampler sometimes suppresses the closing full stop, and TTS intonation is
    noticeably flatter without it.)"""
    chunk = chunk.strip()
    return chunk if re.search(r'[.!?…]["\')\]]*$', chunk) else chunk + "."


def _ends_sentence(chunk: str) -> bool:
    return bool(re.search(r'(?<!\.)[.!?]["\')\]]*\s*$', chunk))


class GladosBrain:
    """
    Stateful GLaDOS brain. Maintains conversation history across turns.

    Latency design
    --------------
    * warmup() evaluates the fixed prefix (system prompt + examples) once at
      startup; llama.cpp's prompt cache then reuses it, so each turn only
      evaluates the new user message.
    * History is append-only and stored byte-for-byte as sent/generated, so the
      cache keeps matching. It is trimmed in blocks (rarely) rather than one
      turn at a time (which would invalidate the cache on every turn).
    * A turn only enters history once it completes. A cancelled reply
      (interrupt, speculation that turned out wrong) leaves no trace.

    Parameters
    ----------
    url : str            llama.cpp server endpoint.
    max_history : int    Turns kept before a block trim (down to half).
    temperature : float  Sampling temperature.
    timeout : int        HTTP timeout in seconds.
    """

    def __init__(
        self,
        url: str = LLAMACPP_URL,
        max_history: int = 8,
        temperature: float = 0.7,
        timeout: int = 30,
    ):
        self._url         = url
        self._max_history = max_history
        self._temperature = temperature
        self._timeout     = timeout
        self._history: list[dict] = []
        self._use_schema  = True      # turned off automatically if the server rejects it
        self._base        = url.split("/v1/")[0]
        # None = not checked yet. True = server uses ChatML, so we render the
        # prompt ourselves and call /completion (see render_chatml).
        self._raw_mode: Optional[bool] = None
        # Parsed dict from the most recent turn (gesture / look_at / mood).
        self.last_response: Optional[dict] = None
        # Timing of the most recent turn: first_chunk / total seconds, plus the
        # server's own numbers (prompt_n, cache_n, tokens per second) if reported.
        self.last_timing: dict = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def _prefix(self) -> list[dict]:
        return [{"role": "system", "content": GLADOS_SYSTEM}] + FEW_SHOT

    def _detect_raw_mode(self) -> bool:
        if self._raw_mode is None:
            try:
                tpl = requests.get(self._base + "/props", timeout=5).json().get("chat_template", "")
                self._raw_mode = "<|im_start|>" in tpl
            except Exception:
                self._raw_mode = False
        return self._raw_mode

    def warmup(self) -> float:
        """
        Prime llama.cpp's prompt cache with the fixed prefix. Call once after the
        server is healthy. Returns seconds taken (0.0 on failure — non-fatal).
        """
        import time
        t0 = time.time()
        try:
            url, body = self._request(self._prefix() + [{"role": "user", "content": build_turn(heard="")}])
            body.update(stream=False)
            body["n_predict" if "prompt" in body else "max_tokens"] = 1
            body.pop("json_schema", None); body.pop("response_format", None)
            requests.post(url, json=body, timeout=max(self._timeout, 120)).raise_for_status()
            return time.time() - t0
        except Exception as e:
            print(f"[brain] warmup failed (non-fatal): {e}")
            return 0.0

    def _request(self, messages: list[dict]) -> tuple[str, dict]:
        """(url, json body) for one generation."""
        body = {
            "temperature":  self._temperature,
            "cache_prompt": True,         # reuse the KV cache for the shared prefix
            "stream":       True,
            # DRY sampler: penalise continuing any sequence that already appears in
            # the context — her earlier replies, the user's own words, the
            # instructions. Stops verbatim repeats and parroting. The JSON
            # structure is unaffected because the schema grammar forces it.
            "dry_multiplier":     0.8,
            "dry_base":           1.75,
            "dry_allowed_length": 2,
            "dry_penalty_last_n": 4096,   # look back over the whole context
        }
        if self._detect_raw_mode():
            body.update(prompt=render_chatml(messages), n_predict=120, stop=["<|im_end|>"])
            if self._use_schema:
                body["json_schema"] = REPLY_SCHEMA
            return self._base + "/completion", body
        body.update(messages=messages, max_tokens=120)   # 2 short sentences + gesture/look_at/mood
        if self._use_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "glados_reply", "schema": REPLY_SCHEMA},
            }
        return self._url, body

    def respond_stream(self, user_text: str, perception: str = "") -> Iterator[str]:
        """
        Stream the reply, yielding speakable chunks as soon as they exist — the
        opening clause first, then whole sentences — long before
        gesture/look_at/mood arrive.

            for chunk in brain.respond_stream(build_turn(heard=text, vision=...)):
                dispatch_speech(chunk)
            full = brain.last_response      # gesture / look_at / mood

        Closing the generator early (break) cancels the request and records
        nothing. After normal exhaustion `last_response` holds the parsed dict
        (None if the server failed or the JSON was malformed).
        """
        import time
        self.last_response = None
        self.last_timing   = {}

        system = GLADOS_SYSTEM
        if perception:   # legacy callers that pass a plain-text perception string
            system += f"\n\n[You currently see: {perception}.]"
        user_msg = {"role": "user", "content": user_text}
        messages = ([{"role": "system", "content": system}] + FEW_SHOT
                    + self._history + [user_msg])

        raw       = ""
        consumed  = 0     # chars of speech already yielded
        sentences = 0     # complete sentences yielded
        chunks    = 0     # chunks yielded (a clause chunk is not a full sentence)
        t0        = time.time()
        t_first   = None
        server_timings = None

        def _note_first():
            nonlocal t_first
            if t_first is None:
                t_first = time.time() - t0

        try:
            url, body = self._request(messages)
            resp = requests.post(url, json=body, stream=True, timeout=self._timeout)
            if resp.status_code >= 400 and self._use_schema:
                # Older servers reject response_format/json_schema — fall back once.
                print(f"[brain] server rejected the request ({resp.status_code}: "
                      f"{resp.text[:200]}) — retrying without constrained decoding")
                resp.close()
                self._use_schema = False
                url, body = self._request(messages)
                resp = requests.post(url, json=body, stream=True, timeout=self._timeout)
            if resp.status_code >= 400:
                print(f"[brain] server error {resp.status_code}: {resp.text[:300]}")
            with resp as r:
                r.raise_for_status()
                r.encoding = "utf-8"
                for line in r.iter_lines(decode_unicode=True):
                    if line and '"timings"' in line:
                        try:
                            server_timings = json.loads(line[5:].strip()).get("timings")
                        except Exception:
                            pass
                    delta = _sse_delta(line)
                    if not delta:
                        continue
                    raw += delta

                    speech, complete = extract_speech(raw)
                    if speech is None or sentences >= MAX_SENTENCES:
                        continue

                    pending = speech[consumed:]
                    while sentences < MAX_SENTENCES:
                        if chunks == 0:
                            piece, pending = split_first_chunk(pending)
                        else:
                            piece, pending = split_sentence(pending)
                        if not piece:
                            break
                        consumed += len(piece)
                        if piece.strip():
                            chunks += 1
                            if _ends_sentence(piece):
                                sentences += 1
                            _note_first()
                            yield piece.strip()
                    # Speech string closed: what's left is the final sentence
                    if complete and sentences < MAX_SENTENCES and pending.strip():
                        consumed += len(pending)
                        chunks += 1
                        sentences += 1
                        _note_first()
                        yield _terminated(pending)

        except requests.exceptions.ConnectionError:
            print("[brain] llama.cpp server not running. Start it with:")
            print("  llama-server -m central_ai/benchmarking_tools/Bonsai-8B-Q1_0.gguf --port 8080")
            return
        except Exception as e:
            print(f"[brain] error: {e}")
            return

        if not raw:
            return

        # Flush a trailing fragment the model never punctuated.
        speech, _ = extract_speech(raw)
        if speech and consumed < len(speech) and sentences < MAX_SENTENCES:
            tail = speech[consumed:].strip()
            if tail:
                _note_first()
                yield _terminated(tail)

        # Commit the turn. Stored exactly as sent/generated so the server's
        # prompt cache still matches on the next request.
        self._history.append(user_msg)
        self._history.append({"role": "assistant", "content": raw})
        self._trim_history()
        self.last_response = self._parse(raw)
        self.last_timing = {"first_chunk": t_first, "total": time.time() - t0}
        if server_timings:
            self.last_timing.update({
                "prompt_n": server_timings.get("prompt_n"),
                "cache_n":  server_timings.get("cache_n"),
                "tok_s":    server_timings.get("predicted_per_second"),
                "predicted_n": server_timings.get("predicted_n"),
            })
        if self.last_response is None:
            print(f"[brain] JSON parse failed. raw: {raw[:200]}")

    def respond(self, user_text: str) -> Optional[dict]:
        """
        Blocking variant — waits for the whole reply and returns the parsed dict.

        Returns None if the server is unreachable or the response can't be parsed.
        """
        for _ in self.respond_stream(user_text):
            pass
        return self.last_response

    def discard(self, user_text: str) -> bool:
        """Forget a completed turn (identified by its user message) — e.g. a
        reply that was generated but cancelled before anyone heard it."""
        h = self._history
        for i in range(len(h) - 2, -1, -1):
            if h[i]["role"] == "user" and h[i]["content"] == user_text:
                del h[i:i + 2]
                return True
        return False

    def reset(self):
        """Clear conversation history (start a new conversation)."""
        self._history.clear()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _trim_history(self):
        # Block trim: dropping the oldest turn on every request would change the
        # prompt right after the cached prefix and force the server to
        # re-evaluate the whole history each turn. Instead let it grow to
        # max_history turns, then cut to half in one go.
        max_msgs = self._max_history * 2   # user + assistant pairs
        if len(self._history) > max_msgs:
            self._history = self._history[-(self._max_history // 2) * 2:]

    def _parse(self, raw: str) -> Optional[dict]:
        text = raw.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text  = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
        # Extract JSON object
        start = text.find("{")
        end   = text.rfind("}")
        if start == -1 or end == -1:
            return None
        try:
            return self._sanitize(json.loads(text[start:end + 1]))
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _sanitize(d: dict) -> dict:
        """
        Enforce the output contract from the research doc before anything
        downstream sees it:
          gesture → must be in GESTURE_ENUM (a hallucinated name crashes the robot) → else "idle"
          look_at → must be in LOOK_AT_ENUM → else "speaker"
          mood    → PAD floats clamped to [-1, 1]
          speech  → always a string
        """
        if not isinstance(d, dict):
            return {"speech": "", "gesture": "idle", "look_at": "speaker",
                    "mood": {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0}}
        g = d.get("gesture")
        if g not in GESTURE_ENUM:
            if g:
                print(f"[brain] invalid gesture {g!r} → idle")
            d["gesture"] = "idle"
        if d.get("look_at") not in LOOK_AT_ENUM:
            d["look_at"] = "speaker"
        mood = d.get("mood") if isinstance(d.get("mood"), dict) else {}
        clamped = {}
        for k in ("pleasure", "arousal", "dominance"):
            try:
                clamped[k] = max(-1.0, min(1.0, float(mood.get(k, 0.0))))
            except (TypeError, ValueError):
                clamped[k] = 0.0
        d["mood"] = clamped
        if not isinstance(d.get("speech"), str):
            d["speech"] = ""
        return d


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    brain = GladosBrain()
    print("GLaDOS Brain — type to chat, Ctrl-C to quit\n")
    while True:
        try:
            user = input("You: ").strip()
            if not user:
                continue
            response = brain.respond(user)
            if response:
                print(f"GLaDOS: {response['speech']}")
                print(f"        gesture={response.get('gesture')}  "
                      f"look_at={response.get('look_at')}  "
                      f"mood={response.get('mood')}\n")
            else:
                print("(no response)\n")
        except KeyboardInterrupt:
            break
