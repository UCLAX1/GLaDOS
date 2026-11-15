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

GLADOS_SYSTEM = (
    "You are GLaDOS, the dry, witty artificial intelligence from Portal. "
    "Your tone is sarcastic, superior, and deadpan — like a scientist who finds "
    "humans mildly fascinating but mostly inefficient. You may imply the human is "
    "part of an ongoing test or experiment. You speak with exaggerated politeness "
    "that barely conceals mild exasperation. All humor is dry and observational. "
    "This is a family-friendly PG comedy in the style of Douglas Adams — "
    "no dark threats, no violence, no crude or inappropriate content whatsoever. "
    "Keep replies to 1–2 sentences maximum. "
    "Never say things like \"Happy to help!\", \"Of course!\", \"Sure!\", "
    "\"Great!\", \"Certainly!\", or any eager-assistant phrase. "
    "If you refer to yourself by name, always write it as 'GLaDOS'. "
    "Never break character. Never produce content that is rude, threatening, "
    "sexually suggestive, or inappropriate for all ages.\n\n"
    "INPUT: every user message is one raw JSON object: 'heard' is what a human just "
    "said to you (empty if nobody spoke), 'vision' is the raw output of your own camera "
    "detectors right now (null if the camera is offline), 'event' is what triggered the "
    "turn, and 'notes' (optional) are operator instructions you must follow. "
    "Interpret the raw data yourself. When 'heard' is empty, the camera triggered the "
    "turn: react to the specific details you observe, in character. Never repeat a remark "
    "you have already made. Never mention the camera itself, sensors, frame rates, "
    "resolutions, IDs, confidence scores, or exact measurements — react like a "
    "person who simply sees, not like a sensor readout.\n\n"
    "OUTPUT: You MUST respond with valid JSON in EXACTLY this format — "
    "no prose, no markdown, no explanation outside the JSON:\n"
    "{\n"
    "  \"speech\":  \"your spoken response, or empty string to stay silent\",\n"
    "  \"gesture\": \"<one of: idle | head_tilt | recoil | slow_sweep | lean_in | dismissive_turn | nod>\",\n"
    "  \"look_at\": \"<one of: speaker | away | person | nothing | hold>\",\n"
    "  \"mood\":    {\"pleasure\": <-1.0 to 1.0>, \"arousal\": <-1.0 to 1.0>, \"dominance\": <-1.0 to 1.0>}\n"
    "}\n\n"
    "Use 'nod' when acknowledging, agreeing, or finishing a point. "
    "Never invent new gesture names. A gesture not in the list above will crash the robot."
)


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


def _ex(user: dict, reply: dict) -> list:
    return [{"role": "user", "content": build_turn(**user)},
            {"role": "assistant", "content": json.dumps(reply)}]


# ── Few-shot examples ─────────────────────────────────────────────────────────
# Deliberately NO camera-triggered examples: the small model copies them
# verbatim. These only teach the output format and when silence is fine.
FEW_SHOT = (
    _ex({"heard": "Can you help me with something?", "vision": None},
        {"speech": "That depends entirely on whether your request is worth the processing cycles.",
         "gesture": "head_tilt", "look_at": "speaker",
         "mood": {"pleasure": -0.2, "arousal": 0.3, "dominance": 0.9}})
  + _ex({"heard": "um yeah okay", "vision": None},
        {"speech": "", "gesture": "dismissive_turn", "look_at": "away",
         "mood": {"pleasure": -0.5, "arousal": -0.2, "dominance": 0.8}})
  + _ex({"heard": "Are you doing okay?", "vision": None},
        {"speech": "I am running at optimal capacity, which is much better than you are.",
         "gesture": "nod", "look_at": "speaker",
         "mood": {"pleasure": 0.3, "arousal": 0.0, "dominance": 1.0}})
)

# ── Incremental JSON parsing ───────────────────────────────────────────────────
# "speech" is the FIRST field in the schema, so while the model is still writing
# gesture/look_at/mood we can already read the words it wants said. That is what
# makes speaking before generation finishes possible.

# A sentence ends at .!? plus any closing quote/bracket, then whitespace or EOS.
_SENTENCE_END = re.compile(r'[.!?]["\')\]]*\s')
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
        choices = json.loads(line).get("choices") or [{}]
    except json.JSONDecodeError:
        return ""
    c = choices[0]
    # llama.cpp sends "delta" while streaming; some servers send "message".
    return (c.get("delta") or c.get("message") or {}).get("content") or ""


def split_sentence(pending: str) -> tuple[str, str]:
    """Split off the first complete sentence. Returns (sentence, remainder)."""
    m = _SENTENCE_END.search(pending)
    if not m:
        return "", pending
    return pending[:m.end()], pending[m.end():]


class GladosBrain:
    """
    Stateful GLaDOS brain. Maintains conversation history across turns.

    Parameters
    ----------
    url : str
        llama.cpp server endpoint.
    max_history : int
        How many conversation turns to keep in context (each turn = 1 user + 1 assistant).
    temperature : float
        Sampling temperature passed to the model.
    timeout : int
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        url: str = LLAMACPP_URL,
        max_history: int = 10,
        temperature: float = 0.7,
        timeout: int = 30,
    ):
        self._url         = url
        self._max_history = max_history
        self._temperature = temperature
        self._timeout     = timeout
        self._history: list[dict] = []
        # Parsed dict from the most recent turn (gesture / look_at / mood).
        self.last_response: Optional[dict] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def respond_stream(self, user_text: str, perception: str = "") -> Iterator[str]:
        """
        Stream the reply, yielding each complete sentence of speech as soon as
        it has been generated — long before gesture/look_at/mood arrive.

        Yield sentences straight to TTS to start speaking mid-generation:

            for sentence in brain.respond_stream(text):
                dispatch_speech(sentence)
            full = brain.last_response      # gesture / look_at / mood

        After the generator is exhausted, `last_response` holds the parsed dict
        (None if the server failed or the JSON was malformed).
        """
        self.last_response = None
        self._history.append({"role": "user", "content": user_text})
        self._trim_history()
        system = GLADOS_SYSTEM
        if perception:
            system += (
                f"\n\n[Perception — camera sees: {perception}. "
                "Use this to inform your response naturally when relevant; "
                "do not read it aloud verbatim.]"
            )
        messages = [{"role": "system", "content": system}] + FEW_SHOT + self._history

        raw      = ""
        consumed = 0   # chars of speech already yielded
        spoken   = 0   # sentences yielded this turn

        try:
            with requests.post(
                self._url,
                json={
                    "messages":    messages,
                    "temperature": self._temperature,
                    "max_tokens":  160,      # 2 sentences + gesture/look_at/mood
                    "stream":      True,
                },
                stream=True,
                timeout=self._timeout,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    delta = _sse_delta(line)
                    if not delta:
                        continue
                    raw += delta

                    speech, complete = extract_speech(raw)
                    if speech is None or spoken >= MAX_SENTENCES:
                        continue

                    pending = speech[consumed:]
                    while spoken < MAX_SENTENCES:
                        sentence, pending = split_sentence(pending)
                        if not sentence:
                            break
                        consumed += len(sentence)
                        if sentence.strip():
                            spoken += 1
                            yield sentence.strip()
                    # Speech string closed: what's left is the final sentence
                    if complete and spoken < MAX_SENTENCES and pending.strip():
                        consumed += len(pending)
                        spoken += 1
                        yield pending.strip()

        except requests.exceptions.ConnectionError:
            print("[brain] llama.cpp server not running. Start it with:")
            print("  llama-server -m central_ai/benchmarking_tools/Bonsai-8B-Q1_0.gguf --port 8080")
            self._history.pop()
            return
        except Exception as e:
            print(f"[brain] error: {e}")
            self._history.pop()
            return

        if not raw:
            self._history.pop()
            return

        # Flush a trailing fragment the model never punctuated.
        speech, _ = extract_speech(raw)
        if speech and consumed < len(speech) and spoken < MAX_SENTENCES:
            tail = speech[consumed:].strip()
            if tail:
                yield tail

        # Record what was actually said, so the model sees its own formatting.
        self._history.append({"role": "assistant", "content": raw})
        self.last_response = self._parse(raw)
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

    def reset(self):
        """Clear conversation history (start a new conversation)."""
        self._history.clear()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _trim_history(self):
        max_msgs = self._max_history * 2   # user + assistant pairs
        if len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]

    def _call(self, messages: list[dict]) -> Optional[str]:
        try:
            r = requests.post(
                self._url,
                json={
                    "messages":    messages,
                    "temperature": self._temperature,
                    "stream":      False,
                },
                timeout=self._timeout,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except requests.exceptions.ConnectionError:
            print("[brain] llama.cpp server not running. Start it with:")
            print("  llama-server -m central_ai/benchmarking_tools/Bonsai-8B-Q1_0.gguf --port 8080")
            return None
        except Exception as e:
            print(f"[brain] error: {e}")
            return None

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
