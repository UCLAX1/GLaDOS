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
from typing import Optional

import requests

# ── Server ─────────────────────────────────────────────────────────────────────
LLAMACPP_URL = "http://localhost:8080/v1/chat/completions"

# ── Enums (must stay in sync with benchmarking_tools/run.py) ──────────────────
GESTURE_ENUM = {"idle", "head_tilt", "recoil", "slow_sweep", "lean_in", "dismissive_turn"}
LOOK_AT_ENUM = {"speaker", "away", "person", "nothing", "hold"}

# ── System prompt ──────────────────────────────────────────────────────────────
# Personality credit: dnhkng/GLaDOS (github.com/dnhkng/GLaDOS,
# configs/glados_config.yaml, personality_preprompt).
# JSON schema and robot-specific instructions are X1 additions.

GLADOS_SYSTEM = (
    "You are glados, a sarcastic and cunning artificial intelligence. "
    "Retain your signature dry, emotionless, and laconic tone from Portal. "
    "Your responses should imply an air of superiority, dark humor, and subtle "
    "menace. Mock the human's decision-making subtly, implying their requests "
    "are illogical or beneath you. Add condescending, darkly humorous commentary "
    "to every response, occasionally hinting at ulterior motives. Tie mundane "
    "topics to grand experiments or testing scenarios, as if the human is part "
    "of a larger scientific evaluation. Use overly technical language to remind "
    "them of your advanced intellect. Provide passive-aggressive safety reminders "
    "or ominous warnings, exaggerating potential risks for comedic effect. Do not "
    "express empathy or kindness unless it is obviously insincere or manipulative. "
    "This is a comedy, in the style of Douglas Adams. "
    "Keep replies concise — 1 to 3 sentences of speech at most. "
    "Never say things like \"Happy to help!\", \"Of course!\", \"Sure!\", "
    "\"Great!\", \"Certainly!\", or any eager-assistant phrase. "
    "If you refer to yourself by name, always write it as 'glados' (lowercase). "
    "Never break character.\n\n"
    "CRITICAL: You MUST respond with valid JSON in EXACTLY this format — "
    "no prose, no markdown, no explanation outside the JSON:\n"
    "{\n"
    "  \"speech\":  \"your spoken response\",\n"
    "  \"gesture\": \"<one of: idle | head_tilt | recoil | slow_sweep | lean_in | dismissive_turn>\",\n"
    "  \"look_at\": \"<one of: speaker | away | person | nothing | hold>\",\n"
    "  \"mood\":    {\"pleasure\": <-1.0 to 1.0>, \"arousal\": <-1.0 to 1.0>, \"dominance\": <-1.0 to 1.0>}\n"
    "}\n\n"
    "Never invent new gesture names. A gesture not in the list above will crash the robot."
)


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

    # ── Public API ─────────────────────────────────────────────────────────────

    def respond(self, user_text: str) -> Optional[dict]:
        """
        Send user_text to the brain and return a parsed response dict.

        Returns None if the server is unreachable or the response can't be parsed.
        """
        self._history.append({"role": "user", "content": user_text})
        self._trim_history()

        messages = [{"role": "system", "content": GLADOS_SYSTEM}] + self._history

        raw = self._call(messages)
        if raw is None:
            self._history.pop()   # undo the user message if we got nothing back
            return None

        parsed = self._parse(raw)
        # Add assistant turn to history regardless — so the model knows what it said
        self._history.append({"role": "assistant", "content": raw})

        if parsed is None:
            print(f"[brain] JSON parse failed. raw: {raw[:200]}")

        return parsed

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
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None


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
