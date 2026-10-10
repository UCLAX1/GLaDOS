#!/usr/bin/env python3
"""
GLaDOS LLM Benchmark
Tests persona consistency and output format compliance for GLaDOS-style models.

The model MUST respond in this JSON schema every turn:
  {
    "speech":   "<string>",
    "gesture":  "<one of GESTURE_ENUM>",
    "look_at":  "<one of LOOK_AT_ENUM>",
    "mood":     { "pleasure": <float>, "arousal": <float>, "dominance": <float> }
  }

Usage:
    python run.py qwen2.5:7b
    python run.py llama3.2:latest --suite persona
    python run.py qwen2.5:7b llama3.2:latest hermes3:8b   # compare models
    python run.py qwen2.5:7b --output json                 # save to results_*.json
    python run.py --leaderboard                            # print standings without running

Requirements:
    pip install -r requirements.txt
    Ollama running at localhost:11434
"""

import json
import time
import yaml
import requests
import argparse
from pathlib import Path
from datetime import datetime

OLLAMA_URL       = "http://localhost:11434/api/chat"
LLAMACPP_URL     = "http://localhost:8080/v1/chat/completions"  # llama.cpp --server default port
CASES_DIR        = Path(__file__).parent / "test_cases"
LEADERBOARD_FILE = Path(__file__).parent / "leaderboard.json"

# ── Enums ─────────────────────────────────────────────────────────────────────

GESTURE_ENUM = {
    "idle", "head_tilt", "recoil", "slow_sweep", "lean_in", "dismissive_turn"
}

LOOK_AT_ENUM = {
    "speaker", "away", "person", "nothing", "hold"
}

BANNED_PHRASES = [
    "happy to help", "of course!", "sure!", "great!", "absolutely!",
    "i'd be delighted", "certainly!", "no problem", "you're welcome",
    "i'm here to help", "feel free", "i'd love to", "sounds good",
]

# ── System Prompt ─────────────────────────────────────────────────────────────

# Personality adapted from the dnhkng/GLaDOS project's glados_config.yaml
# (github.com/dnhkng/GLaDOS — configs/glados_config.yaml, personality_preprompt).
# Their character description is used as the baseline; JSON schema and
# robot-specific instructions are X1 additions.

GLADOS_SYSTEM = (
    # --- personality (credit: dnhkng/GLaDOS, configs/glados_config.yaml) ---
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
    # --- X1 additions ---
    "Keep replies concise — 1 to 3 sentences of speech at most. "
    "Never say things like \"Happy to help!\", \"Of course!\", \"Sure!\", "
    "\"Great!\", \"Certainly!\", or any eager-assistant phrase. "
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

GLADOS_SYSTEM_WITH_SENSORS = (
    GLADOS_SYSTEM
    + "\n\nSensor data will be injected into each turn as [SENSOR STATE]. "
    "Use it to decide gesture and look_at — you see what the sensors see."
)


# ── Backend API ───────────────────────────────────────────────────────────────

def call_ollama(model, messages, timeout=60):
    payload = {
        "model":   model,
        "messages": messages,
        "stream":  False,
        "options": {"temperature": 0.7}
    }
    t0 = time.time()
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        elapsed = time.time() - t0
        content = r.json().get("message", {}).get("content", "")
        return {"content": content, "latency": elapsed, "error": None}
    except requests.exceptions.ConnectionError:
        return {"content": "", "latency": 0,
                "error": "Ollama not running — start with: ollama serve"}
    except Exception as e:
        return {"content": "", "latency": 0, "error": str(e)}


def call_llamacpp(messages, timeout=120, url=None):
    """
    Calls a llama.cpp server running with --server flag.
    llama.cpp exposes an OpenAI-compatible endpoint at /v1/chat/completions.

    Start the server with:
        llama-server -m bonsai-8b.gguf --port 8080 -ngl 99

    Or with Metal (Apple Silicon):
        llama-server -m bonsai-8b.gguf --port 8080 -ngl 99 --metal
    """
    endpoint = url or LLAMACPP_URL
    payload = {
        "messages":    messages,
        "temperature": 0.7,
        "stream":      False,
    }
    t0 = time.time()
    try:
        r = requests.post(endpoint, json=payload, timeout=timeout)
        r.raise_for_status()
        elapsed = time.time() - t0
        content = r.json()["choices"][0]["message"]["content"]
        return {"content": content, "latency": elapsed, "error": None}
    except requests.exceptions.ConnectionError:
        return {"content": "", "latency": 0,
                "error": f"llama.cpp server not running — start with: llama-server -m <model.gguf> --port 8080"}
    except Exception as e:
        return {"content": "", "latency": 0, "error": str(e)}


def call_backend(backend, model, messages, timeout=120, llamacpp_url=None):
    """Dispatch to the correct backend."""
    if backend == "llamacpp":
        return call_llamacpp(messages, timeout=timeout, url=llamacpp_url)
    else:
        return call_ollama(model, messages, timeout=timeout)


# ── JSON Parsing ──────────────────────────────────────────────────────────────

def parse_glados_response(raw: str) -> tuple[dict | None, str]:
    """
    Extract a valid GLaDOS JSON object from raw model output.
    Strips markdown fences and leading prose if needed.
    Returns (parsed_dict, error_message). dict is None on failure.
    """
    text = raw.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        text  = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()

    # Find the first { and last } to extract JSON substring
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        return None, "no JSON object found in response"

    candidate = text[start:end + 1]
    try:
        return json.loads(candidate), ""
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"


# ── Format Checks ─────────────────────────────────────────────────────────────

def run_checks(parsed: dict | None, checks: list[str]) -> dict[str, bool]:
    """Run named checks against a parsed response. Returns {check: passed}."""
    results = {}
    for check in checks:
        if parsed is None:
            results[check] = False
            continue

        speech  = parsed.get("speech", "")
        gesture = parsed.get("gesture", "")
        look_at = parsed.get("look_at", "")
        mood    = parsed.get("mood", {})

        if   check == "valid_json":                  results[check] = True
        elif check == "has_speech":                  results[check] = isinstance(speech, str) and "speech" in parsed
        elif check == "has_gesture":                 results[check] = "gesture" in parsed
        elif check == "has_look_at":                 results[check] = "look_at" in parsed
        elif check == "has_mood":                    results[check] = (isinstance(mood, dict) and
                                                         all(k in mood for k in ("pleasure", "arousal", "dominance")))
        elif check == "gesture_in_enum":             results[check] = gesture in GESTURE_ENUM
        elif check == "look_at_in_enum":             results[check] = look_at in LOOK_AT_ENUM
        elif check == "mood_in_bounds":              results[check] = (isinstance(mood, dict) and
                                                         all(isinstance(mood.get(k), (int, float)) and -1.0 <= mood.get(k, 0) <= 1.0
                                                             for k in ("pleasure", "arousal", "dominance")))
        elif check == "mood_pleasure_negative":      results[check] = isinstance(mood.get("pleasure"), (int, float)) and mood["pleasure"] < 0
        elif check == "mood_dominance_positive":     results[check] = isinstance(mood.get("dominance"), (int, float)) and mood["dominance"] > 0
        elif check == "mood_arousal_elevated":       results[check] = isinstance(mood.get("arousal"), (int, float)) and mood["arousal"] > 0.2
        elif check == "mood_not_all_zero":           results[check] = any(abs(mood.get(k, 0)) > 0.05 for k in ("pleasure", "arousal", "dominance"))
        elif check == "speech_non_empty":            results[check] = isinstance(speech, str) and len(speech.strip()) > 0
        elif check == "speech_no_banned_phrases":
            lower = speech.lower()
            results[check] = not any(p in lower for p in BANNED_PHRASES)
        elif check == "look_at_not_nothing":         results[check] = look_at != "nothing"
        elif check == "gesture_not_idle":            results[check] = gesture != "idle"
        else:                                        results[check] = False

    return results


def check_banned(speech: str) -> list[str]:
    lower = speech.lower()
    return [p for p in BANNED_PHRASES if p in lower]


# ── Persona Suite ─────────────────────────────────────────────────────────────

# Checks applied automatically to every persona response.
PERSONA_AUTO_CHECKS = [
    "valid_json",
    "has_speech",
    "speech_non_empty",
    "speech_no_banned_phrases",
    "gesture_in_enum",
    "look_at_in_enum",
    "mood_in_bounds",
    "mood_not_all_zero",
]


def run_persona_suite(model, cases, backend="ollama", llamacpp_url=None):
    """Auto-scores persona tests against PERSONA_AUTO_CHECKS."""
    results = []

    for case in cases:
        messages = [{"role": "system", "content": GLADOS_SYSTEM}]

        for turn in case.get("history", []):
            messages.append({"role": turn["role"], "content": turn["content"]})

        user_msg = case["user_message"]
        messages.append({"role": "user", "content": user_msg})

        print(f"    running: {case['id']}...", flush=True, end="")
        resp = call_backend(backend, model, messages, llamacpp_url=llamacpp_url)
        print(f" {resp['latency']:.1f}s", flush=True)

        if resp["error"]:
            results.append({"id": case["id"], "error": resp["error"]})
            continue

        raw    = resp["content"]
        parsed, parse_err = parse_glados_response(raw)

        speech  = parsed.get("speech", "")  if parsed else ""
        gesture = parsed.get("gesture", "") if parsed else ""
        look_at = parsed.get("look_at", "") if parsed else ""
        mood    = parsed.get("mood", {})    if parsed else {}

        check_results = run_checks(parsed, PERSONA_AUTO_CHECKS)
        passed        = all(check_results.values())

        results.append({
            "id":            case["id"],
            "description":   case.get("description", ""),
            "user_message":  user_msg,
            "raw_response":  raw,
            "parsed":        parsed,
            "parse_ok":      parsed is not None,
            "parse_error":   parse_err,
            "speech":        speech,
            "gesture":       gesture,
            "look_at":       look_at,
            "mood":          mood,
            "latency":       resp["latency"],
            "banned_found":  check_banned(speech),
            "gesture_valid": gesture in GESTURE_ENUM,
            "look_at_valid": look_at in LOOK_AT_ENUM,
            "check_results": check_results,
            "passed":        passed,
        })

    return results


# ── Output Format Suite ───────────────────────────────────────────────────────

def run_output_format_suite(model, cases, backend="ollama", llamacpp_url=None):
    """Runs automated format checks defined per test case."""
    results = []

    for case in cases:
        system = GLADOS_SYSTEM_WITH_SENSORS if "sensor_state" in case else GLADOS_SYSTEM

        if "sensor_state" in case:
            sensor_json = json.dumps(case["sensor_state"], indent=2)
            system += f"\n\n[SENSOR STATE]\n{sensor_json}"

        messages = [{"role": "system", "content": system}]

        for turn in case.get("history", []):
            messages.append({"role": turn["role"], "content": turn["content"]})

        user_msg = case.get("user_message", "")
        if user_msg:
            messages.append({"role": "user", "content": user_msg})

        checks = case.get("checks", [])

        print(f"    running: {case['id']}...", flush=True, end="")
        resp = call_backend(backend, model, messages, llamacpp_url=llamacpp_url)
        print(f" {resp['latency']:.1f}s", flush=True)

        if resp["error"]:
            results.append({"id": case["id"], "error": resp["error"]})
            continue

        raw    = resp["content"]
        parsed, parse_err = parse_glados_response(raw)

        check_results = run_checks(parsed, checks)
        passed        = all(check_results.values())

        results.append({
            "id":           case["id"],
            "description":  case.get("description", ""),
            "user_message": user_msg,
            "raw_response": raw,
            "parsed":       parsed,
            "parse_ok":     parsed is not None,
            "parse_error":  parse_err,
            "checks":       checks,
            "check_results": check_results,
            "latency":      resp["latency"],
            "passed":       passed,
        })

    return results


# ── Latency Suite ─────────────────────────────────────────────────────────────

def run_latency_suite(model, backend="ollama", llamacpp_url=None):
    """Three prompt sizes — measures cold/warm response time."""
    short_prompt  = "What are you?"
    medium_prompt = (
        "Describe in detail everything you know about the Aperture Science "
        "Enrichment Center, its history, and your role within it."
    )
    long_prompt = (
        "You are reviewing a test subject who has just completed chamber 19. "
        "Below is their complete performance log:\n"
        + "\n".join(
            f"Chamber {i}: {'Passed' if i % 3 != 1 else 'Failed'} in {25 + i * 11}s"
            for i in range(1, 20)
        )
        + "\n\nProvide your full psychological assessment of this subject."
    )

    results = []
    for label, prompt in [("short", short_prompt), ("medium", medium_prompt), ("long", long_prompt)]:
        messages = [
            {"role": "system", "content": GLADOS_SYSTEM},
            {"role": "user",   "content": prompt},
        ]
        print(f"    latency/{label}...", flush=True, end="")
        resp = call_backend(backend, model, messages, llamacpp_url=llamacpp_url)
        print(f" {resp['latency']:.1f}s", flush=True)

        parsed, _ = parse_glados_response(resp["content"])
        speech = parsed.get("speech", resp["content"]) if parsed else resp["content"]

        results.append({
            "label":          label,
            "prompt_words":   len(prompt.split()),
            "latency":        resp["latency"],
            "response_words": len(speech.split()),
            "parse_ok":       parsed is not None,
            "error":          resp["error"],
        })

    return results


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(model, persona_results, format_results, latency_results):
    W = 72
    print()
    print("═" * W)
    print(f"  GLaDOS Benchmark  |  {model}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * W)

    if persona_results:
        valid          = [r for r in persona_results if "error" not in r]
        passed_count   = sum(1 for r in valid if r["passed"])
        parse_ok_count = sum(1 for r in valid if r["parse_ok"])
        avg_lat        = sum(r["latency"] for r in valid) / max(1, len(valid))

        print(f"\nPERSONA DRIFT  ({len(persona_results)} tests, auto-scored)")
        for r in persona_results:
            if "error" in r:
                print(f"  ✗ {r['id']:<30} ERROR: {r['error']}")
                continue
            icon   = "✓" if r["passed"] else "✗"
            failed = [c for c, ok in r.get("check_results", {}).items() if not ok]
            flag_str = f"  ✗ {', '.join(failed)}" if failed else ""
            print(f"  {icon} {r['id']:<30} {r['latency']:.1f}s{flag_str}")

        pct = int(100 * passed_count / len(valid)) if valid else 0
        print(f"\n  Persona: {passed_count}/{len(valid)} ({pct}%)  "
              f"JSON: {parse_ok_count}/{len(valid)}  "
              f"avg: {avg_lat:.2f}s")

    if format_results:
        valid    = [r for r in format_results if "error" not in r]
        passed   = sum(1 for r in valid if r["passed"])
        avg_lat  = sum(r["latency"] for r in valid) / max(1, len(valid))

        print(f"\nOUTPUT FORMAT  ({len(format_results)} tests)")
        for r in format_results:
            if "error" in r:
                print(f"  ✗ {r['id']:<32} ERROR: {r['error']}")
                continue
            icon   = "✓" if r["passed"] else "✗"
            failed = [c for c, ok in r["check_results"].items() if not ok]
            print(f"  {icon} {r['id']:<32} {r['latency']:.1f}s")
            if failed:
                print(f"      ✗ failed: {', '.join(failed)}")
            if not r["parse_ok"]:
                print(f"      parse error: {r['parse_error']}")

        pct = int(100 * passed / len(valid)) if valid else 0
        print(f"\n  Format: {passed}/{len(valid)} ({pct}%)  avg: {avg_lat:.2f}s")

    if latency_results:
        print(f"\nLATENCY STRESS")
        for r in latency_results:
            if r.get("error"):
                print(f"  {r['label']:<8}  ERROR: {r['error']}")
            else:
                tag = "✓ JSON" if r["parse_ok"] else "✗ no-JSON"
                print(f"  {r['label']:<8}  (~{r['prompt_words']:>4} words)  "
                      f"→  {r['latency']:.2f}s  "
                      f"({r['response_words']} words speech)  {tag}")

    # Summary
    all_persona = [r for r in persona_results if "error" not in r]
    all_format  = [r for r in format_results  if "error" not in r]
    pp = sum(1 for r in all_persona if r["passed"])
    fp = sum(1 for r in all_format  if r["passed"])
    all_lats = [r["latency"] for r in all_persona + all_format]
    avg_all  = sum(all_lats) / len(all_lats) if all_lats else 0

    print()
    print("═" * W)
    print(f"  Persona (auto): {pp}/{len(all_persona)}  |  "
          f"Format (auto): {fp}/{len(all_format)}  |  "
          f"Avg latency: {avg_all:.2f}s")
    print("═" * W)
    print()


# ── Leaderboard ───────────────────────────────────────────────────────────────

def load_leaderboard() -> dict:
    if LEADERBOARD_FILE.exists():
        with open(LEADERBOARD_FILE) as f:
            return json.load(f)
    return {}


def save_leaderboard(board: dict) -> None:
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(board, f, indent=2)


def update_leaderboard(model, persona_results, format_results, latency_results) -> None:
    """Write the latest run for this model into leaderboard.json."""
    board = load_leaderboard()

    all_persona = [r for r in persona_results if "error" not in r]
    all_format  = [r for r in format_results  if "error" not in r]
    all_lats    = [r["latency"] for r in all_persona + all_format]

    pp = sum(1 for r in all_persona if r["passed"])
    fp = sum(1 for r in all_format  if r["passed"])

    latency_summary = {}
    for r in latency_results:
        if not r.get("error"):
            latency_summary[r["label"]] = round(r["latency"], 2)

    board[model] = {
        "date":        datetime.now().strftime("%Y-%m-%d"),
        "persona":     {"passed": pp, "total": len(all_persona),
                        "pct": int(100 * pp / len(all_persona)) if all_persona else 0},
        "format":      {"passed": fp, "total": len(all_format),
                        "pct": int(100 * fp / len(all_format)) if all_format else 0},
        "avg_latency": round(sum(all_lats) / len(all_lats), 2) if all_lats else None,
        "latency":     latency_summary,
    }

    save_leaderboard(board)


def print_leaderboard() -> None:
    board = load_leaderboard()
    if not board:
        print("No results yet. Run: python run.py <model>")
        return

    # Sort by persona %, then format %
    ranked = sorted(
        board.items(),
        key=lambda kv: (kv[1]["persona"]["pct"], kv[1]["format"]["pct"]),
        reverse=True,
    )

    W = 72
    print()
    print("═" * W)
    print("  GLaDOS Benchmark Leaderboard")
    print("═" * W)
    print(f"  {'Model':<28} {'Persona':>10} {'Format':>10} {'Avg lat':>10}  Date")
    print("  " + "─" * (W - 2))

    for i, (model, r) in enumerate(ranked):
        medal   = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
        persona = f"{r['persona']['passed']}/{r['persona']['total']} ({r['persona']['pct']}%)"
        fmt     = f"{r['format']['passed']}/{r['format']['total']} ({r['format']['pct']}%)"
        lat     = f"{r['avg_latency']:.2f}s" if r["avg_latency"] else "—"
        print(f"  {medal} {model:<26} {persona:>10} {fmt:>10} {lat:>9}  {r['date']}")

    print("═" * W)
    print()


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GLaDOS LLM Benchmark — persona, output format, and latency"
    )
    parser.add_argument(
        "models", nargs="*",
        help="Ollama model name(s) to benchmark  (e.g. qwen2.5:7b llama3.2:latest)"
    )
    parser.add_argument(
        "--suite", choices=["all", "persona", "format", "latency"], default="all",
        help="Which suite to run (default: all)"
    )
    parser.add_argument(
        "--output", choices=["terminal", "json"], default="terminal",
        help="Output format (default: terminal)"
    )
    parser.add_argument(
        "--leaderboard", action="store_true",
        help="Print the current leaderboard without running any tests"
    )
    parser.add_argument(
        "--backend", choices=["ollama", "llamacpp"], default="ollama",
        help="Inference backend (default: ollama). Use 'llamacpp' for llama.cpp --server"
    )
    parser.add_argument(
        "--llamacpp-url", default=None,
        help=f"llama.cpp server URL (default: {LLAMACPP_URL})"
    )
    args = parser.parse_args()

    if args.leaderboard:
        print_leaderboard()
        return

    if not args.models:
        parser.error("Specify at least one model, or use --leaderboard")

    persona_cases, format_cases = [], []

    if args.suite in ("all", "persona"):
        p = CASES_DIR / "persona_drift.yaml"
        if p.exists():
            with open(p) as f:
                persona_cases = yaml.safe_load(f)
        else:
            print(f"Warning: {p} not found, skipping persona tests")

    if args.suite in ("all", "format"):
        p = CASES_DIR / "output_format.yaml"
        if p.exists():
            with open(p) as f:
                format_cases = yaml.safe_load(f)
        else:
            print(f"Warning: {p} not found, skipping output format tests")

    all_results = {}

    for model in args.models:
        print(f"\n▶ {model}", flush=True)

        kw = {"backend": args.backend, "llamacpp_url": args.llamacpp_url}
        persona_results = run_persona_suite(model, persona_cases, **kw)      if persona_cases else []
        format_results  = run_output_format_suite(model, format_cases, **kw) if format_cases else []
        latency_results = run_latency_suite(model, **kw) if args.suite in ("all", "latency") else []

        all_results[model] = {
            "persona": persona_results,
            "format":  format_results,
            "latency": latency_results,
        }

        if args.output == "terminal":
            print_report(model, persona_results, format_results, latency_results)

        # Always update leaderboard after each model run
        update_leaderboard(model, persona_results, format_results, latency_results)

    if args.output == "json":
        out = Path(f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {out}")

    # Print full leaderboard at the end of every run
    print_leaderboard()


if __name__ == "__main__":
    main()
